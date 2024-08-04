extern crate ffmpeg_the_third as ffmpeg;

use clap::Parser;
use image::ImageReader;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::io::{self, Error, ErrorKind};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{error, info};
use yolov8_rknn::{
    cv::FrameExtractor,
    od::RknnAppContext,
    read_lines,
    upload::{UpError, UploaderWorker},
};

use ffmpeg::{format, media};
// use tracing_subscriber::fmt::time::ChronoLocal;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model to use
    #[arg(short, long, default_value = "model/safety_hat.rknn")]
    model: String,

    /// Path to the input for inference. Could be image or video.
    #[arg(short, long)]
    input: String,

    /// Object detection results uploading switch.
    /// You don't need it unless you have deployed your own RESTful API.
    #[arg(short, long)]
    upload: bool,
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .compact()
        // .with_timer(ChronoLocal::new(String::from("[%F %T]")))
        .without_time()
        .with_target(false)
        .init();

    let args = Args::parse();
    let lines = read_lines("model/safety_hat.txt")?;
    let labels = lines.flatten().collect::<Vec<String>>();

    if args.input.starts_with("rtsp") {
        // Init ffmpeg
        ffmpeg::init()?;
        let mut ictx = format::input(&args.input)?;

        // Print detailed information about the input or output format,
        // such as duration, bitrate, streams, container, programs, metadata, side data, codec and time base.
        format::context::input::dump(&ictx, 0, Some(&args.input));

        // Find video stream from input context.
        let input = ictx
            .streams()
            .best(media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let video_stream_index = input.index();

        let mut frame_extractor = FrameExtractor::new(&input, [640, 480])?; // Adjust to your model's input size

        let upworker = if args.upload {
            let u = UploaderWorker::new();
            Some(u)
        } else {
            None
        };

        for (stream, packet) in ictx.packets().filter_map(|r| r.ok()) {
            // Find key frame
            if !packet.is_key() {
                continue;
            }
            if stream.index() == video_stream_index {
                frame_extractor.send_packet_to_decoder(&packet)?;
                let mut app_ctx = RknnAppContext::new();
                app_ctx.init_model(&args.model)?;
                if let Some(r) = frame_extractor.process_frames(&app_ctx)? {
                    let results = r
                        .into_iter()
                        .map(|(id, prob, f_box)| (labels[id as usize].clone(), prob, f_box))
                        .collect::<Vec<_>>();

                    if let Some(upload_worker) = upworker.as_ref() {
                        if let Err(UpError::ChannelError(e)) =
                            upload_worker.upload_odres(results.clone())
                        {
                            error!("Failed to send od result to UploadWorker channel: {e}");
                        }
                    }
                    info!("Object detected: {results:?}");
                } else {
                    info!("No object detected.");
                }
            }
        }
        frame_extractor.send_eof_to_decoder()?;
        frame_extractor.process_frames(&RknnAppContext::new())?;
    } else {
        // Read image raw bytes
        let reader = ImageReader::open(&args.input)?;
        let img = match reader.decode() {
            Ok(m) => m,
            Err(e) => {
                return Err(Error::new(ErrorKind::InvalidInput, e.to_string()));
            }
        };
        let img = img.resize_to_fill(
            640,
            480, // Adjust to your model's input size
            image::imageops::FilterType::Nearest,
        );

        let img_bytes = Arc::new(img.as_bytes().to_vec()); // Convert image to byte vector for thread safety
        let labels = Arc::new(labels);
        let model_path = Arc::new(args.model.clone());

        let start = Instant::now();

        let total_inferences = 1000; // Total number of inferences
        let thread_num = 6;
        let pool = ThreadPoolBuilder::new()
            .num_threads(thread_num)
            .build()
            .unwrap();

        // Create a vector of Mutex-wrapped RknnAppContext instances
        let models: Vec<_> = (0..thread_num)
            .map(|_| {
                let mut app_ctx = RknnAppContext::new();
                app_ctx
                    .init_model(&model_path)
                    .expect("Model initialization failed");
                Mutex::new(app_ctx)
            })
            .collect();

        let models = Arc::new(models);

        pool.scope(|s| {
            (0..total_inferences)
                .into_par_iter()
                .for_each_with(s, |s, i| {
                    let img_bytes = Arc::clone(&img_bytes);
                    let labels = Arc::clone(&labels);
                    let models = Arc::clone(&models);

                    // Get a model from the pool
                    let model = &models[i % thread_num];
                    let mut app_ctx = model.lock().unwrap();

                    let inference_start = Instant::now();
                    let od_results = app_ctx.inference_model(&img_bytes).unwrap();
                    let inference_duration = inference_start.elapsed();

                    let results = od_results
                        .get_results()
                        .into_iter()
                        .map(|(id, prob, f_box)| (labels[id as usize].clone(), prob, f_box))
                        .collect::<Vec<_>>();

                    info!(
                        "Inference {}: Results: {:?}, Time: {:?}",
                        i + 1,
                        results,
                        inference_duration
                    );
                });
        });

        let duration = start.elapsed();
        let throughput = total_inferences as f64 / duration.as_secs_f64();
        info!(
            "Total inference time: {:?}, Throughput: {:.2} images/sec",
            duration, throughput
        );
    }
    Ok(())
}
