extern crate ffmpeg_the_third as ffmpeg;

use ab_glyph::{Font, FontArc, PxScale};
use clap::Parser;
use image::{ImageBuffer, ImageReader, Rgb};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rkod::{
    cv::FrameExtractor,
    od::RknnAppContext,
    read_lines,
    upload::{UpError, UploaderWorker},
};

use std::fs::File;
use std::io::{self, Error, ErrorKind};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info};

use ffmpeg::{format, media};

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

fn draw_results_on_image(
    img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    results: &[(String, f32, [f32; 4])],
) -> io::Result<()> {
    let font_data = include_bytes!("../assets/DejaVuSans.ttf");
    let font = FontArc::try_from_slice(font_data).unwrap();
    let scale = PxScale::from(20.0);

    for (label, prob, bbox) in results {
        let (x, y, w, h) = (
            bbox[0] as i32,
            bbox[1] as i32,
            (bbox[2] - bbox[0]) as i32,
            (bbox[3] - bbox[1]) as i32,
        );
        draw_hollow_rect_mut(
            img,
            Rect::at(x, y).of_size(w as u32, h as u32),
            Rgb([0, 255, 0]),
        );
        draw_text_mut(
            img,
            Rgb([0, 255, 0]),
            x,
            y - 15,
            scale,
            &font,
            &format!("{}: {:.2}", label, prob),
        );
    }
    Ok(())
}

// Placeholder for obtaining the current frame from the FrameExtractor
fn get_current_frame(
    frame_extractor: &FrameExtractor,
) -> io::Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    // Implement the correct method to retrieve the current frame
    // For now, returning an error to indicate this is a placeholder
    Err(io::Error::new(ErrorKind::Other, "Method not implemented"))
}

fn main() -> io::Result<()> {
    tracing_subscriber::fmt()
        .compact()
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

                    // Draw results on the image and save
                    let mut img = get_current_frame(&frame_extractor)?;
                    draw_results_on_image(&mut img, &results)?;
                    let _ = img.save("output_with_results.png");
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

        let total_inferences = 1; // Total number of inferences
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

                    // Draw results on the image and save
                    let img_bytes_vec =
                        Arc::try_unwrap(img_bytes).unwrap_or_else(|arc| (*arc).clone());
                    let mut img = ImageBuffer::from_raw(640, 480, img_bytes_vec).unwrap();
                    draw_results_on_image(&mut img, &results).unwrap();
                    img.save(format!("output_with_results_{}.png", i + 1))
                        .unwrap();
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
