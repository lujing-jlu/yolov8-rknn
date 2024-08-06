use crate::post_process::*;
use crate::{
    _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR, _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
    _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR, _rknn_tensor_format_RKNN_TENSOR_NCHW,
    _rknn_tensor_format_RKNN_TENSOR_NHWC, _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC,
    _rknn_tensor_type_RKNN_TENSOR_INT8, _rknn_tensor_type_RKNN_TENSOR_UINT8, dump_tensor_attr,
    rknn_context, rknn_init, rknn_input, rknn_input_output_num, rknn_inputs_set, rknn_output,
    rknn_outputs_get, rknn_outputs_release, rknn_query, rknn_run, rknn_tensor_attr,
};
use libc::c_void;
use std::{
    collections::HashSet,
    fs::File,
    io::{self, Read, Result},
    mem::size_of,
    ptr::null_mut,
};
use tracing::{error, info};
// const OBJ_NAME_MAX_SIZE: u8 = 64;
const OBJ_NUMB_MAX_SIZE: i32 = 128;
const OBJ_CLASS_NUM: i32 = 80;
const NMS_THRESH: f32 = 0.45;
const BOX_THRESH: f32 = 0.25;
const PROB_THRESHOLD: f32 = 0.2;
const MAP_SIZE: [[i32; 2]; 3] = [[80, 80], [40, 40], [20, 20]];
const MASK_NUM: i32 = 32;
const STRIDES: [i32; 3] = [8, 16, 32];
const SEG_WIDTH: i32 = 160;
const SEG_HEIGHT: i32 = 160;
const INPUT_WIDTH: i32 = 640;
const INPUT_HEIGHT: i32 = 640;
const HEAD_NUM: i32 = 3;
// #[derive(Debug, Default, Clone, Copy)]
// struct ImageRect {
//     left: i32,
//     top: i32,
//     right: i32,
//     bottom: i32,
// }

#[derive(Debug, Default, Clone)]
pub struct DetectRect {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    class_id: Option<usize>,
    score: f32,
    mask: Vec<f32>,
}

impl DetectRect {
    pub fn new() -> Self {
        DetectRect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 0.0,
            ymax: 0.0,
            class_id: None,
            score: 0.0,
            mask: vec![],
        }
    }
}

#[derive(Debug, Default, Clone)]
struct ObjectDetection {
    // rect: ImageRect,
    prob: f32,
    cls_id: i32,
    f_box: [f32; 4],
    mask: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct ObjectDetectList {
    count: i32,
    results: Vec<ObjectDetection>,
}

impl ObjectDetectList {
    pub fn new(
        class_id: &Vec<i32>,
        obj_probs: &Vec<f32>,
        order: &Vec<usize>,
        filter_boxes: &Vec<[f32; 4]>,
        masks: &Vec<Vec<u8>>,
    ) -> Result<Self> {
        if class_id.len() != obj_probs.len() || order.len() != class_id.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "class_id, obj_probs and order should be the same length.",
            ));
        }
        let mut count = 0i32;
        let mut results: Vec<ObjectDetection> = Vec::new();
        for i in 0..class_id.len() {
            if count >= OBJ_NUMB_MAX_SIZE {
                break;
            }
            let n = order[i];
            if n == 0xffff {
                continue;
            }

            if obj_probs[n] < PROB_THRESHOLD {
                break;
            }

            let res = ObjectDetection {
                prob: obj_probs[n],
                cls_id: class_id[n],
                f_box: filter_boxes[n],
                mask: masks[n].clone(),
            };
            results.push(res);
            count += 1;
        }
        Ok(Self { count, results })
    }

    pub fn get_results(&self) -> Vec<(i32, f32, [f32; 4], Vec<u8>)> {
        self.results
            .iter()
            .map(|r| (r.cls_id, r.prob, r.f_box, r.mask.clone()))
            .collect::<Vec<_>>()
    }

    pub fn get_count(&self) -> i32 {
        self.count
    }
}

#[derive(Debug, Clone)]
pub struct RknnAppContext {
    rknn_ctx: rknn_context,
    io_num: rknn_input_output_num,
    input_attrs: Vec<rknn_tensor_attr>,
    output_attrs: Vec<rknn_tensor_attr>,
    model_channel: i32,
    model_width: i32,
    model_height: i32,
    is_quant: bool,
    mesh_grid: Vec<f32>,
}

impl RknnAppContext {
    pub fn new() -> Self {
        let rknn_ctx = 0u64;
        let io_num = rknn_input_output_num {
            n_input: 0u32,
            n_output: 0u32,
        };
        let (input_attrs, output_attrs) = (Vec::new(), Vec::new());
        let (model_channel, model_width, model_height) = (0i32, 0i32, 0i32);
        let is_quant = false;
        let mesh_grid = generate_mesh_grid();
        Self {
            rknn_ctx,
            io_num,
            input_attrs,
            output_attrs,
            model_channel,
            model_width,
            model_height,
            is_quant,
            mesh_grid,
        }
    }

    pub fn width(&self) -> u32 {
        self.model_width as _
    }

    pub fn height(&self) -> u32 {
        self.model_height as _
    }

    pub fn init_model(&mut self, path: &str) -> Result<()> {
        let mut ctx: rknn_context = 0;

        let mut model = File::open(path)?;
        let mut model_buf: Vec<u8> = Vec::new();

        let model_len = model.read_to_end(&mut model_buf)?;
        let model = model_buf.as_mut_ptr() as *mut c_void;

        let ret = unsafe { rknn_init(&mut ctx, model, model_len as u32, 0, null_mut()) };

        // info!("model_buf: {model_buf:?}");
        drop(model_buf);

        if ret < 0 {
            error!("Failed to init rknn. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Failed to init rknn",
            ));
        }

        // Get Model Input Output Number
        let mut io_num = rknn_input_output_num {
            n_input: 0u32,
            n_output: 0u32,
        };

        let ret = unsafe {
            rknn_query(
                ctx,
                _rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut c_void,
                size_of::<rknn_input_output_num>() as u32,
            )
        };
        if ret < 0 {
            error!("Failed to query rknn. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Failed to query rknn",
            ));
        }
        info!(
            "Model input num: {}, output num: {}",
            io_num.n_input, io_num.n_output
        );

        // Get Model Input Info
        info!("Input tensors:");
        let mut input_attrs: Vec<rknn_tensor_attr> = Vec::new();

        for i in 0..io_num.n_input {
            let mut attr = rknn_tensor_attr {
                index: i,
                n_dims: 0,
                dims: [0; 16],
                name: [0; 256],
                n_elems: 0,
                size: 0,
                fmt: 0,
                type_: 0,
                qnt_type: 0,
                fl: 0,
                zp: 0,
                scale: 0.0,
                w_stride: 0,
                size_with_stride: 0,
                pass_through: 0,
                h_stride: 0,
            };
            let ret = unsafe {
                rknn_query(
                    ctx,
                    _rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    size_of::<rknn_tensor_attr>() as u32,
                )
            };
            if ret != 0 {
                error!("Failed to query rknn. Error code: {ret}");
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Failed to query rknn",
                ));
            }
            dump_tensor_attr(&attr);
            input_attrs.push(attr);
        }

        // Get Model Output Info
        // info!("Output tensor");
        let mut output_attrs: Vec<rknn_tensor_attr> = Vec::new();
        for i in 0..io_num.n_output {
            let mut attr = rknn_tensor_attr {
                index: i,
                n_dims: 0,
                dims: [0; 16],
                name: [0; 256],
                n_elems: 0,
                size: 0,
                fmt: 0,
                type_: 0,
                qnt_type: 0,
                fl: 0,
                zp: 0,
                scale: 0.0,
                w_stride: 0,
                size_with_stride: 0,
                pass_through: 0,
                h_stride: 0,
            };
            let ret = unsafe {
                rknn_query(
                    ctx,
                    _rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    size_of::<rknn_tensor_attr>() as u32,
                )
            };
            if ret != 0 {
                error!("Failed to query rknn. Error code: {ret}");
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Failed to query rknn",
                ));
            }
            // dump_tensor_attr(&attr);
            output_attrs.push(attr);
        }
        // Set to context
        self.rknn_ctx = ctx;
        if output_attrs[0].qnt_type == _rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
            && output_attrs[0].type_ == _rknn_tensor_type_RKNN_TENSOR_INT8
        {
            self.is_quant = true;
        } else {
            self.is_quant = false;
        }

        self.io_num = io_num;
        self.input_attrs = input_attrs.clone();
        self.output_attrs = output_attrs;

        if input_attrs[0].fmt == _rknn_tensor_format_RKNN_TENSOR_NCHW {
            info!("model is NCHW input fmt");
            self.model_channel = input_attrs[0].dims[1] as i32;
            self.model_height = input_attrs[0].dims[2] as i32;
            self.model_width = input_attrs[0].dims[3] as i32;
        } else {
            info!("model is NHWC input fmt");
            self.model_height = input_attrs[0].dims[1] as i32;
            self.model_width = input_attrs[0].dims[2] as i32;
            self.model_channel = input_attrs[0].dims[3] as i32;
        }
        info!(
            "model input height={}, width={}, channel={}",
            self.model_height, self.model_width, self.model_channel
        );
        Ok(())
    }

    pub fn inference_model(&self, img: &[u8]) -> Result<ObjectDetectList> {
        let img_buf = img.as_ptr() as *mut c_void;
        let mut inputs: Vec<rknn_input> = Vec::new();
        for n in 0..self.io_num.n_input {
            let input = rknn_input {
                index: n,
                size: (self.model_width * self.model_height * self.model_channel) as u32,
                type_: _rknn_tensor_type_RKNN_TENSOR_UINT8,
                fmt: _rknn_tensor_format_RKNN_TENSOR_NHWC,
                // pass_through - if 1 directly pass image buff to rknn node else if 0 do conversion first.
                pass_through: 0,
                buf: img_buf,
            };
            inputs.push(input);
        }

        // info!("Setting rknn inputs...n_input: {}", self.io_num.n_input);

        let ret =
            unsafe { rknn_inputs_set(self.rknn_ctx, self.io_num.n_input, inputs.as_mut_ptr()) };

        if ret < 0 {
            error!("Failed to set rknn input. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Failed to set rknn input",
            ));
        }

        // info!("Running rknn...");

        let ret = unsafe { rknn_run(self.rknn_ctx, null_mut()) };

        if ret < 0 {
            error!("Failed to run rknn. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Failed to run rknn",
            ));
        }

        let mut outputs: Vec<rknn_output> = Vec::new();
        println!("n_output:{}", self.io_num.n_output);
        for i in 0..self.io_num.n_output {
            let output = rknn_output {
                index: i,
                want_float: !self.is_quant as u8,
                is_prealloc: 0,
                size: 0,
                buf: null_mut() as *mut c_void,
            };
            outputs.push(output);
        }

        // info!("Generating outputs...");

        let ret = unsafe {
            rknn_outputs_get(
                self.rknn_ctx,
                self.io_num.n_output,
                outputs.as_mut_ptr(),
                null_mut(),
            )
        };

        if ret < 0 {
            error!("Failed to get rknn outputs. Error code: {ret}");
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Failed to get rknnoutputs",
            ));
        }

        // Post process
        // let mut valid_count = 0;

        // info!("Post process begins...");

        let mut filter_boxes: Vec<[f32; 4]> = Vec::new();
        let mut obj_probs: Vec<f32> = Vec::new();
        let mut class_id: Vec<i32> = Vec::new();
        let mut masks: Vec<Vec<Vec<u8>>> = Vec::new();

        // 确保n_output至少为8
        if outputs.len() < 8 {
            error!(
                "Unexpected number of outputs. Expected at least 8, got {}",
                outputs.len()
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unexpected number of outputs",
            ));
        }

        // 存储缩放因子和零点
        let mut out_scales: Vec<f32> = Vec::new();
        let mut out_zps: Vec<i32> = Vec::new();

        for i in 0..self.io_num.n_output {
            out_scales.push(self.output_attrs[i as usize].scale);
            out_zps.push(self.output_attrs[i as usize].zp);
        }

        // 创建一个 Vec<*mut i8>，长度为 io_num_n_output
        let mut pblob: Vec<*mut i8> = vec![std::ptr::null_mut(); self.io_num.n_output as usize];

        // 遍历并设置每个 pblob 元素
        for (i, output) in outputs.iter().enumerate() {
            pblob[i] = output.buf as *mut i8;
        }

        let mut grid_index: i32 = -2;
        let mut cls_max: f32 = 0.0;
        let mut cls_index: i32 = 0;

        self.get_conv_detection_result(pblob, out_zps, out_scales);
    }

    fn get_conv_detection_result(
        &mut self,
        pblob: Vec<*mut i8>,
        qnt_zp: Vec<i32>,
        qnt_scale: Vec<f32>,
    ) {
        let mut detect_rects: Vec<DetectRect> = Vec::new();
        let mut seg_mask: Vec<Vec<[u8; 3]>> =
            vec![vec![[0, 0, 0]; SEG_WIDTH as usize]; SEG_HEIGHT as usize];

        let mut ret = 0;
        if self.mesh_grid.is_empty() {
            ret = self.generate_mesh_grid();
        }

        let mut grid_index = 0;
        let mut xmin = 0.0;
        let mut ymin = 0.0;
        let mut xmax = 0.0;
        let mut ymax = 0.0;
        let mut cls_val = 0.0;
        let mut cls_max = 0.0;
        let mut cls_index = 0;

        let mut quant_zp_cls = 0;
        let mut quant_zp_reg = 0;
        let mut quant_zp_msk = 0;
        let mut quant_zp_seg = 0;
        let mut quant_scale_cls = 0.0;
        let mut quant_scale_reg = 0.0;
        let mut quant_scale_msk = 0.0;
        let mut quant_scale_seg = 0.0;

        for i in 0..3 {
            let reg = outputs[7 + i].buf as *const i8;
            let cls = outputs[0 + i].buf as *const i8;
            let msk = outputs[3 + i].buf as *const i8;

            // 获取量化参数
            let quant_zp_reg = self.output_attrs[7 + i].zp;
            let quant_zp_cls = self.output_attrs[0 + i].zp;
            let quant_zp_msk = self.output_attrs[3 + i].zp;

            let quant_scale_reg = self.output_attrs[7 + i].scale;
            let quant_scale_cls = self.output_attrs[0 + i].scale;
            let quant_scale_msk = self.output_attrs[3 + i].scale;

            let mut sfsum: f32;
            let mut locval: f32;
            let mut locvaltemp: f32;
            let mut reg_deq = [0.0f32; 16];

            for h in 0..MAP_SIZE[i][0] {
                for w in 0..MAP_SIZE[i][1] {
                    grid_index += 2;

                    if 1 == OBJ_CLASS_NUM {
                        cls_max = sigmoid(deqnt2f32(
                            unsafe {
                                *cls.offset(
                                    (0 * MAP_SIZE[i][0] * MAP_SIZE[i][1] + h * MAP_SIZE[i][1] + w)
                                        as isize,
                                ) as i32
                            },
                            quant_zp_cls,
                            quant_scale_cls,
                        ));
                        cls_index = 0;
                    } else {
                        for cl in 0..OBJ_CLASS_NUM {
                            let cls_val = unsafe {
                                *cls.offset(
                                    (cl * MAP_SIZE[i][0] * MAP_SIZE[i][1] + h * MAP_SIZE[i][1] + w)
                                        as isize,
                                )
                            };
                            if cl == 0 {
                                cls_max = cls_val as f32;
                                cls_index = cl;
                            } else {
                                if cls_val > cls_max as i8 {
                                    cls_max = cls_val as f32;
                                    cls_index = cl;
                                }
                            }
                        }
                        cls_max = sigmoid(deqnt2f32(cls_max as i32, quant_zp_cls, quant_scale_cls));
                    }

                    if cls_max > BOX_THRESH {
                        let mut reg_dfl = Vec::new();
                        for lc in 0..4 {
                            sfsum = 0.0;
                            locval = 0.0;
                            for df in 0..16 {
                                locvaltemp = (deqnt2f32(
                                    unsafe {
                                        (*reg.offset(
                                            ((lc * 16 + df) * MAP_SIZE[i][0] * MAP_SIZE[i][1]
                                                + h * MAP_SIZE[i][1]
                                                + w)
                                                as isize,
                                        ))
                                        .into()
                                    },
                                    quant_zp_reg,
                                    quant_scale_reg,
                                ))
                                .exp();
                                reg_deq[df as usize] = locvaltemp;
                                sfsum += locvaltemp;
                            }
                            for df in 0..16 {
                                locvaltemp = reg_deq[df] / sfsum;
                                locval += locvaltemp * df as f32;
                            }
                            reg_dfl.push(locval);
                        }

                        let stride = STRIDES[i] as f32;
                        let xmin = (self.mesh_grid[grid_index as usize] - reg_dfl[0]) * stride;
                        let ymin = (self.mesh_grid[grid_index as usize + 1] - reg_dfl[1]) * stride;
                        let xmax = (self.mesh_grid[grid_index as usize] + reg_dfl[2]) * stride;
                        let ymax = (self.mesh_grid[grid_index as usize + 1] + reg_dfl[3]) * stride;

                        let xmin = xmin.max(0.0);
                        let ymin = ymin.max(0.0);
                        let xmax = xmax.min(self.model_width as f32);
                        let ymax = ymax.min(self.model_height as f32);

                        if xmin >= 0.0
                            && ymin >= 0.0
                            && xmax <= self.model_width as f32
                            && ymax <= self.model_height as f32
                        {
                            let mut temp = DetectRect::default();
                            temp.xmin = xmin / self.model_width as f32;
                            temp.ymin = ymin / self.model_height as f32;
                            temp.xmax = xmax / self.model_width as f32;
                            temp.ymax = ymax / self.model_height as f32;
                            temp.class_id = Some(cls_index as usize);
                            temp.score = cls_max;

                            for ms in 0..MASK_NUM {
                                temp.mask.push(deqnt2f32(
                                    unsafe {
                                        (*msk.offset(
                                            (ms * MAP_SIZE[i][0] * MAP_SIZE[i][1]
                                                + h * MAP_SIZE[i][1]
                                                + w)
                                                as isize,
                                        ))
                                        .into()
                                    },
                                    quant_zp_msk,
                                    quant_scale_msk,
                                ));
                            }
                            println!("{:?}", temp.mask);
                            detect_rects.push(temp);
                        }
                    }
                }
            }
        }

        detect_rects.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let detection_rects = self.process_detections(&mut detect_rects, outputs);

        if obj_probs.len() == 0 {
            // warn!("No object detected");
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "No object detected",
            ));
        }

        let class_set: HashSet<i32> = HashSet::from_iter(class_id.clone().into_iter());

        // info!("obj_probs: {obj_probs:?}");
        // info!("class_id: {class_id:?}");

        let mut order = (0..obj_probs.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| obj_probs[b].total_cmp(&obj_probs[a]));

        // nms

        for &c in class_set.iter() {
            nms(&filter_boxes, &class_id, &mut order, c);
        }

        // nms end

        let od_result = match ObjectDetectList::new(
            &class_id,
            &obj_probs,
            &order,
            &filter_boxes,
            &vec![vec![]],
        ) {
            Ok(r) => r,
            Err(e) => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, e));
            }
        };

        // info!("Rknn running: context is now {}", self.rknn_ctx);
        let _ = unsafe {
            rknn_outputs_release(self.rknn_ctx, self.io_num.n_output, outputs.as_mut_ptr())
        };

        Ok(od_result)
    }

    fn process_detections(
        &self,
        detect_rects: &mut Vec<DetectRect>,
        seg: Vec<i8>,
        color_lists: Vec<[u8; 3]>,
        seg_mask: &mut Vec<Vec<[u8; 3]>>,
        outputs: Vec<rknn_output>,
    ) -> Vec<f32> {
        let mut detection_rects = Vec::new();

        let mut seg_mask: Vec<Vec<[u8; 3]>> = vec![vec![[0, 0, 0]; SEG_WIDTH]; SEG_HEIGHT];

        let seg = outputs[6].buf as *const i8;
        quant_zp_seg = qnt_zp[6];
        quant_scale_seg = qnt_scale[6];
        // 首先处理检测框
        for i in 0..detect_rects.len() {
            let xmin1 = detect_rects[i].xmin;
            let ymin1 = detect_rects[i].ymin;
            let xmax1 = detect_rects[i].xmax;
            let ymax1 = detect_rects[i].ymax;
            let class_id = detect_rects[i].class_id;
            let score = detect_rects[i].score;

            if let Some(class_id_val) = class_id {
                // 将检测结果按照class_id、score、xmin1、ymin1、xmax1、ymax1的格式存放在vector<float>中
                detection_rects.push(class_id_val as f32);
                detection_rects.push(score);
                detection_rects.push(xmin1);
                detection_rects.push(ymin1);
                detection_rects.push(xmax1);
                detection_rects.push(ymax1);

                for j in (i + 1)..detect_rects.len() {
                    let xmin2 = detect_rects[j].xmin;
                    let ymin2 = detect_rects[j].ymin;
                    let xmax2 = detect_rects[j].xmax;
                    let ymax2 = detect_rects[j].ymax;
                    let iou = iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2);
                    if iou > NMS_THRESH {
                        detect_rects[j].class_id = None;
                    }
                }
            }
        }

        // 然后处理分割掩码
        for rect in detect_rects.iter() {
            if let Some(class_id) = rect.class_id {
                let left = (rect.xmin * SEG_WIDTH as f32 + 0.5) as usize;
                let top = (rect.ymin * SEG_HEIGHT as f32 + 0.5) as usize;
                let right = (rect.xmax * SEG_WIDTH as f32 + 0.5) as usize;
                let bottom = (rect.ymax * SEG_HEIGHT as f32 + 0.5) as usize;

                for h in top..bottom {
                    for w in left..right {
                        let mut seg_sum = 0.0;
                        for s in 0..MASK_NUM {
                            seg_sum += rect.mask[s as usize]
                                * deqnt2f32(
                                    (seg[(s * SEG_WIDTH * SEG_HEIGHT as i32
                                        + h as i32 * SEG_WIDTH as i32
                                        + w as i32)
                                        as usize] as usize)
                                        .try_into()
                                        .unwrap(),
                                    quant_zp_seg,
                                    quant_scale_seg,
                                );
                        }

                        if 1.0 / (1.0 + (-seg_sum).exp()) > 0.5 {
                            seg_mask[h][w] = color_lists[class_id / 10];
                        }
                    }
                }
            }
        }

        detection_rects
    }
}

fn generate_mesh_grid() -> Vec<f32> {
    let mut mesh_grid = Vec::new();
    for index in 0..3 {
        for i in 0..MAP_SIZE[index][0] {
            for j in 0..MAP_SIZE[index][1] {
                mesh_grid.push(j as f32 + 0.5);
                mesh_grid.push(i as f32 + 0.5);
            }
        }
    }

    // println!("=== yolov8 mesh_grid Generate success!");

    mesh_grid
}
fn qnt_f32_to_affine(threshold: f32, score_zp: i32, score_scale: f32) -> i8 {
    let dst_val = (threshold / score_zp as f32) + score_scale as f32;
    match (dst_val <= -128.0, dst_val >= 127.0) {
        (true, _) => -128i8,
        (false, true) => 127i8,
        (false, false) => dst_val as i8,
    }
}

fn compute_dfl(tensor: Vec<f32>, dfl_len: usize) -> [f32; 4] {
    let mut draw_box = [0.0f32; 4];
    for b in 0..4 as usize {
        let mut exp_t: Vec<f32> = Vec::new();
        let mut exp_sum = 0.0f32;
        let mut acc_sum = 0.0f32;
        for i in 0..dfl_len {
            let expon = tensor[i + b * dfl_len].exp();
            exp_t.push(expon);
            exp_sum += expon;
        }

        for i in 0..dfl_len {
            acc_sum += exp_t[i] / exp_sum * (i as f32);
        }
        draw_box[b] = acc_sum;
    }
    draw_box
}

pub fn nms(
    filter_boxes: &Vec<[f32; 4]>,
    class_id: &Vec<i32>,
    order: &mut Vec<usize>,
    filter_id: i32,
) {
    for i in 0..class_id.len() {
        if (order[i] == 0xffff) || (class_id[i] != filter_id) {
            continue;
        }
        let n = order[i];
        for j in (i + 1)..class_id.len() {
            let m = order[j];
            if m == 0xffff || class_id[i] != filter_id {
                continue;
            }
            let iou = cal_overlap([
                filter_boxes[n][0],
                filter_boxes[n][1],
                filter_boxes[n][2],
                filter_boxes[n][3],
                filter_boxes[m][0],
                filter_boxes[m][1],
                filter_boxes[m][2],
                filter_boxes[m][3],
            ]);
            if iou > NMS_THRESH {
                order[j] = 0xffff;
            }
        }
    }
}

/// mxy: [xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1]
fn cal_overlap(mxy: [f32; 8]) -> f32 {
    let xmax = if mxy[2] >= mxy[6] { mxy[6] } else { mxy[2] };
    let xmin = if mxy[0] >= mxy[4] { mxy[0] } else { mxy[4] };
    let ymax = if mxy[3] >= mxy[7] { mxy[7] } else { mxy[3] };
    let ymin = if mxy[1] >= mxy[5] { mxy[1] } else { mxy[5] };
    let w = if xmax - xmin + 1. > 0. {
        xmax - xmin + 1.
    } else {
        0.
    };
    let h = if ymax - ymin + 1. > 0. {
        ymax - ymin + 1.
    } else {
        0.
    };
    let i = w * h;
    let u = (mxy[2] - mxy[0] + 1.) * (mxy[3] - mxy[1] + 1.)
        + (mxy[6] - mxy[4] + 1.) * (mxy[7] - mxy[5] + 1.)
        - i;
    if u <= 0. {
        0.
    } else {
        i / u
    }
}

// Function to convert raw mask data to a 2D array
fn convert_to_2d_array(data: &[u8], width: usize, height: usize) -> Vec<Vec<u8>> {
    let mut mask = vec![vec![0; width]; height];
    for y in 0..height {
        for x in 0..width {
            mask[y][x] = data[y * width + x];
        }
    }
    mask
}

pub fn deqnt2f32(qnt: i32, zp: i32, scale: f32) -> f32 {
    (qnt as f32 - zp as f32) * scale
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

pub fn fast_exp(x: f32) -> f32 {
    let v: u32 = (12102203.1616540672 * x + 1064807160.56887296) as u32;
    f32::from_bits(v)
}

pub fn iou(
    x_min1: f32,
    y_min1: f32,
    x_max1: f32,
    y_max1: f32,
    x_min2: f32,
    y_min2: f32,
    x_max2: f32,
    y_max2: f32,
) -> f32 {
    let x_min = x_min1.max(x_min2);
    let y_min = y_min1.max(y_min2);
    let x_max = x_max1.min(x_max2);
    let y_max = y_max1.min(y_max2);

    let inter_width = (x_max - x_min).max(0.0);
    let inter_height = (y_max - y_min).max(0.0);

    let inter = inter_width * inter_height;

    let area1 = (x_max1 - x_min1) * (y_max1 - y_min1);
    let area2 = (x_max2 - x_min2) * (y_max2 - y_min2);

    let total = area1 + area2 - inter;

    inter / total
}
