use std::f32;
use std::vec::Vec;

const OBJ_NUMB_MAX_SIZE: i32 = 128;
const OBJ_CLASS_NUM: i32 = 80;
const NMS_THRESH: f32 = 0.45;
const BOX_THRESH: f32 = 0.25;
const PROB_THRESHOLD: f32 = 0.2;
const MAP_SIZE: [[usize; 2]; 3] = [[80, 80], [40, 40], [20, 20]];
const MASK_NUM: i32 = 32;
const STRIDES: [i32; 3] = [8, 16, 32];
const SEG_WIDTH: i32 = 160;
const SEG_HEIGHT: i32 = 160;
const INPUT_WIDTH: i32 = 640;
const INPUT_HEIGHT: i32 = 640;
const HEAD_NUM: usize = 3;

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

pub fn deqnt2f32(qnt: i32, zp: i32, scale: f32) -> f32 {
    (qnt as f32 - zp as f32) * scale
}

pub struct GetResultRectYolov8seg {
    head_num: usize,
    map_size: [[i32; 2]; 3],
    mesh_grid: Vec<f32>,
    strides: [i32; 3],
    input_width: usize,
    input_height: usize,
    class_num: usize,
    object_thresh: f32,
    nms_thresh: f32,
    mask_num: usize,
    seg_width: usize,
    seg_height: usize,
    color_lists: Vec<[u8; 3]>,
}

impl GetResultRectYolov8seg {
    pub fn new() -> Self {
        let map_size = [[80, 80], [40, 40], [20, 20]];
        let mesh_grid = generate_mesh_grid(map_size);
        let strides = [8, 16, 32];
        GetResultRectYolov8seg {
            head_num: 3,
            map_size,
            mesh_grid,
            strides,
            input_width: 640,
            input_height: 640,
            class_num: 80,
            object_thresh: 0.25,
            nms_thresh: 0.45,
            mask_num: 32,
            seg_width: 160,
            seg_height: 160,
            color_lists: vec![],
        }
    }

    pub fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + fast_exp(-x))
    }

    fn generate_mesh_grid(&mut self) -> i32 {
        let ret = 0;
        if HEAD_NUM == 0 {
            println!("=== yolov8 MeshGrid Generate failed!");
        }

        for index in 0..HEAD_NUM {
            for i in 0..MAP_SIZE[index][0] {
                for j in 0..MAP_SIZE[index][1] {
                    self.mesh_grid.push(j as f32 + 0.5);
                    self.mesh_grid.push(i as f32 + 0.5);
                }
            }
        }

        println!("=== yolov8 MeshGrid Generate success!");

        ret
    }

    fn get_conv_detection_result(
        &mut self,
        pblob: Vec<*mut i8>,
        qnt_zp: &Vec<i32>,
        qnt_scale: &Vec<f32>,
        detection_rects: &mut Vec<f32>,
        seg_mask: &mut Vec<Vec<[u8; 3]>>,
    ) -> i32 {
        let mut ret = 0;
        if self.mesh_grid.is_empty() {
            ret = self.generate_mesh_grid();
        }

        let mut grid_index = 0;
        // let mut cls_val = 0.0;
        let mut cls_max = 0.0;
        let mut cls_index = 0;

        // let mut reg: *mut i8 = std::ptr::null_mut();
        // let mut cls: *mut i8 = std::ptr::null_mut();
        // let mut msk: *mut i8 = std::ptr::null_mut();
        let mut detect_rects = Vec::new();

        for index in 0..HEAD_NUM {
            // 使用 unsafe 块处理指针

            let reg = pblob[7 + index];
            let cls = pblob[0 + index];
            let msk = pblob[3 + index];

            // 获取量化参数
            let quant_zp_reg: i32 = qnt_zp[7 + index];
            let quant_zp_cls: i32 = qnt_zp[0 + index];
            let quant_zp_msk: i32 = qnt_zp[3 + index];

            let quant_scale_reg: f32 = qnt_scale[7 + index];
            let quant_scale_cls: f32 = qnt_scale[0 + index];
            let quant_scale_msk: f32 = qnt_scale[3 + index];

            for h in 0..MAP_SIZE[index][0] {
                for w in 0..MAP_SIZE[index][1] {
                    let grid_index_0 = grid_index;
                    let grid_index_1 = grid_index + 1;
                    grid_index += 2;

                    if self.class_num == 1 {
                        unsafe {
                            cls_max = self.sigmoid(deqnt2f32(
                                (*cls.add(
                                    0 * MAP_SIZE[index][0] as usize * MAP_SIZE[index][1] as usize
                                        + h * MAP_SIZE[index][1] as usize
                                        + w,
                                ))
                                .into(),
                                quant_zp_cls,
                                quant_scale_cls,
                            ));
                        }
                        cls_index = 0;
                    } else {
                        for cl in 0..self.class_num {
                            let cls_val: f32;
                            unsafe {
                                cls_val = *cls.add(
                                    cl * MAP_SIZE[index][0] * MAP_SIZE[index][1]
                                        + h * MAP_SIZE[index][1]
                                        + w,
                                ) as f32;
                            }

                            if cl == 0 {
                                cls_max = cls_val;
                                cls_index = cl;
                            } else {
                                if cls_val > cls_max {
                                    cls_max = cls_val;
                                    cls_index = cl;
                                }
                            }
                        }
                        cls_max = self.sigmoid(deqnt2f32(
                            cls_max.round() as i32,
                            quant_zp_cls,
                            quant_scale_cls,
                        ));
                    }

                    if cls_max > self.object_thresh {
                        let mut reg_dfl = Vec::new();
                        for lc in 0..4 {
                            let mut sfsum = 0.0;
                            let mut locval = 0.0;
                            for df in 0..16 {
                                let locvaltemp = f32::exp(deqnt2f32(
                                    unsafe {
                                        *reg.add(
                                            (lc * 16 + df)
                                                * MAP_SIZE[index][0]
                                                * MAP_SIZE[index][1]
                                                + h * MAP_SIZE[index][1]
                                                + w,
                                        )
                                    }
                                    .into(),
                                    quant_zp_reg,
                                    quant_scale_reg,
                                ));

                                reg_dfl.push(locvaltemp);
                                sfsum += locvaltemp;
                            }
                            for df in 0..16 {
                                let locvaltemp = reg_dfl[df] / sfsum;
                                locval += locvaltemp * df as f32;
                            }
                            reg_dfl.push(locval);
                        }

                        let stride = self.strides[index] as f32;
                        let mut xmin = (self.mesh_grid[grid_index_0] - reg_dfl[0]) * stride;
                        let mut ymin = (self.mesh_grid[grid_index_1] - reg_dfl[1]) * stride;
                        let mut xmax = (self.mesh_grid[grid_index_0] + reg_dfl[2]) * stride;
                        let mut ymax = (self.mesh_grid[grid_index_1] + reg_dfl[3]) * stride;

                        xmin = xmin.max(0.0);
                        ymin = ymin.max(0.0);
                        xmax = xmax.min(self.input_width as f32);
                        ymax = ymax.min(self.input_height as f32);

                        if xmin >= 0.0
                            && ymin >= 0.0
                            && xmax <= self.input_width as f32
                            && ymax <= self.input_height as f32
                        {
                            let mut temp = DetectRect::new();
                            temp.xmin = xmin / self.input_width as f32;
                            temp.ymin = ymin / self.input_height as f32;
                            temp.xmax = xmax / self.input_width as f32;
                            temp.ymax = ymax / self.input_height as f32;
                            temp.class_id = Some(cls_index);
                            temp.score = cls_max;

                            for ms in 0..self.mask_num {
                                temp.mask[ms] = deqnt2f32(
                                    unsafe {
                                        (*msk.add(
                                            ms * MAP_SIZE[index][0] * MAP_SIZE[index][1]
                                                + h * MAP_SIZE[index][1]
                                                + w,
                                        ))
                                        .into()
                                    },
                                    quant_zp_msk,
                                    quant_scale_msk,
                                );
                            }
                            detect_rects.push(temp);
                        }
                    }
                }
            }
        }

        detect_rects.sort_by(|rect1, rect2| rect2.score.partial_cmp(&rect1.score).unwrap());

        for i in 0..detect_rects.len() {
            let xmin1 = detect_rects[i].xmin;
            let ymin1 = detect_rects[i].ymin;
            let xmax1 = detect_rects[i].xmax;
            let ymax1 = detect_rects[i].ymax;
            let class_id = detect_rects[i].class_id;
            let score = detect_rects[i].score;

            if let Some(class_id) = class_id {
                detection_rects.push(class_id as f32);
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
                    if iou > self.nms_thresh {
                        detect_rects[j].class_id = None;
                    }
                }
            }
        }

        let seg = pblob[6];
        let quant_zp_seg = qnt_zp[6];
        let quant_scale_seg = qnt_scale[6];

        for i in 0..detect_rects.len() {
            if let Some(class_id) = detect_rects[i].class_id {
                let left = (detect_rects[i].xmin * self.seg_width as f32).round() as usize;
                let top = (detect_rects[i].ymin * self.seg_height as f32).round() as usize;
                let right = (detect_rects[i].xmax * self.seg_width as f32).round() as usize;
                let bottom = (detect_rects[i].ymax * self.seg_height as f32).round() as usize;
                for h in top..bottom {
                    for w in left..right {
                        let mut seg_sum = 0.0;
                        for s in 0..self.mask_num {
                            let offset =
                                s * self.seg_width * self.seg_height + h * self.seg_width + w;
                            seg_sum += detect_rects[i].mask[s]
                                * deqnt2f32(
                                    unsafe { (*seg.add(offset)).into() },
                                    quant_zp_seg,
                                    quant_scale_seg,
                                );
                        }

                        if 1.0 / (1.0 + f32::exp(-seg_sum)) > 0.5 {
                            seg_mask[h][w][0] = self.color_lists[class_id / 10][0];
                            seg_mask[h][w][1] = self.color_lists[class_id / 10][1];
                            seg_mask[h][w][2] = self.color_lists[class_id / 10][2];
                        }
                    }
                }
            }
        }

        ret
    }
}

struct DetectRect {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    class_id: Option<usize>,
    score: f32,
    mask: Vec<f32>,
}

impl DetectRect {
    fn new() -> Self {
        DetectRect {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 0.0,
            ymax: 0.0,
            class_id: None,
            score: 0.0,
            mask: vec![0.0; 16], // Assuming 16 for the mask size, adjust as needed
        }
    }
}

fn generate_mesh_grid(map_size: [[i32; 2]; 3]) -> Vec<f32> {
    let mut mesh_grid = Vec::new();
    for index in 0..3 {
        for i in 0..map_size[index][0] {
            for j in 0..map_size[index][1] {
                mesh_grid.push(j as f32 + 0.5);
                mesh_grid.push(i as f32 + 0.5);
            }
        }
    }

    // println!("=== yolov8 mesh_grid Generate success!");

    mesh_grid
}
