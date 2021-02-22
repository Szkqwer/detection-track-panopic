#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import warnings

from yolo4_coco_bdd import YOLO_COCO_BDD

# from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import argparse
import cv2
import os

import time

import numpy as np
from PIL import Image
import torch

import torch.backends.cudnn as cudnn

from segmentation.config import config, update_config

from segmentation.model import build_segmentation_model_from_cfg

from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation

import segmentation.data.transforms.transforms as T
from segmentation.utils.save_annotation import get_panoptic_annotation

warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config1 = tf.ConfigProto()
config1.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config1.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config1))


# 定义全景分割参数
def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./configs/panoptic_deeplab_R101_os32_cityscapes.yaml',
                        # required=True,
                        type=str)
    parser.add_argument('--input-files',
                        help='input files, could be image, image list or video',
                        default='../input_img/hh.png',
                        # required=True,
                        type=str)
    parser.add_argument('--output-dir',
                        help='output directory',
                        default='../output_img/',
                        # required=True,
                        type=str)
    parser.add_argument('--extension',
                        help='file extension if input is image list',
                        default='.png',
                        type=str)
    parser.add_argument('--merge-image',
                        help='merge image with predictions',
                        action='store_true')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


# 定义全景分割颜色
class CityscapesMeta(object):
    def __init__(self):
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.ignore_label = 255

    def class_name(self):
        name_list = ['road',
                     'sidewalk',
                     'building',
                     'wall',
                     'fence',
                     'pole',
                     'traffic light',
                     'traffic sign',
                     'vegetation',
                     'terrain',
                     'sky',
                     'person',
                     'rider',
                     'car',
                     'truck',
                     'bus',
                     'train',
                     'motorcycle',
                     'bicycle']
        return name_list

    def get_bgr_color(self):
        colors = [
            (128, 64, 128),
            (232, 35, 244),
            (70, 70, 70),
            (156, 102, 102),
            (153, 153, 190),
            (153, 153, 153),
            (30, 170, 250),
            (0, 220, 220),
            (35, 142, 107),
            (152, 251, 152),
            (180, 130, 70),
            (60, 20, 220),
            (0, 0, 255),
            (142, 0, 0),
            (70, 0, 0),
            (100, 60, 0),
            (100, 80, 0),
            (230, 0, 0),
            (32, 11, 119)
        ]
        return colors

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap


# 获取全景分割的图片
def get_panopic_img(img, model, device, un_appear_class, meta_dataset):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                config.DATASET.MEAN,
                config.DATASET.STD
            )
        ]
    )

    with torch.no_grad():

        # pad image
        raw_shape = img.shape[:2]
        raw_h = raw_shape[0]
        raw_w = raw_shape[1]
        new_h = (raw_h + 31) // 32 * 32 + 1
        new_w = (raw_w + 31) // 32 * 32 + 1
        input_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        input_image[:, :] = config.DATASET.MEAN
        input_image[:raw_h, :raw_w, :] = img

        image, _ = transforms(input_image, None)

        image = image.unsqueeze(0).to(device)

        out_dict = model(image)

        semantic_pred = get_semantic_segmentation(out_dict['semantic'])

        panoptic_pred, center_pred = get_panoptic_segmentation(
            semantic_pred,
            out_dict['center'],
            out_dict['offset'],
            thing_list=meta_dataset.thing_list,
            label_divisor=meta_dataset.label_divisor,
            stuff_area=config.POST_PROCESSING.STUFF_AREA,
            void_label=(
                    meta_dataset.label_divisor *
                    meta_dataset.ignore_label),
            threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
            nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
            top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
            foreground_mask=None)

        panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

        # 找边界
        class_id = panoptic_pred // meta_dataset.label_divisor
        object_id = panoptic_pred % meta_dataset.label_divisor
        panoptic_for_edges = np.stack([class_id * 1, object_id * 10, np.zeros_like(class_id)],
                                      axis=2).astype(np.uint8)

        new_un_appear_class = []
        for i in range(0, len(un_appear_class)):
            if un_appear_class[i] not in class_id:
                new_un_appear_class.append(i)

        # 获取实例分割边界
        edges = cv2.Canny(panoptic_for_edges, 0.5, 1)
        edges_bool = (edges / 255).astype(np.bool)
        edges_bool = edges_bool & object_id.astype('bool')
        edges = (edges_bool.astype('uint8')) * 255
        # 将得到的边界膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges_bool = (edges / 255).astype(np.bool)

        # crop predictions
        panoptic_pred = panoptic_pred[:raw_h, :raw_w]
        edges_bool = edges_bool[:raw_h, :raw_w]

        img = get_panoptic_annotation(panoptic_pred,
                                      edges_bool=edges_bool,
                                      label_divisor=meta_dataset.label_divisor,
                                      colormap=meta_dataset.create_label_colormap(),
                                      image=img)

        # cv2.imwrite('a.png', img)
        # print('aaa')
        return img.astype('uint8'), new_un_appear_class


if __name__ == '__main__':
    args = parse_args()
    # 初始化模型
    yolo_coco_bdd = YOLO_COCO_BDD()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    device = torch.device('cuda:{}'.format(gpus[0]))
    # 读取全景分割model
    panipic_model = build_segmentation_model_from_cfg(config)
    model_state_file = config.TEST.MODEL_FILE
    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
        panipic_model.load_state_dict(model_weights, strict=True)
        print('panipic model loaded')

    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')
    panipic_model = panipic_model.to(device)
    startTime = time.time()

    # 定义输出图片路径
    # #     picture_folder='s3://dsjgb/video/887SCHcailiao/imageSeg'
    # picture_folder = './szktest/video/887SCHcailiao/imageSegs'

    # 定义计数文本路径
    #     count_file_root='s3://dsjgb/video/887SCHcailiao/totalCount.txt'
    count_file_root = './tmp/totalCount.txt'

    # 定义输入视频
    input_video_name = 'final_demo.mp4'
    input_video_root = './input_videos/' + input_video_name

    # 定义输出视频
    #     output_video_root='s3://dsjgb/video/887SCHcailiao/outputVideo.avi'
    # output_video_root = './szktest/video/887SCHcailiao/outputVideo.avi'

    # 定义数据
    cos_dis_rate = 0.95
    nn_budget = 50
    max_cosine_distance = 0.8
    nms_max_overlap = 1.0
    max_iou_distance = 0.8
    max_age = 45  # 消失后保留帧数
    min_confirm = 4  # 最少确认帧数

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_coco_bdd = Tracker(metric=metric,
                               max_age=max_age,
                               max_iou_distance=max_iou_distance,
                               min_confirm=min_confirm,
                               classNum=len(yolo_coco_bdd.class_names),
                               cos_dis_rate=cos_dis_rate)

    video_capture_coco_bdd = cv2.VideoCapture(input_video_root)

    w = int(video_capture_coco_bdd.get(3))
    h = int(video_capture_coco_bdd.get(4))
    input_fps = int(video_capture_coco_bdd.get(5))
    print('input video shpae:', w, h, 'FPS:', input_fps)

    # 定义全景分割数据
    meta_dataset = CityscapesMeta()
    panopic_class_name = meta_dataset.class_name()
    panopic_class_color = meta_dataset.get_bgr_color()
    need_show_class = [0, 1, 2, 3, 4, 5, 8, 9, 10]
    un_appear_class = [n for n in range(0, len(panopic_class_name))]

    # 设置输出的视频数据
    new_w = 1024
    new_h = 576
    w_blank = 200
    h_blank = 0
    new_fps = input_fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_coco_bdd = cv2.VideoWriter('./tmp/result_coco_bdd.avi', fourcc, new_fps,
                                   # (new_h + 2 * h_blank,new_w + 2 * w_blank))
                                   (new_w + 2 * w_blank, new_h + 2 * h_blank))

    # 绘制背景图
    background = np.ones((new_h + 2 * h_blank, new_w + 2 * w_blank, 3)) * 255
    cv2.putText(background, 'panipic color:', (0, 20), 0, 5e-3 * 120, (0, 0, 0), 2)
    for i in range(0, len(panopic_class_name)):
        text_x = 0
        text_y = 40 + i * 25
        text = panopic_class_name[i]
        cv2.putText(background, text, (text_x, text_y), 0, 5e-3 * 100, panopic_class_color[i], 2)
    text_x = w_blank + new_w + 10
    text_y = 20
    text = 'real time counting:'
    cv2.putText(background, text, (text_x, text_y), 0, 5e-3 * 120, (0, 0, 0), 2)

    # 设置各种颜色
    track_color = (235, 205, 205)
    track_text_color = (0, 255, 255)
    detection_color = (255, 0, 0)
    detection_text_color = (185, 185, 0)

    # 保存每一帧的数据
    # track_place_data = {}
    # detection_place_data = {}
    # track_text_data = {}
    # detection_text_data = {}

    # 运行一定帧数计算fps
    t1 = time.time()
    frameIndex = 0
    frameBlank = 120

    while True:
        ret, frame = video_capture_coco_bdd.read()
        if ret != True:
            break
        # 跳帧
        if frameIndex % 1 == 0:
            # 检测
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            image = image.resize((new_w, new_h), Image.BICUBIC)
            boxs, confidence, classIndexList = yolo_coco_bdd.detect_image(image, new_w, new_h)

            # 全景分割
            # panipic_img,un_appear_class=get_panopic_img(np.array(image), panipic_model, device,un_appear_class,meta_dataset)
            panipic_img = np.array(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))

            now_frame = np.array(background)

            # 特征获取
            features_coco_bdd = encoder(frame, boxs)

            detections = []
            for bbox, confidence, feature, classIndex in zip(boxs, confidence, features_coco_bdd, classIndexList):
                detections.append(Detection(bbox, confidence, feature, classIndex))

            tracker_coco_bdd.predict()
            tracker_coco_bdd.update(detections)

            # frame_track_place_data = []
            # frame_detection_place_data = []
            # frame_track_text_data = []
            # frame_detection_text_data = []
            # 当前被确认物体
            now_confirmed = np.zeros((len(yolo_coco_bdd.class_names))).astype('uint8')

            for track in tracker_coco_bdd.tracks:
                # 已经确认帧数
                confirm_num = tracker_coco_bdd.confirmed_time_list[track.classIndex][track.objectIndex]

                if not track.is_confirmed() or track.time_since_update > 0 or confirm_num < min_confirm:
                    continue
                bbox = track.to_tlbr()

                # 显示索引从1开始
                index = tracker_coco_bdd.confirmed_id_list[track.classIndex].index(track.track_id) + 1
                class_name = str(yolo_coco_bdd.class_names[track.classIndex])
                now_confirmed[[track.classIndex]] += 1
                text = class_name + '_' + str(index)

                # 加入数据
                # frame_track_place_data.append(bbox)
                # frame_track_text_data.append(text)

                # 分割图片
                # if confirm_num == min_confirm:
                #     tempFrame = np.array(frame)
                #
                #     cv2.rectangle(tempFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track_color, 2)
                #     cv2.putText(tempFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)
                #
                #     cv2.imwrite('/tmp/temp.png', tempFrame)
                #     picture_root = picture_folder + '/' + class_name + '/' + text + '.png'
                #     mox.file.copy('/tmp/temp.png', picture_root)

                # 在视频里绘制track框与对应编号
                # cv2.rectangle(panipic_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),track_color, 2)
                # cv2.putText(panipic_img, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)

                # 直接绘制
                cv2.rectangle(panipic_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track_color, 2)
                cv2.putText(panipic_img, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)

            # 绘制当前被确认物体到图像
            for i in range(0, len(yolo_coco_bdd.class_names)):
                text_x = w_blank + new_w + 30
                text_y = 40 + i * 25
                text = yolo_coco_bdd.class_names[i] + ': ' + str(now_confirmed[i])
                cv2.putText(now_frame, text, (text_x, text_y), 0, 5e-3 * 100, (0, 0, 0), 2)

            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2)

                # 加入数据
                # frame_detection_place_data.append(bbox)
                # frame_detection_text_data.append(score + '%')

                # 在视频里绘制detection框与对应得分
                # cv2.rectangle(panipic_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), detection_color, 2)
                # cv2.putText(panipic_img, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, detection_text_color, 2)

                # 直接绘制
                cv2.rectangle(panipic_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), detection_color,
                              2)
                cv2.putText(panipic_img, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, detection_text_color,
                            2)

            # track_place_data[frameIndex] = frame_track_place_data
            # detection_place_data[frameIndex] = frame_detection_place_data
            # track_text_data[frameIndex] = frame_track_text_data
            # detection_text_data[frameIndex] = frame_detection_text_data

            # out_coco_bdd.write(panipic_img)

            now_frame[:, w_blank:w_blank + new_w, :] = panipic_img
            out_coco_bdd.write(now_frame.astype('uint8'))

        if frameIndex % frameBlank == frameBlank - 1:
            t2 = time.time()
            print("frame index = %d" % (frameIndex), ',', "FPS = %f" % (frameBlank / (t2 - t1)))
            t1 = time.time()
        frameIndex += 1

    video_capture_coco_bdd.release()
    out_coco_bdd.release()

    out_capture_coco_bdd = cv2.VideoCapture('./tmp/result_coco_bdd.avi')
    # out_capture_bdd = cv2.VideoCapture('./tmp/result_bdd.avi')
    # out = cv2.VideoWriter('./tmp/result.avi', fourcc, 30, (new_w, new_h))
    # frameIndex = 0
    # while True:
    #     ret_coco_bdd, frame_coco_bdd = out_capture_coco_bdd.read()
    #     # ret_bdd, frame_bdd = out_capture_bdd.read()
    #
    #     if ret_coco_bdd != True :
    #         break
    #
    #         # 合并
    #     # coco帧加入track
    #     if len(track_place_data[frameIndex]) > 0:
    #
    #         for i in range(0, len(track_place_data[frameIndex])):
    #             cv2.rectangle(frame_coco_bdd,
    #                           (int(track_place_data[frameIndex][i][0]), int(track_place_data[frameIndex][i][1])),
    #                           (int(track_place_data[frameIndex][i][2]), int(track_place_data[frameIndex][i][3])),
    #                           track_color, 2)
    #
    #             cv2.putText(frame_coco_bdd, track_text_data[frameIndex][i], (int(track_place_data[frameIndex][i][0]),
    #                                                                      int(track_place_data[frameIndex][i][1])), 0,
    #                         5e-3 * 130, track_text_color, 2)
    #
    #     # if len(track_place_data[frameIndex + 1]) > 0:
    #     #     for i in range(0, len(track_place_data[frameIndex + 1])):
    #     #         cv2.rectangle(frame_coco, (
    #     #         int(track_place_data[frameIndex + 1][i][0]), int(track_place_data[frameIndex + 1][i][1])),
    #     #                       (
    #     #                       int(track_place_data[frameIndex + 1][i][2]), int(track_place_data[frameIndex + 1][i][3])),
    #     #                       track_color, 2)
    #     #
    #     #         cv2.putText(frame_coco, track_text_data[frameIndex + 1][i],
    #     #                     (int(track_place_data[frameIndex + 1][i][0]),
    #     #                      int(track_place_data[frameIndex + 1][i][1])), 0, 5e-3 * 130, track_text_color, 2)
    #
    #     # coco帧加入detection
    #     if len(detection_place_data[frameIndex]) > 0:
    #         for i in range(0, len(detection_place_data[frameIndex])):
    #             cv2.rectangle(frame_coco_bdd, (
    #             int(detection_place_data[frameIndex][i][0]), int(detection_place_data[frameIndex][i][1])),
    #                           (
    #                           int(detection_place_data[frameIndex][i][2]), int(detection_place_data[frameIndex][i][3])),
    #                           detection_color, 2)
    #
    #             cv2.putText(frame_coco_bdd, detection_text_data[frameIndex][i],
    #                         (int(detection_place_data[frameIndex][i][0]),
    #                          int(detection_place_data[frameIndex][i][3])), 0, 5e-3 * 130, detection_text_color, 2)
    #
    #     # if len(detection_place_data[frameIndex + 1]) > 0:
    #     #     for i in range(0, len(detection_place_data[frameIndex + 1])):
    #     #         cv2.rectangle(frame_coco, (
    #     #         int(detection_place_data[frameIndex + 1][i][0]), int(detection_place_data[frameIndex + 1][i][1])),
    #     #                       (int(detection_place_data[frameIndex + 1][i][2]),
    #     #                        int(detection_place_data[frameIndex + 1][i][3])), detection_color, 2)
    #     #
    #     #         cv2.putText(frame_coco, detection_text_data[frameIndex + 1][i],
    #     #                     (int(detection_place_data[frameIndex + 1][i][0]),
    #     #                      int(detection_place_data[frameIndex + 1][i][3])), 0, 5e-3 * 130, detection_text_color, 2)
    #
    #     # bdd帧加入track
    #     # if len(track_place_data[frameIndex]) > 0:
    #     #
    #     #     for i in range(0, len(track_place_data[frameIndex])):
    #     #         cv2.rectangle(frame_bdd,
    #     #                       (int(track_place_data[frameIndex][i][0]), int(track_place_data[frameIndex][i][1])),
    #     #                       (int(track_place_data[frameIndex][i][2]), int(track_place_data[frameIndex][i][3])),
    #     #                       track_color, 2)
    #     #
    #     #         cv2.putText(frame_bdd, track_text_data[frameIndex][i], (int(track_place_data[frameIndex][i][0]),
    #     #                                                                 int(track_place_data[frameIndex][i][1])), 0,
    #     #                     5e-3 * 130, track_text_color, 2)
    #     #
    #     # if len(track_place_data[frameIndex + 1]) > 0:
    #     #     for i in range(0, len(track_place_data[frameIndex + 1])):
    #     #         cv2.rectangle(frame_bdd, (
    #     #         int(track_place_data[frameIndex + 1][i][0]), int(track_place_data[frameIndex + 1][i][1])),
    #     #                       (
    #     #                       int(track_place_data[frameIndex + 1][i][2]), int(track_place_data[frameIndex + 1][i][3])),
    #     #                       track_color, 2)
    #     #
    #     #         cv2.putText(frame_bdd, track_text_data[frameIndex + 1][i], (int(track_place_data[frameIndex + 1][i][0]),
    #     #                                                                     int(track_place_data[frameIndex + 1][i][
    #     #                                                                             1])), 0, 5e-3 * 130,
    #     #                     track_text_color, 2)
    #     #
    #     # # bdd帧加入detection
    #     # if len(detection_place_data[frameIndex]) > 0:
    #     #     for i in range(0, len(detection_place_data[frameIndex])):
    #     #         cv2.rectangle(frame_bdd, (
    #     #         int(detection_place_data[frameIndex][i][0]), int(detection_place_data[frameIndex][i][1])),
    #     #                       (
    #     #                       int(detection_place_data[frameIndex][i][2]), int(detection_place_data[frameIndex][i][3])),
    #     #                       detection_color, 2)
    #     #
    #     #         cv2.putText(frame_bdd, detection_text_data[frameIndex][i], (int(detection_place_data[frameIndex][i][0]),
    #     #                                                                     int(detection_place_data[frameIndex][i][
    #     #                                                                             3])), 0, 5e-3 * 130,
    #     #                     detection_text_color, 2)
    #     #
    #     # if len(detection_place_data[frameIndex + 1]) > 0:
    #     #     for i in range(0, len(detection_place_data[frameIndex + 1])):
    #     #         cv2.rectangle(frame_bdd, (
    #     #         int(detection_place_data[frameIndex + 1][i][0]), int(detection_place_data[frameIndex + 1][i][1])),
    #     #                       (int(detection_place_data[frameIndex + 1][i][2]),
    #     #                        int(detection_place_data[frameIndex + 1][i][3])), detection_color, 2)
    #     #
    #     #         cv2.putText(frame_bdd, detection_text_data[frameIndex + 1][i],
    #     #                     (int(detection_place_data[frameIndex + 1][i][0]),
    #     #                      int(detection_place_data[frameIndex + 1][i][3])), 0, 5e-3 * 130, detection_text_color, 2)
    #
    #     out.write(frame_coco_bdd)
    #     # out.write(frame_bdd)
    #
    #     frameIndex += 1

    # out.release()
    # mox.file.copy('/tmp/result.avi', output_video_root)

    # 写入每种数量与具体情况

    print('writing result')
    # 统计跟踪去重数量
    result_coco_bdd = tracker_coco_bdd.confirmed_time_list
    result_dict = {}
    # 加入coco_bdd类
    for i in range(0, len(result_coco_bdd)):
        sum = 0
        for j in range(0, len(result_coco_bdd[i])):
            if result_coco_bdd[i][j] >= min_confirm:
                sum += 1
        # if yolo_coco_bdd.class_names[i] in yolo_coco_bdd.chosen_classes:
        result_dict[yolo_coco_bdd.class_names[i]] = sum

    count_file = open(count_file_root, 'w')
    count_file.write('track information: ' + '\n')
    sorted_result = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    for key, value in sorted_result:
        count_file.write(key + ':' + str(value) + '\n')

    count_file.write('\n')
    count_file.write('panopic information:' + '\n')
    for i in range(0, len(need_show_class)):
        if i in un_appear_class:
            count_file.write(panopic_class_name[i] + ': ' + 'not appeared' + '\n')
        else:
            count_file.write(panopic_class_name[i] + ': ' + 'appeared' + '\n')

    count_file.close()
    endTime = time.time()
    print('using time:', endTime - startTime)
    print('over')
