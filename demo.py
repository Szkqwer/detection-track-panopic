#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import warnings
from yolo4_coco import YOLO_COCO
from yolo4_bdd import YOLO_BDD

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
config1.gpu_options.per_process_gpu_memory_fraction = 0.4
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
    yolo_coco = YOLO_COCO()
    yolo_bdd = YOLO_BDD()

    panipic_model = build_segmentation_model_from_cfg(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    device = torch.device('cuda:{}'.format(gpus[0]))
    panipic_model = panipic_model.to(device)
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):

        model_weights = torch.load(model_state_file)

        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            # logger.info('Evaluating a intermediate checkpoint.')
        panipic_model.load_state_dict(model_weights, strict=True)
        # model.load_state_dict(torch.load(model_state_file), strict=True)
        print('panipic model loaded')
        # logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')
    startTime = time.time()

    # 定义输出图片路径
    #     picture_folder='s3://dsjgb/video/887SCHcailiao/imageSeg'
    picture_folder = './szktest/video/887SCHcailiao/imageSegs'

    # 定义计数文本路径
    #     count_file_root='s3://dsjgb/video/887SCHcailiao/totalCount.txt'
    count_file_root = './tmp/totalCount.txt'

    # 定义输入视频
    input_video_name = '1.mp4'
    input_video_root = './input_videos/' + input_video_name

    # 定义输出视频
    #     output_video_root='s3://dsjgb/video/887SCHcailiao/outputVideo.avi'
    # output_video_root = './szktest/video/887SCHcailiao/outputVideo.avi'

    # 定义数据
    cos_dis_rate = 0.95
    nn_budget = 50
    max_cosine_distance = 0.8
    nms_max_overlap = 1.0
    max_iou_distance = 0.75
    max_age = 20  # 消失后保留帧数
    min_confirm = 2  # 最少确认帧数

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker_coco = Tracker(metric=metric,
                           max_age=max_age,
                           max_iou_distance=max_iou_distance,
                           min_confirm=min_confirm,
                           classNum=len(yolo_coco.class_names),
                           cos_dis_rate=cos_dis_rate)
    tracker_bdd = Tracker(metric=metric,
                          max_age=max_age,
                          max_iou_distance=max_iou_distance,
                          min_confirm=min_confirm,
                          classNum=len(yolo_bdd.class_names),
                          cos_dis_rate=cos_dis_rate)

    # 删除之前识别的图片
    # mox.file.remove(picture_folder, recursive=True)

    # 处理测试的视频
    # mox.file.copy(input_video_root, '/tmp/' + input_video_name)
    # video_capture_coco = cv2.VideoCapture('/tmp/' + input_video_name)

    # 定义全景分割数据
    meta_dataset = CityscapesMeta()
    panopic_class_name = meta_dataset.class_name()
    need_show_class = [0, 1, 2, 3, 4, 5, 8, 9, 10]
    panopic_class_color = meta_dataset.get_bgr_color()
    un_appear_class = [n for n in range(0, len(panopic_class_name))]

    video_capture_coco = cv2.VideoCapture(input_video_root)
    w = int(video_capture_coco.get(3))
    h = int(video_capture_coco.get(4))
    input_fps = int(video_capture_coco.get(5))
    print('input video shpae:', w, h, 'FPS:', input_fps)

    # 设置输出的视频数据
    new_w = 1024
    new_h = 576
    w_blank = 200
    h_blank = 0
    new_fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_coco = cv2.VideoWriter('./tmp/result_coco.avi', fourcc, new_fps,
                               (new_w + 2 * w_blank, new_h + 2 * h_blank))

    # 绘制背景图
    background = np.ones((new_h + 2 * h_blank, new_w + 2 * w_blank, 3)) * 255
    cv2.putText(background, 'panipic color:', (0, 20), 0, 5e-3 * 120, (0, 0, 0), 2)
    for i in range(0, len(panopic_class_name)):
        text_x = 0
        text_y = 40 + i * 25
        text = panopic_class_name[i]
        cv2.putText(background, text, (text_x, text_y), 0, 5e-3 * 100, panopic_class_color[i], 2)
        # cv2.rectangle(background, (text_x, text_y), (int(bbox[2]), int(bbox[3])), track_color, 2)

    # 设置各种颜色
    track_color = (255, 205, 255)
    track_text_color = (0, 255, 255)
    detection_color = (255, 0, 0)
    detection_text_color = (185, 185, 0)

    # 保存每一帧的数据
    track_place_data = {}
    detection_place_data = {}
    track_text_data = {}
    detection_text_data = {}
    track_num_text_data = {}

    # 运行一定帧数计算fps
    t1 = time.time()
    frameIndex = 0
    frameBlank = 120

    while True:
        ret, frame = video_capture_coco.read()
        if ret != True:
            break
        # 跳帧
        if frameIndex % 4 == 0:
            frame = cv2.resize(frame, (new_w, new_h))
            # coco预测
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs, confidence, classIndexList = yolo_coco.detect_image(image)

            panipic_img, un_appear_class = get_panopic_img(np.array(image), panipic_model, device, un_appear_class,
                                                           meta_dataset)
            now_frame = np.array(background)

            # 特征获取
            features_coco = encoder(frame, boxs)

            detections = []
            for bbox, confidence, feature, classIndex in zip(boxs, confidence, features_coco, classIndexList):
                detections.append(Detection(bbox, confidence, feature, classIndex))

            tracker_coco.predict()
            tracker_coco.update(detections)

            # 视频输出帧
            # outFrame = np.array(frame)

            frame_track_place_data = []
            frame_detection_place_data = []
            frame_track_text_data = []
            frame_detection_text_data = []
            frame_track_num_text_data = []

            # 当前被确认物体
            now_confirmed = np.zeros((len(yolo_coco.class_names))).astype('uint8')
            for track in tracker_coco.tracks:
                # 已经确认帧数
                confirm_num = tracker_coco.confirmed_time_list[track.classIndex][track.objectIndex]

                if not track.is_confirmed() or track.time_since_update > 0 or confirm_num < min_confirm:
                    continue
                bbox = track.to_tlbr()
                frame_track_place_data.append(bbox)

                # 显示索引从1开始
                index = tracker_coco.confirmed_id_list[track.classIndex].index(track.track_id) + 1
                class_name = str(yolo_coco.class_names[track.classIndex])
                text = class_name + '_' + str(index)
                now_confirmed[track.classIndex] += 1
                frame_track_text_data.append(text)
                # 分割图片
                # if confirm_num == min_confirm:
                #     tempFrame = np.array(frame)
                #
                #     cv2.rectangle(tempFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track_color, 2)
                #     cv2.putText(tempFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)

                # cv2.imwrite('/tmp/temp.png', tempFrame)
                # picture_root = picture_folder + '/' + class_name + '/' + text + '.png'
                # mox.file.copy('/tmp/temp.png', picture_root)

                # 在视频里绘制track框与对应编号
                #                 cv2.rectangle(outFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),track_color, 2)
                #                 cv2.putText(outFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)

            # 当前被确认物体数据
            for i in range(0, len(yolo_coco.class_names)):
                if yolo_coco.class_names[i] in yolo_coco.chosen_classes:
                    text = yolo_coco.class_names[i] + ': ' + str(now_confirmed[i])
                    frame_track_num_text_data.append(text)

            for det in detections:
                bbox = det.to_tlbr()
                frame_detection_place_data.append(bbox)

                score = "%.2f" % round(det.confidence * 100, 2)
                frame_detection_text_data.append(score + '%')
                # 在视频里绘制detection框与对应得分

                #                 cv2.rectangle(outFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), detection_color, 2)
                #                 cv2.putText(outFrame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, detection_text_color, 2)

            track_place_data[frameIndex] = frame_track_place_data
            detection_place_data[frameIndex] = frame_detection_place_data
            track_text_data[frameIndex] = frame_track_text_data
            detection_text_data[frameIndex] = frame_detection_text_data
            track_num_text_data[frameIndex] = frame_track_num_text_data

            now_frame[:, w_blank:w_blank + new_w, :] = panipic_img
            out_coco.write(now_frame.astype('uint8'))

        if frameIndex % frameBlank == frameBlank - 1:
            t2 = time.time()
            print("frame index = %d" % (frameIndex), ',', "FPS = %f" % (frameBlank / (t2 - t1)))
            t1 = time.time()
        frameIndex += 1

    video_capture_coco.release()
    out_coco.release()

    # 再次遍历
    out_bdd = cv2.VideoWriter('./tmp/result_bdd.avi', fourcc, new_fps,
                              (new_w + 2 * w_blank, new_h + 2 * h_blank))
    frameIndex = 0
    t1 = time.time()
    # video_capture_bdd = cv2.VideoCapture('./tmp/' + input_video_name)
    video_capture_bdd = cv2.VideoCapture(input_video_root)
    while True:
        ret, frame = video_capture_bdd.read()
        if ret != True:
            break
        if frameIndex % 4 == 2:
            frame = cv2.resize(frame, (new_w, new_h))
            boxs = []
            confidence = []
            classIndexList = []
            # bdd预测
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs, confidence, classIndexList = yolo_bdd.detect_image(image)
            features_bdd = encoder(frame, boxs)
            panipic_img, un_appear_class = get_panopic_img(np.array(image), panipic_model, device, un_appear_class,
                                                           meta_dataset)

            now_frame = np.array(background)
            detections = []
            for bbox, confidence, feature, classIndex in zip(boxs, confidence, features_bdd, classIndexList):
                detections.append(Detection(bbox, confidence, feature, classIndex))

            tracker_bdd.predict()
            tracker_bdd.update(detections)

            # 视频输出帧
            # outFrame = np.array(frame)

            frame_track_place_data = []
            frame_detection_place_data = []
            frame_track_text_data = []
            frame_detection_text_data = []
            frame_track_num_text_data = []
            # 当前被确认物体
            now_confirmed = np.zeros((len(yolo_bdd.class_names))).astype('uint8')

            for track in tracker_bdd.tracks:
                # 已经确认帧数
                confirm_num = tracker_bdd.confirmed_time_list[track.classIndex][track.objectIndex]

                if not track.is_confirmed() or track.time_since_update > 0 or confirm_num < min_confirm:
                    continue
                bbox = track.to_tlbr()
                frame_track_place_data.append(bbox)

                # 显示索引从1开始
                index = tracker_bdd.confirmed_id_list[track.classIndex].index(track.track_id) + 1
                class_name = str(yolo_bdd.class_names[track.classIndex])
                text = class_name + '_' + str(index)
                frame_track_text_data.append(text)
                now_confirmed[track.classIndex] += 1

                # 分割图片
                # if confirm_num == min_confirm:
                #     tempFrame = np.array(frame)
                #
                #     cv2.rectangle(tempFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track_color, 2)
                #     cv2.putText(tempFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)

                # cv2.imwrite('/tmp/temp.png', tempFrame)
                # picture_root = picture_folder + '/' + class_name + '/' + text + '.png'
                # mox.file.copy('/tmp/temp.png', picture_root)

                # 在视频里绘制track框与对应编号
                #                 cv2.rectangle(outFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), track_color, 2)
                #                 cv2.putText(outFrame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 120, track_text_color, 2)

            # 当前被确认物体数据
            for i in range(0, len(yolo_bdd.class_names)):
                if yolo_bdd.class_names[i] in yolo_bdd.chosen_classes:
                    text = yolo_bdd.class_names[i] + ': ' + str(now_confirmed[i])
                    frame_track_num_text_data.append(text)

            for det in detections:
                bbox = det.to_tlbr()
                frame_detection_place_data.append(bbox)

                score = "%.2f" % round(det.confidence * 100, 2)
                frame_detection_text_data.append(score + '%')

                # 在视频里绘制detection框与对应得分
                #                 cv2.rectangle(outFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), detection_color, 2)
                #                 cv2.putText(outFrame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, detection_text_color, 2)

            track_place_data[frameIndex] = frame_track_place_data
            detection_place_data[frameIndex] = frame_detection_place_data
            track_text_data[frameIndex] = frame_track_text_data
            detection_text_data[frameIndex] = frame_detection_text_data
            track_num_text_data[frameIndex] = frame_track_num_text_data

            now_frame[:, w_blank:w_blank + new_w, :] = panipic_img
            out_bdd.write(now_frame.astype('uint8'))

        if frameIndex % frameBlank == frameBlank - 1:
            t2 = time.time()
            print("frame index = %d" % (frameIndex), ',', "FPS = %f" % (frameBlank / (t2 - t1)))
            t1 = time.time()
        frameIndex += 1
    out_bdd.release()

    # 合并视频
    print('merging videos')
    out_capture_coco = cv2.VideoCapture('./tmp/result_coco.avi')
    out_capture_bdd = cv2.VideoCapture('./tmp/result_bdd.avi')
    out = cv2.VideoWriter('./tmp/result.avi', fourcc, new_fps, (new_w + 2 * w_blank, new_h + 2 * h_blank))
    # video_capture = cv2.VideoCapture('/tmp/' + input_video_name)
    # video_capture = cv2.VideoCapture(output_video_root)
    frameIndex = 0
    while True:
        ret_coco, frame_coco = out_capture_coco.read()
        ret_bdd, frame_bdd = out_capture_bdd.read()

        if ret_coco != True or ret_bdd != True:
            break

        # 合并
        # coco帧加入track
        if len(track_place_data[frameIndex]) > 0:
            for i in range(0, len(track_place_data[frameIndex])):
                cv2.rectangle(frame_coco,
                              (int(track_place_data[frameIndex][i][0] + w_blank),
                               int(track_place_data[frameIndex][i][1])),
                              (int(track_place_data[frameIndex][i][2] + w_blank),
                               int(track_place_data[frameIndex][i][3])),
                              track_color, 2)

                cv2.putText(frame_coco, track_text_data[frameIndex][i],
                            (int(track_place_data[frameIndex][i][0] + w_blank),
                             int(track_place_data[frameIndex][i][1])), 0,
                            5e-3 * 130, track_text_color, 2)

        if len(track_place_data[frameIndex + 2]) > 0:
            for i in range(0, len(track_place_data[frameIndex +2])):
                cv2.rectangle(frame_coco, (
                    int(track_place_data[frameIndex +2][i][0] + w_blank), int(track_place_data[frameIndex +2][i][1])),
                              (
                                  int(track_place_data[frameIndex +2][i][2] + w_blank),
                                  int(track_place_data[frameIndex +2][i][3])),
                              track_color, 2)

                cv2.putText(frame_coco, track_text_data[frameIndex +2][i],
                            (int(track_place_data[frameIndex +2][i][0] + w_blank),
                             int(track_place_data[frameIndex +2][i][1])), 0, 5e-3 * 130, track_text_color, 2)

        # coco帧加入detection
        if len(detection_place_data[frameIndex]) > 0:
            for i in range(0, len(detection_place_data[frameIndex])):
                cv2.rectangle(frame_coco, (
                    int(detection_place_data[frameIndex][i][0] + w_blank), int(detection_place_data[frameIndex][i][1])),
                              (
                                  int(detection_place_data[frameIndex][i][2] + w_blank),
                                  int(detection_place_data[frameIndex][i][3])),
                              detection_color, 2)

                cv2.putText(frame_coco, detection_text_data[frameIndex][i],
                            (int(detection_place_data[frameIndex][i][0] + w_blank),
                             int(detection_place_data[frameIndex][i][3])), 0, 5e-3 * 130, detection_text_color, 2)

        if len(detection_place_data[frameIndex + 2]) > 0:
            for i in range(0, len(detection_place_data[frameIndex +2])):
                cv2.rectangle(frame_coco, (
                    int(detection_place_data[frameIndex +2][i][0] + w_blank),
                    int(detection_place_data[frameIndex +2][i][1])),
                              (int(detection_place_data[frameIndex +2][i][2] + w_blank),
                               int(detection_place_data[frameIndex +2][i][3])), detection_color, 2)

                cv2.putText(frame_coco, detection_text_data[frameIndex +2][i],
                            (int(detection_place_data[frameIndex +2][i][0] + w_blank),
                             int(detection_place_data[frameIndex +2][i][3])), 0, 5e-3 * 130, detection_text_color, 2)

        # coco帧加入实时数量
        if len(track_num_text_data[frameIndex]) > 0:
            text_x = w_blank + new_w + 10
            text_y = 20
            text = 'real time counting:'
            cv2.putText(frame_coco, text, (text_x, text_y), 0, 5e-3 * 120, (0, 0, 0), 2)
            for i in range(0, len(yolo_coco.chosen_classes)):
                # text_x = w_blank + new_w + 10
                text_y = 40 + i * 25
                text = track_num_text_data[frameIndex][i]
                cv2.putText(frame_coco, text, (text_x, text_y), 0, 5e-3 * 100, (0, 0, 0), 2)

        if len(track_num_text_data[frameIndex + 2]) > 0:
            text_x = w_blank + new_w + 10
            text_y = 20
            text = 'real time counting:'
            cv2.putText(frame_coco, text, (text_x, text_y), 0, 5e-3 * 120, (0, 0, 0), 2)
            for i in range(0, len(yolo_bdd.chosen_classes)):
                # text_x = w_blank + new_w + 10
                text_y = 40 + (i + len(yolo_coco.chosen_classes)) * 25
                text = track_num_text_data[frameIndex +2][i]
                cv2.putText(frame_coco, text, (text_x, text_y), 0, 5e-3 * 100, (0, 0, 0), 2)

        # bdd帧加入track
        if len(track_place_data[frameIndex]) > 0:

            for i in range(0, len(track_place_data[frameIndex])):
                cv2.rectangle(frame_bdd,
                              (int(track_place_data[frameIndex][i][0] + w_blank),
                               int(track_place_data[frameIndex][i][1])),
                              (int(track_place_data[frameIndex][i][2] + w_blank),
                               int(track_place_data[frameIndex][i][3])),
                              track_color, 2)

                cv2.putText(frame_bdd, track_text_data[frameIndex][i],
                            (int(track_place_data[frameIndex][i][0] + w_blank),
                             int(track_place_data[frameIndex][i][1])), 0,
                            5e-3 * 130, track_text_color, 2)

        if len(track_place_data[frameIndex + 2]) > 0:
            for i in range(0, len(track_place_data[frameIndex +2])):
                cv2.rectangle(frame_bdd, (
                    int(track_place_data[frameIndex +2][i][0] + w_blank), int(track_place_data[frameIndex +2][i][1])),
                              (
                                  int(track_place_data[frameIndex +2][i][2] + w_blank),
                                  int(track_place_data[frameIndex +2][i][3])),
                              track_color, 2)

                cv2.putText(frame_bdd, track_text_data[frameIndex +2][i],
                            (int(track_place_data[frameIndex +2][i][0] + w_blank),
                             int(track_place_data[frameIndex +2][i][1])), 0, 5e-3 * 130, track_text_color, 2)

        # bdd帧加入detection
        if len(detection_place_data[frameIndex]) > 0:
            for i in range(0, len(detection_place_data[frameIndex])):
                cv2.rectangle(frame_bdd, (
                    int(detection_place_data[frameIndex][i][0] + w_blank), int(detection_place_data[frameIndex][i][1])),
                              (
                                  int(detection_place_data[frameIndex][i][2] + w_blank),
                                  int(detection_place_data[frameIndex][i][3])),
                              detection_color, 2)

                cv2.putText(frame_bdd, detection_text_data[frameIndex][i],
                            (int(detection_place_data[frameIndex][i][0] + w_blank),
                             int(detection_place_data[frameIndex][i][
                                     3])), 0, 5e-3 * 130,
                            detection_text_color, 2)

        if len(detection_place_data[frameIndex +2]) > 0:
            for i in range(0, len(detection_place_data[frameIndex +2])):
                cv2.rectangle(frame_bdd, (
                    int(detection_place_data[frameIndex +2][i][0] + w_blank),
                    int(detection_place_data[frameIndex +2][i][1])),
                              (int(detection_place_data[frameIndex +2][i][2] + w_blank),
                               int(detection_place_data[frameIndex +2][i][3])), detection_color, 2)

                cv2.putText(frame_bdd, detection_text_data[frameIndex +2][i],
                            (int(detection_place_data[frameIndex +2][i][0] + w_blank),
                             int(detection_place_data[frameIndex +2][i][3])), 0, 5e-3 * 130, detection_text_color, 2)

        # bdd帧加入实时数量
        if len(track_num_text_data[frameIndex]) > 0:
            text_x = w_blank + new_w + 10
            text_y = 20
            text = 'real time counting:'
            cv2.putText(frame_bdd, text, (text_x, text_y), 0, 5e-3 * 120, (0, 0, 0), 2)
            for i in range(0, len(yolo_coco.chosen_classes)):
                # text_x = w_blank + new_w + 10
                text_y = 40 + i * 25
                text = track_num_text_data[frameIndex][i]
                cv2.putText(frame_bdd, text, (text_x, text_y), 0, 5e-3 * 100, (0, 0, 0), 2)

        if len(track_num_text_data[frameIndex +2]) > 0:
            text_x = w_blank + new_w + 10
            text_y = 20
            text = 'real time counting:'
            cv2.putText(frame_bdd, text, (text_x, text_y), 0, 5e-3 * 120, (0, 0, 0), 2)
            for i in range(0, len(yolo_bdd.chosen_classes)):
                # text_x = w_blank + new_w + 10
                text_y = 40 + (i + len(yolo_coco.chosen_classes)) * 25
                text = track_num_text_data[frameIndex +2][i]
                cv2.putText(frame_bdd, text, (text_x, text_y), 0, 5e-3 * 100, (0, 0, 0), 2)

        out.write(frame_coco)
        out.write(frame_bdd)

        frameIndex += 4

    out.release()
    # mox.file.copy('/tmp/result.avi', output_video_root)

    # 写入每种数量与具体情况
    print('writing result')
    result_coco = tracker_coco.confirmed_time_list
    result_bdd = tracker_bdd.confirmed_time_list

    result_dict = {}
    # 加入coco类
    for i in range(0, len(result_coco)):
        sum = 0
        for j in range(0, len(result_coco[i])):
            if result_coco[i][j] >= min_confirm:
                sum += 1
        if yolo_coco.class_names[i] in yolo_coco.chosen_classes:
            result_dict[yolo_coco.class_names[i]] = sum

    # 加入bdd类
    for i in range(0, len(result_bdd)):
        sum = 0
        for j in range(0, len(result_bdd[i])):
            if result_bdd[i][j] >= min_confirm:
                sum += 1
        if yolo_bdd.class_names[i] in yolo_bdd.chosen_classes:
            result_dict[yolo_bdd.class_names[i]] = sum

    # 从大到小排序输出到文本
    count_file = open(count_file_root, 'w')
    count_file.write('count information: ' + '\n')
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
