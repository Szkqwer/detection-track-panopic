# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, min_confirm=5, classNum=80, cos_dis_rate=0.9):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.min_confirm = min_confirm

        self.cos_dis_rate = cos_dis_rate

        self.classNum = classNum
        # 储存所有track出现帧数 0为未确定
        self.confirmed_time_list = []
        # 储存被确定的track的id
        self.confirmed_id_list = []
        for i in range(0, classNum):
            self.confirmed_time_list.append([])
            self.confirmed_id_list.append([])

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 获取对应ID
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            # 更新出现次数
            self.confirmed_time_list[self.tracks[track_idx].classIndex][self.tracks[track_idx].objectIndex] += 1
            # 被确认的次数大于阈值则添加
            if self.confirmed_time_list[self.tracks[track_idx].classIndex][
                self.tracks[track_idx].objectIndex] >= self.min_confirm:
                if self.tracks[track_idx].track_id not in self.confirmed_id_list[self.tracks[track_idx].classIndex]:
                    self.confirmed_id_list[self.tracks[track_idx].classIndex].append(self.tracks[track_idx].track_id)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        temp = []
        for t in self.tracks:
            if not t.is_deleted():
                temp.append(t)

        self.tracks = temp

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 计算cos距离并将大于阈值的设为无穷
            cos_cost_matrix = self.metric.distance(features, targets)
            cos_cost_matrix[cos_cost_matrix > self.metric.matching_threshold] = 10e+5
            # 计算马氏距离并将大于阈值的设为无穷
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, np.array(cos_cost_matrix), tracks, dets, track_indices,
                detection_indices,cos_dis_rate=self.cos_dis_rate)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.

        # 获取每一类的tracks和detections
        temp_tracks = []
        temp_detections = []
        for classes in range(0, self.classNum):
            temp_tracks.append([])
            temp_detections.append([])
        for i in range(0, len(self.tracks)):
            temp_tracks[self.tracks[i].classIndex].append(self.tracks[i])
        for i in range(0, len(detections)):
            temp_detections[detections[i].classIndex].append(detections[i])

        # 对于同类的tracks和detections查找对应关系
        matches = []
        unmatched_tracks = []
        unmatched_detections = []
        for class_index in range(0, self.classNum):
            confirmed_tracks = [
                i for i, t in enumerate(temp_tracks[class_index]) if t.is_confirmed()]
            unconfirmed_tracks = [
                i for i, t in enumerate(temp_tracks[class_index]) if not t.is_confirmed()]


            # 对于已经确认的使用cos距离匹配
            matches_a, unmatched_tracks_a, unmatched_detections_a = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                temp_tracks[class_index], temp_detections[class_index], confirmed_tracks)

            # IOU.
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if
                temp_tracks[class_index][k].time_since_update == 1]

            # 未出现设为为匹配
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                temp_tracks[class_index][k].time_since_update != 1]

            matches_b, unmatched_tracks_b, unmatched_detections_b = linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, temp_tracks[class_index],
                temp_detections[class_index], iou_track_candidates, unmatched_detections_a)

            # 合并
            temp_matches = matches_a + matches_b
            temp_unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            temp_unmatched_detections = list(set(unmatched_detections_a + unmatched_detections_b))
            for track_idx, detection_idx in temp_matches:
                original_track_idx = self.tracks.index(temp_tracks[class_index][track_idx])
                original_detection_idx = detections.index(temp_detections[class_index][detection_idx])
                matches.append(tuple([original_track_idx, original_detection_idx]))

            for i in range(0, len(temp_unmatched_tracks)):
                original_track_idx = self.tracks.index(temp_tracks[class_index][temp_unmatched_tracks[i]])
                unmatched_tracks.append(original_track_idx)

            for i in range(0, len(temp_unmatched_detections)):
                original_detection_idx = detections.index(temp_detections[class_index][temp_unmatched_detections[i]])
                unmatched_detections.append(original_detection_idx)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            classIndex=detection.classIndex,
            objectIndex=len(self.confirmed_time_list[detection.classIndex]),
            feature=detection.feature,
            time_cross=10
        ))
        self._next_id += 1
        self.confirmed_time_list[detection.classIndex].append(0)
