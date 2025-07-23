import cv2
import numpy as np
from typing import Tuple
from sklearn.cluster import DBSCAN
import warnings
import time

warnings.filterwarnings('ignore')


class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_distance = 100
        self.pixels_per_meter = 17

    def update(self, centroids):
        current_time = time.time()
        new_tracked_objects = {}

        if self.tracked_objects:
            for centroid in centroids:
                min_dist = float('inf')
                matching_id = None

                for obj_id, obj_data in self.tracked_objects.items():
                    dist = np.linalg.norm(np.array(centroid) - np.array(obj_data['positions'][-1]))
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        matching_id = obj_id

                if matching_id is not None:
                    obj_data = self.tracked_objects[matching_id]
                    obj_data['positions'].append(centroid)
                    obj_data['timestamps'].append(current_time)

                    if len(obj_data['positions']) >= 2:
                        time_diff = obj_data['timestamps'][-1] - obj_data['timestamps'][-2]
                        if time_diff > 0:
                            dist = np.linalg.norm(
                                np.array(obj_data['positions'][-1]) -
                                np.array(obj_data['positions'][-2])
                            )
                            speed = (dist / self.pixels_per_meter) / time_diff
                            speed_kph = speed * 3.6
                            obj_data['speed'] = speed_kph

                            # Update average speed
                            obj_data['total_speed'] += speed_kph
                            obj_data['speed_count'] += 1
                            obj_data['average_speed'] = obj_data['total_speed'] / obj_data['speed_count']

                    new_tracked_objects[matching_id] = obj_data
                else:
                    new_tracked_objects[self.next_id] = {
                        'positions': [centroid],
                        'timestamps': [current_time],
                        'speed': 0,
                        'total_speed': 0,
                        'speed_count': 0,
                        'average_speed': 0
                    }
                    self.next_id += 1
        else:
            for centroid in centroids:
                new_tracked_objects[self.next_id] = {
                    'positions': [centroid],
                    'timestamps': [current_time],
                    'speed': 0,
                    'total_speed': 0,
                    'speed_count': 0,
                    'average_speed': 0
                }
                self.next_id += 1

        self.tracked_objects = new_tracked_objects
        return self.tracked_objects


class ObjectDetector:
    def __init__(self, min_area: int = -1):
        self.min_area = min_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.morph_kernel = np.ones((11, 11), np.uint8)
        self.tracker = ObjectTracker()
        self.frame_count = 0
        self.skip_frames = 1

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.morph_kernel)
        return thresh

    def interpolate_edges(self, contours, spacing=25):
        points = []
        for i in range(len(contours) - 1):
            pt1 = contours[i]
            pt2 = contours[i + 1]
            points.append(pt1)

            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            num_points = max(2, int(dist / spacing))

            for j in range(1, num_points):
                offset = j / num_points
                point = pt1 + offset * (pt2 - pt1)
                points.append(point.astype(int))

        points.append(contours[-1])  # last point
        return np.array(points)

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        display_frame = frame.copy()

        fg_mask = self.remove_background(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sobelxy = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=5)
        edges = cv2.Canny(cv2.convertScaleAbs(sobelxy), 40, 80)
        filtered = cv2.bitwise_and(edges, edges, mask=fg_mask)

        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        centroids = []

        for contour in contours:
            contour_points = contour.reshape(-1, 2)
            interpolated = self.interpolate_edges(contour_points)
            points.extend(interpolated)

        num_objects = 0
        if points:
            points = np.array(points).reshape(-1, 2)
            clustering = DBSCAN(eps=30, min_samples=220).fit(points)
            labels = clustering.labels_
            num_objects = len(set(labels)) - (1 if -1 in labels else 0)

            for label in set(labels):
                if label != -1:
                    mask = labels == label
                    cluster_points = points[mask]

                    if len(cluster_points) >= 3:
                        hull = cv2.convexHull(cluster_points.astype(np.float32))

                        M = cv2.moments(hull)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            centroids.append((cx, cy))

                        cv2.polylines(display_frame, [hull.astype(np.int32)], True, (0, 255, 0), 2)

        tracked_objects = self.tracker.update(centroids)

        for obj_data in tracked_objects.values():
            if obj_data['positions']:
                pos = obj_data['positions'][-1]
                speed_kph = obj_data['speed']
                avg_speed_kph = obj_data['average_speed']

                # Display current speed and average speed on the frame
                cv2.putText(display_frame, f"Speed: {speed_kph:.1f} km/h",
                            (int(pos[0]) - 30, int(pos[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.putText(display_frame, f"Avg Speed: {avg_speed_kph:.1f} km/h",
                            (int(pos[0]) - 30, int(pos[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display_frame, num_objects

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % self.skip_frames == 0:
                display_frame, num_objects = self.detect_objects(frame)
                cv2.putText(display_frame, f"Objects: {num_objects}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Object Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = ObjectDetector(min_area=-1)
    #current one is for the good results
    video_path = "CP1/BLK-HDPTZ12 Security Camera Parkng Lot Surveillance Video - Supercircuits (720p, h264, youtube) (online-video-cutter.com).mp4"
    detector.process_video(video_path)


if __name__ == "__main__":
    main()