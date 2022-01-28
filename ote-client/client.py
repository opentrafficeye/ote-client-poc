import math
import queue
import threading
import time
from typing import Optional

import cv2

from .detector import Detector


class VideoCapture:

    def __init__(self, video_source, fps: Optional[float] = None,
                 buffer: float = 0.0):
        self.cap = cv2.VideoCapture(video_source)
        self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1 / self.fps
        self.frame_queue = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        time.sleep(buffer)
        # self.fps = self.frame_queue.qsize() / buffer
        # print('fps:', self.fps)
        self.now = time.time()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_queue.put(frame)

    def read(self, delta: float):
        q_size = self.frame_queue.qsize()
        base = self.fps * 10
        if q_size < self.fps:
            time.sleep(self.frame_time - delta % self.frame_time)
        # skip_frames = int(delta // self.frame_time)
        elif q_size > base:
            for _ in range((q_size - base) // (4 * self.fps)):
                self.frame_queue.get()

        # for _ in range(skip_frames):
        #     if self.frame_queue.qsize() < self.fps:
        #         break
        #     self.frame_queue.get()
        return self.frame_queue.get()

    def __next__(self):
        self.delta = time.time() - self.now
        self.now = time.time()
        return self.read(self.delta)

    def __iter__(self):
        return self


def main():
    VIDEO_SOURCE = 'http://192.168.0.110:4747/video?type=some.mjpeg'
    # VIDEO_SOURCE = 'http://droidcam.martinkozle.com/video?type=some.mjpeg'
    # VIDEO_SOURCE = 'http://streaming1.neotel.net.mk:8080/hls/petrovec_2.m3u8'
    CLASS_FILEPATH = 'coco.names'
    MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
    CONF_TRESHOLD = 0.5
    OVERLAY_TRESHOLD = 0.5

    detector = Detector()
    detector.read_classes(classes_path=CLASS_FILEPATH)
    detector.get_model(MODEL_URL)
    detector.load_model()

    capture = VideoCapture(VIDEO_SOURCE, buffer=3, fps=30)
    for frame in capture:
        print(f'{capture.delta:.2f}', capture.frame_queue.qsize())

        bbox_image = detector.create_bbox(
            frame, CONF_TRESHOLD, OVERLAY_TRESHOLD)
        cv2.imshow('Result', bbox_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
