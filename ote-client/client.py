import queue
import threading
import time

import cv2


class VideoCapture:

    def __init__(self, video_source):
        self.cap = cv2.VideoCapture(video_source)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def frame_generator(capture: cv2.VideoCapture, log_fps: bool = False):
    frames_passed = 0
    last_time = time.time() - 1e-5
    fps = capture.get(cv2.CAP_PROP_FPS)

    while(True):
        now = time.time()
        if log_fps:
            frames_passed += 1
            if time.time() - last_time > 2:
                print(f'fps: {frames_passed / (time.time() - last_time):.2f}')
                frames_passed = 0
                last_time = time.time()

        yield capture.read()[1]

        time_diff = time.time() - now
        if (time_diff < 1.0/(fps)):
            time.sleep(1.0/(fps) - time_diff)


def main():
    # VIDEO_SOURCE = 'https://droidcam.martinkozle.com/video?type=some.mjpeg'
    VIDEO_SOURCE = 'http://streaming1.neotel.net.mk:8080/hls/petrovec_2.m3u8'
    capture = cv2.VideoCapture()
    capture.open(VIDEO_SOURCE)

    for frame in frame_generator(capture, log_fps=True):
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
