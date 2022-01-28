from detector import *

CLASS_FILEPATH='coco.names'
#MODEL_URL='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
MODEL_URL='http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
IMG_PATH='C:\\Users\\Tomislav\\Pictures\\Camera Roll\\tal-munchen-3-915x596.jpg'
#IMG_PATH='D:\\Custom made programs\\Python\\hackathon\\ote-client-poc\\ote-model-testing\\skopje_slika.jpeg'
VIDEO_SOURCE = 'http://streaming1.neotel.net.mk:8080/hls/petrovec_2.m3u8'

detector=Detector()
detector.read_classes(classes_path=CLASS_FILEPATH)
detector.get_model(MODEL_URL)
detector.load_model()
detector.predict_image(IMG_PATH,conf_treshold=0.6)
#detector.predict_video(VIDEO_SOURCE)