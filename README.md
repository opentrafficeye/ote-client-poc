# Open Trafic Eye (Proof of concept)

This proof of concept app was developed for the hackathon "SMART WITH DATA", organized within the activities of the National Center HPC.mk for competencies for HPC, HPDA and AI of the Republic of North Macedonia, in which it managed to be awarded second place. The fundamental idea behind this project is that people put their mobile devices on their windows which would be able to analyze the number of vehicles in the street and make conclusions of the congestion rate of that area.

The whole machine learning algorithm would be executed in the mobile device and the conclusions would then be sent to a certain backend service which will further aggregate informations from different places and compile a final conclusion for the congestion in the streets.

Models used for the object detection are from the [Tensorflow COCO2017 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md):

## [SSD MobileNet V2](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)
Mean Average Precision: 20.2%

Relative Speed: 19ms


## [SSD MobileNet V2 trained on 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)
Mean Average Precision: 28.2%

Relative speed: 39ms


## [SSD ResNet50 V1 FPN 640x640 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)
Mean Average Precision: 34.3%

Relative Speed: 46ms

## Images from the models performance:

![Paytool Petrovec](https://github.com/opentrafficeye/ote-client-poc/blob/main/petrovec_patarina.png)
![Paytool Petrovec 2](https://github.com/opentrafficeye/ote-client-poc/blob/main/petrovec_patarina_pt2.png)
![Paytool Tetovo](https://github.com/opentrafficeye/ote-client-poc/blob/main/tetovo_patarina.png)



