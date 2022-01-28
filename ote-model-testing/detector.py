import cv2
import time
import os
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(420)

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

            timeDiff = time.time() - now
            if (timeDiff < 1.0/(fps)):
                time.sleep(1.0/(fps) - timeDiff)

class Detector:

    def __init__(self):
        pass

    def read_classes(self, classes_path):
        with open(classes_path, 'r') as file:
            self.classes_list = file.read().splitlines()

        self.color_list = np.random.uniform(
            low=0, high=255, size=(len(self.classes_list), 3))

    def get_model(self, model_url):
        file = os.path.basename(model_url)
        self.model_name = file[:file.index('.')]

        self.cache_dir = './pretrained_models'
        os.makedirs(self.cache_dir, exist_ok=True)

        get_file(fname=file, origin=model_url, cache_dir=self.cache_dir,
                 cache_subdir='checkpoints', extract=True)

    def load_model(self):
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(
            self.cache_dir, "checkpoints", self.model_name, "saved_model"))

    def create_bbox(self, image,conf_treshold=0.5,overlay_treshold=0.5):
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor,dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis,...]
        
        detections = self.model(input_tensor)
        
        bboxs = detections['detection_boxes'][0].numpy()
        class_indexes = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detections['detection_scores'][0].numpy()
        
        im_h,im_w,im_c=image.shape
        
        bbox_idx=tf.image.non_max_suppression(bboxs,class_scores,max_output_size=50,iou_threshold=overlay_treshold,score_threshold=conf_treshold)
        
        
        car_number=0
        if len(bbox_idx) != 0:
            for i in bbox_idx:
                bbox=tuple(bboxs[i].tolist())
                class_confidence= round(100*class_scores[i])
                class_index=class_indexes[i]
                
                class_label_text=self.classes_list[class_index]
                if class_label_text == 'car':
                    car_number+=1
                class_color=self.color_list[class_index]
                
                display_text=f'{class_label_text} : Confidence {class_confidence}'
                ymin,xmin,ymax,xmax=bbox
                
                xmin,xmax,ymin,ymax=(xmin*im_w,xmax*im_w,ymin*im_h,ymax*im_h)
                xmin,xmax,ymin,ymax=int(xmin),int(xmax),int(ymin),int(ymax)
                
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=class_color,thickness=2)
                cv2.putText(image,display_text,(xmin,ymin-10),cv2.FONT_HERSHEY_COMPLEX,1,class_color,thickness=2)
        print(car_number,'Cars detected')
        return image
        

    def predict_image(self, image_path,conf_treshold=0.5,overlay_treshold=0.5):
        img = cv2.imread(image_path)

        bbox_image=self.create_bbox(img,conf_treshold,overlay_treshold)
        
        cv2.imwrite(self.model_name+'.jpg',bbox_image)
        cv2.imshow('Results', bbox_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    
    def predict_video(self,video_path,conf_treshold=0.5,overlay_treshold=0.5):
        
        video_cap=cv2.VideoCapture(video_path)
        
        if (video_cap.isOpened() == False):
            print('Error opening file...')
            return 
        for frame in frame_generator(video_cap, log_fps=True):
            bbox_image=self.create_bbox(frame,conf_treshold,overlay_treshold)
            cv2.imshow('Result',bbox_image)
            key=cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
        video_cap.release()
        cv2.destroyAllWindows()
        # (success,image)=video_cap.read()
        
        # start_time=0
        
        # while success:
        #     curr_time=time.time()
            
        #     fps = 1/(curr_time-start_time)
            
        #     start_time = curr_time
            
        #     bbox_image=self.create_bbox(image,conf_treshold,overlay_treshold)
            
        #     cv2.putText(bbox_image,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
        #     cv2.imshow('Result',bbox_image)
            
        #     key=cv2.waitKey(1) & 0xFF
        #     if key == ord('q'):
        #         break
            
        #     (success,image)=video_cap.read()
        # cv2.destroyAllWindows()
            
            
        
