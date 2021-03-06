import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
from variables import variable_holder, detection_ml

#cd desktop/symsense/ml model/symsense/ml model
#python -m pip install --user opencv-contrib-python



'''
credits: https://github.com/Danotsonof/facial-landmark-detection/blob/master/facial-landmark.ipynb
where models were got from
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
'''

'''
update:
dont use opencv use tensorflow to train landmark detection so you know the probabilities
https://github.com/alexellis/pizero-docker-demo/tree/master/iotnode
https://www.kaggle.com/c/facial-keypoints-detection/data
https://towardsdatascience.com/detecting-facial-features-using-deep-learning-2e23c8660a7a
open network streaming: https://www.pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/

'''
class image_processing(detection_ml):

    def __init__(self):
        super().__init__()
        

    def rpi_stream(self):
        #https://www.pyimagesearch.com/2015/03/30/
        #accessing-the-raspberry-pi-camera-with-opencv-and-python/
        pass


    def plot_image(self):
        pass


    def mvp_video_stream(self):
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")

     
     

        if self.computer_stream:
            vs = VideoStream(src=0).start()
        # loop over the frames from the video stream
     
        while True:
            
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            #########################################################################################
            if self.computer_stream:
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not

            else:
                pass
                #this is where you have code to get video stream from rpi


            #########################################################################################


            #!This is where you would have to send/receive a request/response
            faces,locs = self.detect_face(frame)
            preds = self.detect_mask(faces)


            if self.run_landmarks:
                landmarks = self.detect_landmarks(faces)


            if len(preds) == 0:
                continue

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred,face) in zip(locs, preds,faces):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box

                #get back the landmarks for each face
                (mask, withoutMask) = pred
                
                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                '''

                if self.run_landmarks:
                
                    for landmark in landmarks:
                        for x,y in landmark[0]:
                            # display landmarks on "image_cropped"
                            # with white colour in BGR and thickness 1
                            cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)


                '''


            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()





if __name__ == '__main__':
    imp = image_processing()
    imp.mvp_video_stream()
    
    



