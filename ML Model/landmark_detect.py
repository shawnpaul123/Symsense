import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model,model_from_json
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
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

'''




class detection_ml:

    def __init__(self):
        #read where models are
        self.model_face_loc = './models/face_detector'
        self.model_landmark_loc = r"C:\Users\shawn paul\Desktop\SymSense\ML Model\Symsense\ML Model\models\LFBmodel.yaml"
        self.model_mask_loc = './models/mask_detector'
        self.prototxtPath = os.path.sep.join([self.model_face_loc, "deploy.prototxt"])
        self.weightsPath = os.path.sep.join([self.model_face_loc,"res10_300x300_ssd_iter_140000.caffemodel"])
        #load all the models
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
        self.maskNet = load_model(self.model_mask_loc)
        self.confidence = 0.5
        self.LBFModel = None


    def load_landmark_model(self):
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(self.model_landmark_loc)




    #detect whether or not face is present
    #inps a resized frame//image of 400
    #outs - faces and their locations
    def detect_face(self,frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []


        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            #detections comes from facenet

            conf = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence

            if conf > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        return (faces,locs)


        


    def detect_mask(self,faces):

        if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)

        else:
            preds = []


        return preds
            


    #frame is the pic and face are the coordinates
    def add_landmarks(self,frame,face_coords):
        
        x, landmarks = self.landmark_detector.fit(frame,np.array(face_coords))

        return landmarks
        



    def return_stats(self):
        pass





class image_processing(detection_ml):

    def __init__(self):
        self.example_mask = ""
        self.example_no_mask = ""
        self.computer_stream = True
        super().__init__()




    def rpi_stream(self):
        #https://www.pyimagesearch.com/2015/03/30/
        #accessing-the-raspberry-pi-camera-with-opencv-and-python/
        pass


    def video_stream(self):
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")

        #init the landmark model
        self.load_landmark_model()

        if self.computer_stream:
            vs = VideoStream(src=0).start()
        # loop over the frames from the video stream
        i = 0
        while True:
            i+= 1

            if i == 1000:
                break
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            if self.computer_stream:
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            faces,locs = self.detect_face(frame)
            preds = self.detect_mask(faces)

            if len(preds) == 0:
                continue

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred,f) in zip(locs, preds,faces):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box


                #break after one iteration and save landmarks in txt file
                landmarks = self.add_landmarks(f,locs)

               


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
                
                for landmark in landmarks:
                    for x,y in landmark[0]:
                        # display landmarks on "image_cropped"
                        # with white colour in BGR and thickness 1
                        cv2.circle(frame, (x, y), 1, (255, 255, 255), 1)


            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()



    def image_test(self):
        #gets image or video option
        #runs accordn=igly
        pass


if __name__ == '__main__':
    imp = image_processing()
    imp.video_stream()



