import os
from tensorflow.keras.models import load_model
import cv2





class variable_holder:

        def __init__(self):
            #read where models are
            self.model_face_loc = './models/face_detector'
            self.model_landmark_loc =  './models/landmark_detector'
            self.model_mask_loc = './models/mask_detector'
            self.prototxtPath = os.path.sep.join([self.model_face_loc, "deploy.prototxt"])
            self.weightsPath = os.path.sep.join([self.model_face_loc,"res10_300x300_ssd_iter_140000.caffemodel"])
            self.confidence = 0.5
            #run landmark code
            self.run_landmarks = False
            self.lm_bs = 32
            self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
            self.example_mask = ""
            self.example_no_mask = ""
            self.computer_stream = True
            self.landmark_test = './examples/example_01.jpg'
            #load all the models        
            self.maskNet = load_model(self.model_mask_loc)        
            self.landmarkNet = load_model(self.model_landmark_loc)      
            self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)

            pass