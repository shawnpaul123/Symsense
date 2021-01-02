import cv2
import os

'''
credits: https://github.com/Danotsonof/facial-landmark-detection/blob/master/facial-landmark.ipynb
where models were got from
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
'''





class image_processing:

	def __init__(self):
		pass


	def image_preprocess(self):
		pass




	#check of video or image stream
	# if image -> get pased image url and pass image 
	#and return with mask no maks + landmark detection
	#if video, check if camera -> true then
	#stream based on camera type video to the landmark detection function
	def image_video_stream():
		pass







class detection_ml:

	def __init__(self):
		#read where models are
		self.model_face_loc = './models/face_detector'
		self.model_landmark_loc = ""
		self.model_mask_loc = './models/mask_detector'
		self.prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
		self.weightsPath = os.path.sep.join([face,"res10_300x300_ssd_iter_140000.caffemodel"])
		#load all the models
		self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
		self.maskNet = load_model(self.model_mask_loc)
		self.confidence = 0.5

	#detect whether or not face is present
	#inps a resized frame//image of 400
	#outs - faces and their locations
	def detect_face(self,frame,self.faceNet,self.maskNet):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
		# pass the blob through the network and obtain the face detections
		self.faceNet.setInput(blob)
		detections = faceNet.forward()
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
			preds = maskNet.predict(faces, batch_size=32)


		return preds
			

	def add_landmarks(self):
		pass



	def return_stats(self):
		pass