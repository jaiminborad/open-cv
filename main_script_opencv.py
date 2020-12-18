# script that contains all the subscripts of all the scripts

#------------importing the modules ----------#

import cv2
import numpy as np
from os import getcwd
from scipy.spatial import distance as dist

#-------------------------------------------#

#---------defining the thresholds----------#

Conf_thres_social_distance = 0.2
social_distance = 300 	# measured in pixels
Conf_thres_crowd = 0.2
count_thres_crowd = 8

#-------------------------------------------#

#--------defining the directories to import various deep learning modules -----#
# we are taking the relative path w.r.t present working directory

CWD = getcwd()
video_files = CWD + '/video_files/'
coco_name = CWD + '/models_weights/coco.names'
yolo_weight = CWD + '/models_weights/yolov2-tiny.weights'
yolo_cfg = CWD + '/models_weights/yolov2-tiny.cfg'
caffe_prototxt = CWD + '/models_weights/deploy.prototxt.txt'
caffe_model = CWD + '/models_weights/model_300x300_ssd_iter_140000.caffemodel'
#-----------------------------------------------------------------------------#


# function to detect the motion 

def motion_detection():

	# read the video file-------------specify the full path or relative path 
	# with respect to current project directory
	
	# loading the address of video we want to test 
	test_video = video_files + 'test_video.mp4'
	cap = cv2.VideoCapture(test_video)

	# reading the 2 initial frames 

	_,frame1 = cap.read()
	_,frame2 = cap.read()


	while cap.isOpened():
		#finding difference between 2 frame to find any contours

		diff = cv2.absdiff(frame1, frame2)
		gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (7,7),0)
		_,threshold = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
		dilated = cv2.dilate(threshold, None, iterations=3)
		contours ,_ = cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			(x, y, w, h) = cv2.boundingRect(contour)

			if cv2.contourArea(contour) < 1500:
				continue
			cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 3)
			cv2.putText(frame1, "Status: {}".format('Movement'), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
						3, (0, 0, 255), 2)
		#cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)----for debugging purpose 
		
		end_frame = cv2.resize(frame1, (1100,600))

		
		frame1 = frame2
		ret, frame2 = cap.read()

		# encoding the frame to view it in browser and yield will continously send frames 
		# to the webpage 
		#cv2.imshow('feed',end_frame)
		end_frame = cv2.imencode('.jpg',end_frame)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + end_frame + b'\r\n')

		if cv2.waitKey(40) == 27:	# on pressing escape key it will exit the fucntion
			break

	cv2.destroyAllWindows()
	cap.release()


# function for finding the social distance in the footage

def social_distance_detector():

	with open(coco_name,'rt') as f:
		names = f.read().rstrip('\n').split('\n')


	#load config and weight file 
	net = cv2.dnn.readNetFromDarknet(yolo_cfg,yolo_weight)

	#start taking video as input
	test_video = video_files + 'pedestrians.mp4'
	cap = cv2.VideoCapture(test_video)
	
	while cap.isOpened():
		_,image = cap.read()
		#image = cv2.resize(image,Csize)

		img_height = image.shape[0]
		img_width = image.shape[1]

		#getting blob which will normalize the colors of all 3 channels

		blob = cv2.dnn.blobFromImage(image,1/255,swapRB=True)
		net.setInput(blob)	# give the blob as input to the network

		layernames = net.getLayerNames()
		layernames = [layernames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		output = net.forward(layernames)
		#will contain the coordinates of centroid of detected persons
		centroid=[] 
		result=[]

		#carrying out the detection

		for out in output:
			for detection in out:
				scores = detection[5:]	# gives the list containing scores of all classes
				classId = np.argmax(scores)
				confidence = scores[classId]

				if classId == 0 and confidence > Conf_thres_social_distance:
					box = detection[0:4] * np.array([img_width,img_height,img_width,img_height])
					centre_x,centre_y,width,height = box.astype("int")
					x = int(centre_x - (width/2))
					y = int(centre_y - (height/2))

					result.append(((x,y,x + width,y+height),(centre_x,centre_y)))
					centroid.append((centre_x,centre_y))			

		#applying non maxima supression
		#idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.3)
		#print(idxs)
		# now we are calculating the social distancing distance 
		# using the distance between centroids
		
		#---------------------------------------------------------------------
		violate = set() #set of indices that violate social distance
		Distance = dist.cdist(centroid,centroid,metric="euclidean")

		for i in range(0,Distance.shape[0]):
			for j in range(i+1,Distance.shape[1]):
				if Distance[i,j] < social_distance:
					violate.add(i)
					violate.add(j)
		
		#extracting the bounding box and centroids:
		for (i, (bbox,centroid)) in enumerate(result):
			
			color = (0,255,0)
			if i in violate:
				color = (0,0,255)
			
			#drawing rectangles

			cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,2)
		
		text = "social distance violations: {}".format(len(violate))
		cv2.putText(image,text,(10, image.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

		end_frame = cv2.resize(image, (1100,600))
		end_frame = cv2.imencode('.jpg',end_frame)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + end_frame + b'\r\n')

		if cv2.waitKey(40) & 0xFF == 27:
			break

		# we ought to perform non maximum supression but we are not doing it
		# for sake of siplicity of the program 

	cv2.destroyAllWindows()
	cap.release()


#for people detection we are using SSD [single shot detector]
# the model is not very accurate but it is fast
# in real world camera with GPUs are used for fast processing 
# here we are doing all the calculations on CPU which reduces 
# our accuracy as well as speed of FPS
def people_counter():
	PeopleInFrame = 0
	net = cv2.dnn.readNetFromCaffe(caffe_prototxt,caffe_model)

	# load the address of test video 
	test_video = video_files + 'test_video.mp4'
	video = cv2.VideoCapture(test_video)

	while video.isOpened():
		temp,frame = video.read()
		blob = cv2.dnn.blobFromImage(frame,3)
		net.setInput(blob)
		detections = net.forward()
		PeopleInFrame = 0
		image_height , image_width = frame.shape[:2]


		for i in range(0, detections.shape[2]):
			confidence = detections[0,0,i,2]

			if confidence > Conf_thres_crowd:
				PeopleInFrame += 1


		if PeopleInFrame > count_thres_crowd:
			color = (0,0,255)
		else:
			color = (0,255,0)

		text = " people in frame : {}".format(PeopleInFrame)
		cv2.putText(frame, text, (10,50),cv2.FONT_HERSHEY_SIMPLEX,2,color,3)

		#cv2.imshow('feed',frame)
		end_frame = cv2.resize(frame, (1100,600))
		end_frame = cv2.imencode('.jpg',end_frame)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + end_frame + b'\r\n')

		if cv2.waitKey(40) & 0xFF == 27:
			break
	cv2.destroyAllWindows()
	video.release()


# for debugging purpose only
# inp = int(input("press the key \n "))
# if inp == 1:
# 	motion_detection()
# elif inp == 2:
# 	social_distance_detector()
# elif inp == 3:
# 	people_counter()



	