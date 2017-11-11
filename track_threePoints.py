# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py
# OpenCV3 required! There are a few lines below that will not work in OpenCV2
# This script is configured to track two points and return the coordinate values and a distance calculation

# import the necessary packages

# import sys
# sys.path.insert(0, 'Libraries/pyimagesearch')
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# Parameters for operation
camera_address = 1 # 1 for the USB webcam, 0 for the onboard webcam
minRadius = 7 # Minimum radius of the circle
outputFilename = "track3point.txt"

# define the lower and upper boundaries of the "green"
# ball in the HSV (RGB??) color space, then initialize the
# list of tracked points
#greenLower = (0,98,114)			# Red tracker in Surrey 4040
#greenUpper = (214,160,204)	
#greenLower = (156,142,139) # Webcam on table in Menrva
#greenUpper = (186,247,255)	

greenLower = (68,89,67)			# Track Green in Surrey 3666, lights on
greenUpper = (113,220,130)		
tLower1 = (68,89,67)			# Track Green in Surrey 3666, lights on
tUpper1 = (113,220,130)	
tLower2 = (87,96,110)			# Track Blue Surrey 3666
tUpper2 = (132,221,219)	

pts = deque(maxlen=args["buffer"])
#Green Egg Works better than the Pink one

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(camera_address)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size = (640,480)
video = cv2.VideoWriter('output.avi',fourcc, 30.0, size)
# Define and Open coordinates text file
text_file = open(outputFilename, "w")
text_file.close()
x=0
y=0
x2=0
y2=0
x3=0
y3=0
distance=0
msgOut = ''
msgOut2 = ''


time.sleep(2.5)

# keep looping
while True:
	
	# grab the current frame
	(grabbed, frame) = camera.read()
	text_file = open(outputFilename, "a")

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	#Flip the frame
	frame = cv2.flip(frame, 1)

	# write the frame
	#video.write(frame)
	
	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=600) #This line of code seems to prevent the video.write from working later on
	
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# bgr = image # Alternatively we can use RGB Space

	# write the frame
	#video.write(frame)
	
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	
	mask1 = cv2.inRange(hsv, tLower1, tUpper1)
	mask1 = cv2.erode(mask, None, iterations=2)
	mask1 = cv2.dilate(mask, None, iterations=2)
	
	mask2 = cv2.inRange(hsv, tLower2, tUpper2)
	mask2 = cv2.erode(mask, None, iterations=2)
	mask2 = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2] # The [-2] grabs the list object
	cnts1 = sorted(cnts1, key=cv2.contourArea, reverse=True) # Sort by area
	center1 = None
	center2 = None
	
	cnts3 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2] # The [-2] grabs the list object
	cnts3 = sorted(cnts3, key=cv2.contourArea, reverse=True) # Sort by area
	center3 = None

	# only proceed if at least one contour was found
	if len(cnts1) > 1:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c1 = cnts1[0]
		((x, y), radius) = cv2.minEnclosingCircle(c1)
		M = cv2.moments(c1)
		center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# Grab the second largest contour
		
		c2 = cnts1[1]
		((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
		M = cv2.moments(c2)
		center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		distance = np.sqrt( (x2-x)**2 + (y2-y)**2)

#		msgOut = "{X: %.3f, Y: %.3f, X2: %.3f, Y2: %.3f, Dist: %.3f} \n" % (x, y, x2, y2, distance)
		msgOut = "%.2f,%.2f,%.2f,%.2f,%.3f" % (x, y, x2, y2, distance) 
#		print(msgOut)
#		print('\n')
#		text_file.write(msgOut)

		# only proceed if the radius meets a minimum size
		if radius > minRadius:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center1, 5, (0, 0, 255), -1)
			
		if radius2 > minRadius:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x2), int(y2)), int(radius2),
				(0, 150, 150), 2)
			cv2.circle(frame, center2, 5, (0, 0, 150), -1)

	# update the points queue
	pts.appendleft(center1)
	pts.appendleft(center2)
	
	# Check contours on our second threshold
	if len(cnts3) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c3 = cnts3[0]
		((x3, y3), radius3) = cv2.minEnclosingCircle(c3)
		M = cv2.moments(c3)
		center3 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		
		msgOut2= "{X: %.3f, Y: %.3f,} \n" % (x3, y3)
#		msgOut = "%.3f,%.3f,%.3f,%.3f,%.3f\n" % (x, y, x2, y2, distance) 
#		print(msgOut2)
#		text_file.write(msgOut)

		# only proceed if the radius meets a minimum size
		if radius3 > minRadius:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x3), int(y3)), int(radius3),
				(0, 255, 0), 2)
			cv2.circle(frame, center3, 5, (255, 0, 255), -1)
			
		
	# loop over the set of tracked points and draw a line
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if len(cnts1) > 1 or len(cnts3) > 0:
		msgFull = "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (x, y, x2, y2, distance ,x3, y3)
		print msgFull
		text_file.write(msgOut)
	
	# write the frame
	video.write(frame)

	text_file.close()
	time.sleep(0.07725)
	
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
video.release()
cv2.destroyAllWindows()
text_file.close()
