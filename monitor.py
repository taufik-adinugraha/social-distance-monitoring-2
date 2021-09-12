# import the necessary packages
import detection
import numpy as np
import argparse
import imutils
import cv2
import time
import  matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = 'models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = 'models/yolov3.weights'
configPath = 'models/yolov3.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = int(vs.get(cv2.CAP_PROP_FPS))

# loop over the frames from the video stream
avg_violate_per_second, step = 0, 0
avg_violate_per_mnt = []
percent = []
# initialize the set of indexes that violate the minimum social distance
violate = set()
t1 = time.time()
iframe = 0
while True:
	iframe += 1
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)

	t2 = time.time()
	if t2-t1 > 2.25:
		t1 = t2

		objects = ['person']
		results = detection.detect_object(frame, net, ln, Idxs=[LABELS.index(i) for i in objects if LABELS.index(i) is not None])

		# initialize the set of indexes that violate the minimum social distance
		violate = set()

		# ensure there are *at least* two people detections (required in order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results
			centroids = np.array([r[3] for r in results])
			# get the widths of bounding boxes
			dXs = [r[2][2]-r[2][0] for r in results]

			for i in range(len(results)):
				c1 = centroids[i]
				for j in range(i + 1, len(results)):
					c2 = centroids[j]
					Dx, Dy = np.sqrt((c2[0]-c1[0])**2), np.sqrt((c2[1]-c1[1])**2)
					thresX = (dXs[i] + dXs[j]) * 0.7
					thresY = (dXs[i] + dXs[j]) * 0.25
					# check to see if the distance between any pairs is less than the threshold
					if Dx<thresX and Dy<thresY:
						# update our violation set with the indexes of the centroid pairs
						violate.add(i)
						violate.add(j)

		# loop over the results
		for (i, (classID, prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			dX, dY = endX-startX, endY-startY
			(cX, cY) = centroid[0], centroid[1]-dY//2
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then update the color
			if i in violate:
				color = (0, 0, 255)

			# draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			# cv2.circle(frame, (cX, cY), 4, (255,255,255), -1)
			# cv2.imwrite(f'{iframe}.jpg', frame)

		# calculate the average of social distancing violations per minute
		if len(results) != 0:
			percent.append(len(violate) / len(results) * 100)
		step += 1
		if step == 10:
			step = 0
			if percent:
				out = np.mean(percent)
			else:
				out = 0
			avg_violate_per_mnt.append(out)
			percent = []
			fig, ax = plt.subplots(figsize=(12, 3))
			sns.lineplot(range(len(avg_violate_per_mnt)), avg_violate_per_mnt)
			sns.scatterplot(range(len(avg_violate_per_mnt)), avg_violate_per_mnt)
			ax.set_ylabel("avg violation per 30 s (%)")
			ax.set_xticks([])
			ax.set_yticks([0, 25, 50, 75, 100])
			fig.savefig("tmp.png")

	fig = cv2.imread("tmp.png")
	fig = imutils.resize(fig, width=700)
	frame = np.vstack((frame,fig))

	# check to see if the output frame should be displayed to our screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)
