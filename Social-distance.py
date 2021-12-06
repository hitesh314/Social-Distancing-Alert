import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
image = cv2.imread("pedestrians.jpg")
image = imutils.resize(image, width=min(600, image.shape[1]))
orig = image.copy()

# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4,4), padding=(4, 4), scale=1.03)

# apply non-maxima suppression to the bounding boxes
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)


# draw the bounding boxes and calculate center of each box
centers=[]
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    xC=xA+int((xB-xA)/2)
    yC=yA+int((yB-yA)/2)
    centers.append((xC,yC))
    
# calculate distances of bounding boxes centers
# and add them to a list if they are close enough
# and their y-coordinates are in a range
nears=[]
minDistance=image.shape[1]//6

for i,p1 in enumerate(centers):
    for j,p2 in enumerate(centers[i+1:]):
        if (np.linalg.norm(np.array(p2)-np.array(p1)))<minDistance:
            if abs(p1[1] - p2[1])< image.shape[0]*0.1:
                nears.append((i,j+i+1))

# draw red bounding boxes for close people              
for (i,j) in nears:
    (xA, yA, xB, yB) = pick[i]
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
    (xA, yA, xB, yB) = pick[j]
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
    

# show the output images
cv2.imshow("Alerts", image)
if cv2.waitKey(0):
    cv2.destroyAllWindows()
 
