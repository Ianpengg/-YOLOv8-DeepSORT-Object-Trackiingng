import numpy as np
import cv2
from copy import deepcopy
from utils import utils


roiPoints = np.zeros((4, 2), dtype=np.int64)
transPoints = np.zeros((4, 2), dtype=np.int64)
trans = np.zeros((4, 2), dtype=np.float32)
roi = np.zeros((4, 2), dtype=np.float32)

cap = cv2.VideoCapture('test.avi')

object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

time_count = 0  # calculate the background construct  time
frame_count = 1 # calculate the frame count
isTransform = False;  # record the transform status
ret, frame = cap.read()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while(1 and frame is not None):
    frame_count += 1
    
    frame = cv2.resize(frame, (640, 360), interpolation = cv2.INTER_AREA)

    imgCopy = deepcopy(frame)
    utils.setRoIPoint(roiPoints, roi)  # Set the 4 corners of the ROI region 
    roiImg = utils.getRoI(imgCopy, roiPoints)

    getShowImg = utils.getShow(imgCopy, roiPoints)
    cv2.imshow("getShowImg", getShowImg)

    #do the warpPerspective
    utils.settransPoints(transPoints, trans);
    if not isTransform:
        # Calculate the transform matrix
        M, mask = cv2.findHomography(roi, trans, cv2.RANSAC)  
        isTransform = True

    roiImg = cv2.warpPerspective(roiImg, M, (500, 300))  

    roiImgGray = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)
    roiImgGray = cv2.GaussianBlur(roiImgGray, (7, 7), 0)

    fgmask = object_detector.apply(roiImgGray)

    dilateImg = cv2.dilate(fgmask, kernel, )
    erodeImg = cv2.erode(dilateImg, kernel)

    blob = {'centerPoint':[], 'bottomRight': [], 'topLeft': []}
    utils.getDetecPoint(blob, roiImg, erodeImg)
    car_count = utils.getCarNumber(blob, frame_count)

    text = "Car count:" + str(car_count)
    cv2.putText(roiImg, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("roiImg", roiImg)
    ret, frame = cap.read()
    
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()