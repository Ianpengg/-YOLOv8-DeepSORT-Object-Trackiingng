import numpy as np
import cv2
from copy import deepcopy

gap = (302 - 110) / 3
left_1 = (110, 110 + gap ) # left 1 lane
left_2 = (110 + gap, 110 + gap * 2) # left 2 lane
left_3 = (110 + gap * 2, 302 ) # left 3 lane
carNumber = 0
regin_split = [140, 160, 180, 200]
isDetectedLeft_1_zoom_1 = False
isDetectedLeft_2_zoom_1 = False
isDetectedLeft_3_zoom_1 = False

isDetectedLeft_1_zoom_2 = False
isDetectedLeft_2_zoom_2 = False
isDetectedLeft_3_zoom_2 = False

def setRoIPoint(roiPoints, roi): 
	#RoI的顶点
    roiPoints[0] = (110, 127)
    roiPoints[1] = (302, 127)
    roiPoints[2] = (298, 210)
    roiPoints[3] = (20, 210)

    roi[0] = (110, 127)
    roi[1] = (302, 127)
    roi[2] = (298, 210)
    roi[3] = (20, 210)

  # roiPoints[0] = (710, 454)
  # roiPoints[1] = (1100, 454)
  # roiPoints[2] = (1277, 558)
  # roiPoints[3] = (742, 558)
  # roi[0] = (710, 454)
  # roi[1] = (1100, 454)
  # roi[2] = (1277, 558)
  # roi[3] = (742, 558)

  
def settransPoints(transPoints, trans): 
	#RoI的顶点
    transPoints[0] = (110, 127)
    transPoints[1] = (302, 127)
    transPoints[2] = (298, 210)
    transPoints[3] = (20, 210)

    trans[0] = (110, 127)
    trans[1] = (302, 127)
    trans[2] = (302, 210)
    trans[3] = (110, 210)

    # transPoints[0] = (710, 454)
    # transPoints[1] = (1100, 454)
    # transPoints[2] = (1277, 558)
    # transPoints[3] = (742, 558)

    # trans[0] = (710, 454)
    # trans[1] = (1100, 454)
    # trans[2] = (1100, 558)
    # trans[3] = (710, 558)


def getRoI(img, ppt):
    mask = np.zeros_like(img)
    roi_image = np.zeros_like(img)
    cv2.fillPoly(mask, [ppt], (255, 255, 255))
    roi_image = deepcopy(cv2.bitwise_and(img, mask))
    return roi_image

def getShow(img, ppt):
    mask = deepcopy(img)
    cv2.fillPoly(mask, [ppt], (0, 0, 255))
    mergeImg = cv2.addWeighted(img, 0.8, mask, 0.2, 0)
    return mergeImg

def getDetecPoint(blob, roiImg, erodeImg):
    contours, hierarchy = cv2.findContours(erodeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(0,0))
    
    boundRect = [cv2.boundingRect(c) for c in contours]
    
    for i in range(len(contours)):
        area = int(cv2.contourArea(contours[i]))
        if area >= 300:
            blob['centerPoint'].append(( int(boundRect[i][0] + boundRect[i][2] / 2), int(boundRect[i][1] + boundRect[i][3] / 2) ))
            blob['bottomRight'].append(( int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])))
            cv2.rectangle(roiImg, (boundRect[i][0], boundRect[i][1]), (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), (0, 255, 0), 2)
    # Draw center point and bottom right point
    for i in range(len(blob['centerPoint'])):
        cv2.circle(roiImg, blob['bottomRight'][i], 2, (0, 0, 255))
        cv2.circle(roiImg, blob['centerPoint'][i], 2, (0, 0, 255))
 
def getCarNumber(blob, frame_count):
    """
    Concept:
    1. Split the region into 3 lanes
    2. Draw the narrow region in the Roi
    3. Count the number of car centers in each narrow region of lane
      1. set a flag to check if the car is detected
      2. check the center point of the car is in the narrow region? 
      if yes, then count the number of center points and  set the flag to region 1
      3. if the center point leave the rigion, restore the flag
    """
    
    global carNumber, isDetectedLeft_1_zoom_1, isDetectedLeft_2_zoom_1, isDetectedLeft_3_zoom_1
    for i in range(len(blob['centerPoint'])):
        # Left 1 lane
        #print(blob['centerPoint'][i])
        if blob['centerPoint'][i][0] > left_1[0] and blob['centerPoint'][i][0] <= left_1[1]:
            if (not isDetectedLeft_1_zoom_1 and blob["bottomRight"][i][1] >= regin_split[0] and blob["bottomRight"][i][1] <= regin_split[1]): #region 1:150-160
                carNumber += 1
                isDetectedLeft_1_zoom_1 = True
                # carPointOnLeft_1.startPoint = Point(blob["bottomRight"][i][0], blob["bottomRight"][i][1]) #记录第一个右下角
                # carPointOnLeft_1.firstFrameCount = frame_count  #记录当前位于那一帧
            elif (blob["bottomRight"][i][1] > 190):
                isDetectedLeft_1_zoom_1 = False
        elif blob['centerPoint'][i][0] > left_2[0] and blob['centerPoint'][i][0] <= left_2[1]:
            if (not isDetectedLeft_2_zoom_1 and blob["bottomRight"][i][1] >= regin_split[0] and blob["bottomRight"][i][1] <= regin_split[1]): #region 1:150-160
                carNumber += 1
                isDetectedLeft_2_zoom_1 = True
                # carPointOnLeft_1.startPoint = Point(blob["bottomRight"][i][0], blob["bottomRight"][i][1]) #记录第一个右下角
                # carPointOnLeft_1.firstFrameCount = frame_count  #记录当前位于那一帧
            elif (blob["bottomRight"][i][1] > 190):
                isDetectedLeft_2_zoom_1 = False

        elif blob['centerPoint'][i][0] > left_3[0] and blob['centerPoint'][i][0] <= left_3[1]:
            if (not isDetectedLeft_3_zoom_1 and blob["bottomRight"][i][1] >= regin_split[0] and blob["bottomRight"][i][1] <= regin_split[1]): #region 1:150-160
                carNumber += 1
                isDetectedLeft_3_zoom_1 = True
                # carPointOnLeft_1.startPoint = Point(blob["bottomRight"][i][0], blob["bottomRight"][i][1]) #记录第一个右下角
                # carPointOnLeft_1.firstFrameCount = frame_count  #记录当前位于那一帧
            elif (blob["bottomRight"][i][1] > 190):
                isDetectedLeft_3_zoom_1 = False

    return carNumber


def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1

    return cx, cy