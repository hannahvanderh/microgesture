# followed tutorial from: https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
# https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
# https://stackoverflow.com/questions/67455791/mediapipe-python-link-landmark-with-handedness
# https://docs.python.org/3/library/csv.html
# https://github.com/techfort/opencv-mediapipe-hand-gesture-recognition
# using handTracking_hannah enviroment

from pickle import FALSE, TRUE
from types import NoneType
import cv2
import csv
import mediapipe as mp

import os
import cv2
import numpy as np
import csv

import numpy as np
import torch
import glob
import copy
import itertools

from videoModel import Video

#TODO
#make default parameters and allow the user to pass in the gesture video name, name output files accordingly
#load and process an entire directory of videos (wild animals dataset from last semester)

def getLabelIndex(value):
    split = value.split("_")
    return split[-3].split("=")[1]

def removeNewLines(value):
    return value.replace('\n',' ')

def log_csv(number, landmark_list):
    csv_path = csv_path = '/home/exx/hannah/GitProjects/microgesture/processedImages.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
    return

def processHands(hands_result, save, name):
    for index, hand in enumerate(hands_result):
        landmarkArray = calc_landmark_list(image, hand)
        for landmark in pre_process_landmark(landmarkArray):
            normalized.append(landmark)
        if save == TRUE:
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            cv2.imwrite("/home/exx/hannah/GitProjects/microgesture/savedImages/" + str(name) + ".png", frame)

def calc_landmark_list(image, hand):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(hand.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

cuda = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

# inputFolder = "/data/MGDataset/real_data_224/rgb/"
#inputFolder = "/data/MGDataset/real_data_216x384/rgb/"
inputFolder = "/data/MGDataset/real_data_v2/rgb/"
# paths = glob.glob(inputFolder + "/*.png")

paths = sorted(os.listdir(inputFolder))
pathsCount = paths.__sizeof__()
print("Paths: " + str(pathsCount))

frameCount = 0
failedDetectionCount = 0
videoCount = 0
imageCount = 0
framesToGather = 5
videos = {}

failedDetections = "/home/exx/hannah/GitProjects/microgesture/failedDetections.txt"
if os.path.exists(failedDetections):
  os.remove(failedDetections)

handInfoCsv = "/home/exx/hannah/GitProjects/microgesture/processedImages.csv"
if os.path.exists(handInfoCsv):
  os.remove(handInfoCsv)

failedDetectionFile = open(failedDetections, 'w', newline='')

# initialize mediapipe
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

#config file of settings?
    # output location
    # web cam/file
    # confidence
    # number of hands?

#TODO create videos from all paths
#trim videos down to center frames
#use those frames to create the master array of landmarks
for index, img_path in enumerate(paths):
    imageCount+=1
    print("Image:" + str(imageCount) + "/" + str(pathsCount))

    split = img_path.split("_")
    keySplit = img_path.split("/")
    videoKey = keySplit[-1].rsplit("_", 1)[0]

    if videos.__contains__(videoKey):
        video = videos[videoKey]
    else:
        videoIndex = split[-2].split("=")[1]
        gestureIndex = split[-3].split("=")[1]
        participantIndex = split[-4].split("=")[1]
        video = Video(gestureIndex, participantIndex, videoIndex)
        videos[videoKey] = video
    
    if video:
        frameIndex = split[-1].split("=")[1].split(".")[0]
        video.addFrame(int(frameIndex), inputFolder + img_path)

videoCount = 0
videoLength = len(videos)
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
while videoCount < videoLength:
#for videoKey, video in videos.items():
    videoKey = list(videos)[videoCount]
    video = list(videos.values())[videoCount]

    normalized = []
    handCount = 0
    frameCount = 0
    lastResult = None

    print(videoKey)
    print("Video:" + str(videoCount) + "/" + str(videoLength))
    videoCount+=1

    items = video.trimFrames().items()
    for frameKey, frame_path in items:
        if frameCount >= 50:
            image = cv2.imread(frame_path)
            h, w, c = image.shape

            frame = cv2.flip(image, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_result = None
            tries = 0
            while hands_result == None:
                result_hands = hands.process(framergb)
                hands_result = result_hands.multi_hand_landmarks
                tries+=1

                if hands_result != None: 
                    lastResult = hands_result
                    # processHands(hands_result, frameCount == 20, videoKey)
                    processHands(hands_result, FALSE, '')
                    handCount+=1
                else:
                    if tries == 5:
                        # if lastResult != None:
                        #     processHands(lastResult, FALSE, '')

                        failedDetectionFile.write(frame_path + "\n")
                        failedDetectionCount+=1
                        if failedDetectionCount % 1 == 0:
                            cv2.imwrite("/home/exx/hannah/GitProjects/microgesture/failedDetections/" + str(failedDetectionCount) + ".png", frame)
                        
                        break
  
            if handCount == framesToGather:
                break

        frameCount+=1

    if(handCount == framesToGather):
        log_csv(video.gestureIndex, normalized)  
    else:
        print("Failed to get enough frames")  

print("Failied detections: " + str(failedDetectionCount))

