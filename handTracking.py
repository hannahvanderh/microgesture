# followed tutorial from: https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
# https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
# https://stackoverflow.com/questions/67455791/mediapipe-python-link-landmark-with-handedness
# https://docs.python.org/3/library/csv.html
# using handTracking_hannah enviroment

import cv2
import csv
import mediapipe as mp

import os
import cv2
import numpy as np
import csv

import numpy as np
from scipy.spatial import cKDTree
import torch

#TODO
#make default parameters and allow the user to pass in the gesture video name, name output files accordingly
#load and process an entire directory of videos (wild animals dataset from last semester)

def removeNewLines(value):
    return value.replace('\n',' ')

cuda = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

handInfoCsv = "/home/exx/hannah/GitProjects/microgesture/vid1/output.csv"
if os.path.exists(handInfoCsv):
  os.remove(handInfoCsv)

frameOutputDirectory = "/home/exx/hannah/GitProjects/microgesture/vid1/frames/"
if not os.path.exists(frameOutputDirectory):
    os.makedirs(frameOutputDirectory)

csvfile = open(handInfoCsv, 'w', newline='')
fieldnames = ['Frame', "Landmarks"]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handDictionary = {}
handCount = 0

#config file of settings?
    # output location
    # web cam/file
    # confidence
    # number of hands?

# web cam
#cap = cv2.VideoCapture(0)

# file
cap = cv2.VideoCapture("/data/MGDataset/Synthetic Data/Two/Two_0001.avi")

frameCount = 0
_, frame = cap.read()
h, w, c = frame.shape

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Empty camera frame, assuming end of file.")
        break
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result_hands = hands.process(framergb)

    hands_result = result_hands.multi_hand_landmarks
    handedness_result = result_hands.multi_handedness
    hands_world_result = result_hands.multi_hand_world_landmarks
  
    if handedness_result:
        print('Handedness:', handedness_result)
    if hands_result:
        for index, hand in enumerate(hands_result):
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            writer.writerow({'Frame': str(frameCount), 'Landmarks': removeNewLines(str(hand.landmark))})

    cv2.putText(frame, "Frame: " + str(frameCount), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    #cv2.imshow("Frame", frame)
    cv2.imwrite(frameOutputDirectory + "Frame" + str(frameCount) + ".png", frame)

    frameCount+=1

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

