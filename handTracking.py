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
import torch
import glob


#TODO
#make default parameters and allow the user to pass in the gesture video name, name output files accordingly
#load and process an entire directory of videos (wild animals dataset from last semester)

def getLabelIndex(value):
    split = value.split("_")
    return split[-3].split("=")[1]


def removeNewLines(value):
    return value.replace('\n',' ')

cuda = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

inputFolder = "/data/MGDataset/real_data_224/rgb/"
paths = glob.glob(inputFolder + "/*.png")
print("Paths: " + str(paths.__sizeof__()))
frameCount = 0
failedDetectionCount = 0

failedDetections = "/home/exx/hannah/GitProjects/microgesture/failedDetections.txt"
if os.path.exists(failedDetections):
  os.remove(failedDetections)

handInfoCsv = "/home/exx/hannah/GitProjects/microgesture/processedImages.csv"
if os.path.exists(handInfoCsv):
  os.remove(handInfoCsv)

failedDetectionFile = open(failedDetections, 'w', newline='')
csvfile = open(handInfoCsv, 'w', newline='')
fieldnames = ['Label', "Name", "Landmarks"]
writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter =',', quoting=csv.QUOTE_NONE,
                            escapechar=' ')
writer.writeheader()

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

#config file of settings?
    # output location
    # web cam/file
    # confidence
    # number of hands?

for img_path in paths:
    image = cv2.imread(img_path)
    h, w, c = image.shape
    frame = cv2.flip(image, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(framergb)

    hands_result = result_hands.multi_hand_landmarks

    if hands_result:
        landmarkArray = ''
        for index, hand in enumerate(hands_result):
            for lm in hand.landmark:
                # do I need to normalize these values?
                landmarkArray += str(lm.x) + ","
                landmarkArray += str(lm.y) + ","
                #count these to make sure they are right, remove filepath from csv

            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
            writer.writerow({'Label': getLabelIndex(img_path), 'Name':img_path, 'Landmarks':landmarkArray.rstrip(landmarkArray[-1])})
    else:
        print("No hand result " + img_path)
        failedDetectionFile.write(img_path + "\n")
        failedDetectionCount+=1

    frameCount+=1
    print("Frame:" + str(frameCount))

csvfile.close()
print("Failied detections: " + str(failedDetectionCount))
    #cv2.putText(frame, "Frame: " + str(frameCount), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    #cv2.imshow("Frame", frame)
    #cv2.imwrite(frameOutputDirectory + "Frame" + str(frameCount) + ".png", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# # release the webcam and destroy all active windows
# cap.release()
# cv2.destroyAllWindows()

