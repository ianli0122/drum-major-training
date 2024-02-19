import cv2 
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands.Hands()
mpDraw = mp.solutions.drawing_utils
wrist = [] # time, x, y, z

while True:
    success, img = cap.read()
    results = mpHands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                if (id == 0):
                    wrist.append([time.time(), lm.x, lm.y, lm.z])
            mpDraw.draw_landmarks(img, handLandmarks, mp.solutions.hands.HAND_CONNECTIONS)

    for i in wrist:
        if time.time() - i[0] > 5:
            wrist.remove(i)

    if wrist:
        if (abs(wrist[-1][2] - wrist[-2][2]) < 0.185):
            print(True)
        else:
            print(False)

    cv2.imshow("Video Feed", img)
    cv2.waitKey(1)