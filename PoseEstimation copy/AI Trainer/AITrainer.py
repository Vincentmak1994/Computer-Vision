import cv2
import time
import numpy as np
import PoseEstimationModule as pem

cap = cv2.VideoCapture('sources/pushup_front.mp4')
# 'sources/pushup1.mp4'  'sources/situp.mp4' 'sources/pushup_front.mp4'
pTime = 0
detector = pem.PoseEstimation()
exercise = 'Push Up'
# 'Sit Up'
cameraAngles = ['Front', 'Left', 'Right']
cameraAngle = cameraAngles[0]

count = 0
dir = 0
score = []

findMinMax = []

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (1280, 720))
    # img = cv2.imread('sources/situp_img.jpg')
    # situp_img.jpg pushup_img.jpg
    img = detector.findPose(img, False)

    lmList = detector.findPosePosition(img, False)

    if len(lmList) != 0:
        if exercise == 'Push Up':
            hipWarning = 0

            if cameraAngle == cameraAngles[1]: #Left
                # left elbow Max= 178.18798453964342 Min = 74.92341197069042
                angle = detector.findAngle(img, 11, 13, 15)
                per = np.interp(angle, (75, 177), (0,100)) #convert to 0-100 scale
                # bar = np.interp(angle, (75, 177), (0,100))

                #Warning if not low enough
                # if angle < 100 and angle > 80:
                #     cv2.putText(img, 'Lower!', (150, 600), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 10)


                #Body
                body_angle = detector.findAngle(img, 11, 23, 27)
                if body_angle < 160:
                    cv2.putText(img, 'Pay Attention: Hip!', (150, 600), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 10)
                    hipWarning += 1


            elif cameraAngle == cameraAngles[2]: #Right
                angle = detector.findAngle(img, 12, 14, 16)
                per = np.interp(angle, (75, 177), (0, 100))  # convert to 0-100 scale

                # Warning if not low enough
                # if angle < 100 and angle > 80:
                #     cv2.putText(img, 'Lower!', (150, 600), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 10)

                body_angle = detector.findAngle(img, 11, 23, 27)
                if body_angle < 160:
                    cv2.putText(img, 'Pay Attention: Hip!', (150, 600), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 10)
                    hipWarning += 1


            elif cameraAngle == cameraAngles[0]: #Front
                # left elbow Max= 178.18798453964342 Min = 74.92341197069042
                left_angle = detector.findAngle(img, 11, 13, 15)
                left_per = np.interp(left_angle, (75, 177), (0, 100))  # convert to 0-100 scale

                right_angle = detector.findAngle(img, 12, 14, 16)
                right_per = np.interp(right_angle, (75, 177), (0, 100))  # convert to 0-100 scale

                x1, y1 = lmList[12][1], lmList[12][2]
                x2, y2 = lmList[11][1], lmList[11][2]

                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)



            # Check for curls
            if (cameraAngle == cameraAngles[1] or cameraAngle == cameraAngles[2]):
                if per < 5:  # push up down - angle ~90'
                    if dir == 0:
                        count += .5
                        dir = 1

                if per > 90:  # going up - angle ~180'
                    if dir == 1:
                        count += .5
                        dir = 0

            elif cameraAngle == cameraAngles[0]:
                if left_per < 5 and right_per < 5:  # push up down - angle ~90'
                    if dir == 0:
                        count += .5
                        dir = 1

                if left_per > 90 and right_per > 90:  # going up - angle ~180'
                    if dir == 1:
                        count += .5
                        dir = 0


        #
        #
        # elif exercise == 'Sit Up':
        #     # Right Side max: 145.68028091644504 ,min:46.09246445400166
        #     angle = detector.findAngle(img, 12,24,26)
        #     per = np.interp(angle, (46, 145), (0, 100))
        #     # findMinMax.append(angle)
        #
        #     # Check for curls
        #     if per < 5:  # push up down - angle ~50'
        #         if dir == 0:
        #             count += .5
        #             dir = 1
        #
        #     if per > 90:  # going up - angle ~130'
        #         if dir == 1:
        #             count += .5
        #             dir = 0

        # print(min(findMinMax))
        # print(count)
        cv2.rectangle(img, (0,0), (200, 250), (0,255,0), cv2.FILLED)
                      # (0,450), (250,750), (0,255,0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (10,200), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 10)




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.imshow('Image',img)
    cv2.waitKey(1)

