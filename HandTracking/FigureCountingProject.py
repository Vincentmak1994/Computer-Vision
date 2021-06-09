import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam) #3= width
cap.set(4, hCam) #4 = Heigh

#store images

folderPath = 'FingerImages'
myList = sorted(os.listdir(folderPath))
# print(myList)
overlayList = []

for imPath in myList:
    if imPath != '.DS_Store':
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

# print(len(overlayList))

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]

#Display
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # we use can many if statement to identify different hand single like below!
        # if lmList[8][2] < lmList[6][2]: #openCV starting point at 0 (at the top of the image) so bigger value means lower position
        #     print('Index Finger Open')

        #since this project only consist 6 signs, we can use a for loop to achieve it

        #thumb -> if close it will be inside
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]: #make sure left/right hand
            fingers.append(1)
        else:
            fingers.append(0)

        #four fingers
        for id in range(1,5): #since the thumb does not act the same way as other fingers

            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)

        totalFingers = fingers.count(1) #count how many 1
        print(totalFingers)


        h, w, c = overlayList[totalFingers].shape

        img[0:h, 0:w] = overlayList[totalFingers]  # starting position x:0 y:0

        cv2.rectangle(img, (20,225), (170,425), (0,255,0), cv2.FILLED) #make a box
        cv2.putText(img, str(totalFingers), (40,0), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)



    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)


    cv2.imshow('Image',img)
    cv2.waitKey(1)