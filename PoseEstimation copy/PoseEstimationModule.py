import cv2
import mediapipe as mp
import time
import math

class PoseEstimation():

    def __init__ (self, static_image_mode=True,model_complexity=2, min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
        self.color = (255,0,0) #blue
        self.color2 = (0,0,255) #red

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)

        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosePosition(self, img, draw=True):
        self.lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                # print(id,lm)

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, self.color, cv2.FILLED)
        return self.lmList

    def findAngle(self,img, p1, p2, p3, draw=True):

        #Get the Landmarks
        x1, y1 = self.lmList[p1][1:] #take x and y
        x2, y2 = self.lmList[p2][1:]  # take x and y
        x3, y3 = self.lmList[p3][1:]  # take x and y

        #Calculate Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        # print(angle)

        if angle > 180:
            angle = 360 - angle
        if angle < 0:
            angle = abs(angle)

        #Draw
        if draw:
            cv2.line(img, (x1,y1),(x2,y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            cv2.circle(img, (x1, y1), 10, self.color, cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, self.color, 2)

            cv2.circle(img, (x2, y2), 10, self.color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, self.color, 2)

            cv2.circle(img, (x3, y3), 10, self.color, cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, self.color, 2)

            cv2.putText(img, str(int(angle)), (x2-20, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, self.color2, 2)

        return angle



def main():
    ######################################
    # wCam, hCam = 640, 480
    ######################################
    cap = cv2.VideoCapture(0)
    # 'videos/workout1.mp4'
    # cap.set(3, wCam)
    # cap.set(4, hCam)

    pTime = 0
    detector = PoseEstimation()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosePosition(img, draw=False)

        #Only draw the noise
        # if len(lmList) != 0:
            # cv2.circle(img, (lmList[0][1],lmList[0][2]), 10, (255,0,255), cv2.FILLED)
            # print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()