import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, modelComplexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                # Define the pink color for landmarks
                landmark_style = self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)  # Pink
                # Define the green color for connections
                connection_style = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green
                # Draw landmarks and connections with the specified colors
                self.mpDraw.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=landmark_style,
                                           connection_drawing_spec=connection_style)
        return img, self.results

    def findPositions(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture("PoseVideos/Pose6.mp4")  # Instead of Video Source, Use 0 to Accesses Webcam
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read the video or end of video.")
            break

        img, results = detector.findPose(img)
        lmList = detector.findPositions(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv2.FILLED)  # For Tracking Specific Point

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img, f'FPS: {int(fps)}', (60, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.imshow("Pose Estimation Realtime", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
