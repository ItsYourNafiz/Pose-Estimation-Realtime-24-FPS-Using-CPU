import cv2
import time
import PoseModule

cap = cv2.VideoCapture("PoseVideos/Pose10.mp4")     # Instead of Video Source, Use 0 to Accesses Webcam
pTime = 0
detector = PoseModule.poseDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read the video or end of video.")
        break

    img, results = detector.findPose(img)
    lmList = detector.findPositions(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv2.FILLED)    # For Tracking Specific Point

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