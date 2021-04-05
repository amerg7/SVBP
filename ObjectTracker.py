import cv2
import Functions
import argparse
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    }
# font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
# TRACKER INITIALIZATION
success, frame = cap.read()
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="csrt",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())


trackers = cv2.MultiTracker_create()
bbox =cv2.selectROI("frame",frame)


def drawBox(frame,bbox):
    x, y, w, h = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(frame, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while True:
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    print(bbox)
    timer = cv2.getTickCount()
    ret, frame = cap.read()

    trackers.add(tracker, frame, bbox)
    success, bbox = trackers.update(frame)

    if success:
        drawBox(frame,bbox)
    else:
        cv2.putText(frame, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(frame,(15,15),(200,90),(255,0,255),2)
    cv2.putText(frame, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(frame, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if fps>60: myColor = (20,230,20)
    elif fps>20: myColor = (230,20,20)
    else: myColor = (20,20,230)
    cv2.putText(frame,str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xff ==27:
       break