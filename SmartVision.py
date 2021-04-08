import Functions
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2

# The list below are used for saving the objects tracker boxes
trackerBboxes = []
# The list below are used for saving the faces tracker boxes
trackerFboxes = []

faceCascade = cv2.CascadeClassifier(Functions.cascadePath)
fconfidence = Functions.recognizer.predict
# initializing the camera and set the settings to minimum height and width so that it can be suitable for most cameras
capture = cv2.VideoCapture(0)
minW = 0.1 * capture.get(3)
minH = 0.1 * capture.get(4)
capture.set(3, 1800)
capture.set(4, 1200)
# Here we are passing path weights and configure wights
net = cv2.dnn_DetectionModel(Functions.weightsPath, Functions.configPath)
# this is used to normalize the frame
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while capture.isOpened():
    labels = []
    classNames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.5
    success, frame = capture.read()

    classIds, confs, detectionBboxes = net.detect(frame, confThreshold=0.5)
    Togray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Functions.face_classifier.detectMultiScale(Togray, 1.3, 5)
    # codes below are used to track and detect objects
    # if the object is not human then detect it and track within the frame
    if len(classIds) != 0:
        for classId, confidence, detectionBox in zip(classIds.flatten(), confs.flatten(), detectionBboxes):
            x, y, width, height = int(detectionBox[0]), int(detectionBox[1]), int(detectionBox[2]), int(detectionBox[3])
            # Here if detection didn't detect human then say the name of the object, the confidence percentage
            # and how far is it from the camera holder
            if (x, y, width, height != 0):
                objectDistance = Functions.objDis(x, y, width, height, classId)
                if classId != 1:
                    cv2.rectangle(frame, detectionBox, color=(255, 255, 255), thickness=2)
                    confidence = "{:.0f}".format(confidence * 100, 1)
                    # *************************************************
                    # Code below to start Warning voice based on distance
                    if int(objectDistance) <= 80:
                        warning = "warning " + Functions.classNames[classId - 1] + \
                                  " change your path"
                        # Functions.voiceOutput(warning)
                    # *************************************************
                    if not Functions.isBoxMatched(trackerBboxes, detectionBox):
                        if float(confidence) >= 60 and float(objectDistance) <= 120:
                            trackerBboxes.append(detectionBox)
                            Functions.voiceOutput(Functions.classNames[classId - 1])
                            if len(trackerBboxes) >= 2:
                                trackerBboxes.clear()
                    cv2.putText(frame, "Object name: " + Functions.classNames[classId - 1], (x, y - 45), font, 0.7,
                                (255, 255, 255), 2)
                    cv2.putText(frame, "Confidence: " + str(confidence), (x, y - 25), font, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Distance: " + str(objectDistance) + " cm", (x, y - 5), font, 0.7,
                                (255, 255, 255),
                                2)
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(
                        image=gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(int(minW), int(minH))
                    )
                    # If the detection detects a human then calculate the distance, get the name of the person
                    # and show confidence percentage
                    for (x, y, width, height) in faces:
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
                        id, faceConv = Functions.recognizer.predict(gray[y:y + height, x:x + width])
                        faceDist = Functions.faceDes(x, y, width, height, actual_width=23)
                        ROI_gray = Togray[y:y + height, x:x + width]
                        ROI_gray = cv2.resize(ROI_gray, (48, 48))
                        if np.sum([ROI_gray]) != 0:  # the sum of the matrix must not be 0
                            ROI = ROI_gray.astype('float') / 255.0
                            ROI = img_to_array(ROI)
                            ROI = np.expand_dims(ROI, axis=0)
                            # make a prediction on the face and print the prediction stat
                            preds = Functions.classifier.predict(ROI)[0]
                            label = Functions.class_labels[preds.argmax()]

                            if classId == 1 and faceConv <= 60:
                                id = Functions.names[id]
                                unknownFace = "{0}%".format(round(100 - faceConv))
                                if not Functions.isFBoxMatched(trackerFboxes, detectionBox) and int(faceDist) < 200:
                                    trackerFboxes.append(detectionBox)
                                    Functions.voiceOutput(str(id + " is here with a " + str(label) + " face"))
                                    if len(trackerFboxes) >= 4:
                                        trackerFboxes.clear()
                            else:
                                id = "unknown"
                                unknownFace = "0 %"
                                if not Functions.isFBoxMatched(trackerFboxes, detectionBox):
                                    trackerFboxes.append(detectionBox)
                                    Functions.voiceOutput(str(id + " person with a " + str(label) + " face"))

                            cv2.putText(frame, "Name: " + str(id), (x, y - 65), font, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, "Confidence: " + str(unknownFace), (x, y - 45), font, 0.7,
                                        (255, 255, 255), 2)
                            cv2.putText(frame, "Distance: " + str(faceDist) + " cm", (x, y - 25), font, 0.7,
                                        (255, 255, 255),
                                        2)
                            cv2.putText(frame, "Status indicator: " + label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 255),
                                        2)
    cv2.imshow("Smart Vision", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
capture.release()
cv2.destroyAllWindows()
