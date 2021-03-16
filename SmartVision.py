import Functions
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import multiprocessing
import time
import pyttsx3
import keyboard as keyboard
import win32com.client as wincl


faceCascade = cv2.CascadeClassifier(Functions.cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
fconfidence = Functions.recognizer.predict
voiceEngin = pyttsx3.init()
speak = wincl.Dispatch("SAPI.SpVoice")

id = 0
capture = cv2.VideoCapture(0)
classNames = []
net = cv2.dnn_DetectionModel(Functions.weightsPath, Functions.configPath)

# this is used to normalize the frame
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
minW = 0.1 * capture.get(3)
minH = 0.1 * capture.get(4)
said = False
while True:
            success, show = capture.read()
            classIds, confs, bbox = net.detect(show, confThreshold=0.5)
            labels = []
            Togray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            faces = Functions.face_classifier.detectMultiScale(Togray, 1.3, 5)
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    dist = Functions.obDes(box[0], box[1], box[2], box[3], classId)
                    if dist > str(200):
                        if classId != 1:
                            cv2.rectangle(show, box, color=(255, 255, 255), thickness=4)
                            cv2.putText(show, Functions.classNames[classId-1], (box[0], box[1]), font, 1, (0, 0, 0), 2)
                            cv2.putText(show, str(round(confidence * 100, 1)), (box[0]+10, box[1]+40), font, 1, (0, 0, 2), 2)
                            cv2.putText(show, str(dist), (box[0]+10, box[1]+20), font, 1, (0, 0, 0), 2)
                            if (confidence * 100) > 80 and not said:
                                Functions.voices(classNames[classId-1])
                                said = True
                                if (confidence * 100) < 80 and not said:
                                    Functions.voices(classNames[classId - 1])
                                    said = False
                        else:
                            gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
                            faces = faceCascade.detectMultiScale(
                                gray,
                                scaleFactor=1.2,
                                minNeighbors=5,
                                minSize=(int(minW), int(minH))
                            )

                            for (x, y, width, height) in faces:
                                cv2.rectangle(show, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                id, fconfidence = Functions.recognizer.predict(gray[y:y + height, x:x + width])
                                faceDist = Functions.faceDes(x, y, width, height, actual_width=23)

                                if fconfidence <= 50 and not said:
                                    id = Functions.names[id]
                                    fconfidence = "{0}%".format(round(100 - fconfidence))
                                    Functions.voices(id)
                                    said = True
                                if fconfidence > 50 and not said:
                                    Functions.voices(id)
                                    said = False

                                    for (top, right, bottom, left) in faces:
                                        # cv2.rectangle(show, (top, right), (top + bottom, right + left),
                                        #               (255, 0, 0), 2)
                                        roi_gray = Togray[right:right + left, top:top + bottom]
                                        roi_gray = cv2.resize(roi_gray, (48, 48))
                                        if np.sum([roi_gray]) != 0:  # the sum of the matrix must not be 0
                                            roi = roi_gray.astype('float') / 255.0
                                            roi = img_to_array(roi)
                                            roi = np.expand_dims(roi, axis=0)

                                            # make a prediction on the face and print the prediction stat
                                            preds = Functions.classifier.predict(roi)[0]
                                            label = Functions.class_labels[preds.argmax()]
                                            cv2.putText(show, "Status:" + label, (0, 120), font, 1, (0, 255, 0), 2)

                                else:
                                    id = "unknown person"
                                    fconfidence = "0%".format(round(100 - fconfidence))
                                    if success and not said:
                                        Functions.voices(id)
                                        said = True

                                cv2.putText(show, "Name: " + str(id), (0, 30), font, 1, (0, 0, 0), 2)
                                cv2.putText(show, "sure by: " + str(fconfidence), (0, 60), font, 1, (0, 0, 0), 2)
                                cv2.putText(show, "Distance: " + str(faceDist)+" cm", (0, 90), font, 1, (0, 0, 0), 2)


            cv2.imshow("Webcam", show)
            if cv2.waitKey(1) & 0xFF == 27:
                break
capture.release()
cv2.destroyAllWindows()