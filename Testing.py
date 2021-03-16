import cv2
import pyttsx3
cap = cv2.VideoCapture(0)
voiceEngine = pyttsx3.init()
while(True):
    # Capture frame-by-frame
    success, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if success:
        voiceEngine.say("hello there")
        voiceEngine.runAndWait()


        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()