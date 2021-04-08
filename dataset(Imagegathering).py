import pickle
import cv2

# In line below, the path for the saved images folder.
path = 'Dataset(images)'
# Initialize the camera usage
capture = cv2.VideoCapture(0)

# Load the classifier(model) that is pre-programmed to identify faces
face_detector = cv2.CascadeClassifier('projectAssets\haarcascade_frontalface_default.xml')

ID = pickle.load(open("id.dat", "rb"))
face_id = int("".join(map(str, ID)))
# Initialize the count variable so that it keep records of how many images taken
count = 0

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, width, height) in faces:
        # Save the captured image into the dataset folder
        if ret:
            # if video is rolling take images and name them as described below and save it.
            name = "Dataset(images)\img." + str(count) + '.' + str(face_id) + ".jpg"
            print(name + ' created successfully')
            # writing the extracted images
            cv2.imwrite(name, frame)
            count += 1
            cv2.imshow("name", frame)
    if cv2.waitKey(1) & 0xff == 27:  # press Esc to break the loop
        break
    elif count >= 50:  # Take 50 face sample and stop video
        break

capture.release()
cv2.destroyAllWindows()
