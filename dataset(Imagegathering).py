import pickle
import cv2

path = 'Dataset(images)'
# Initialize the camera usage
capture = cv2.VideoCapture(0)

# Load the classifier(model) that is pre-programmed to identify faces
# In line below, you need to specify the location of the file
face_detector = cv2.CascadeClassifier('projectAssets\haarcascade_frontalface_default.xml')

ID = pickle.load(open("id.dat", "rb"))
face_id = int("".join(map(str, ID)))
# for i in range(0, len(ID)):
#     ID[i] = int(ID[-1])
#     face_id = str(ID)
print(face_id)

# Initialize the count variable so that it keep records of how many images taken
count = 0

while(True):
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # Save the captured image into the dataset folder
        if ret:
            # if video is rolling take images and save them
            # In line below, you need to specify the location of the file
            name = "Dataset(images)\img." + str(count)+'.'+str(face_id)+".jpg"
            print(name+' created successfully')
            # writing the extracted images
            cv2.imwrite(name, frame)
            count +=1
            cv2.imshow("name",frame)
    end = cv2.waitKey(1) & 0xff  # press Esc to break the loop
    if end == 27:
        break
    elif count >= 50:# Take 50 face sample and stop video
        break

capture.release()
cv2.destroyAllWindows()
