import cv2
import numpy as np
from PIL import Image
import os

os.system("python dataset(Imagegathering).py")

# In line below, the path for the saved images folder.
path = 'Dataset(images)'
recognizer = cv2.face.LBPHFaceRecognizer_create()

# In line below, the path for Object Classifier.
detector = cv2.CascadeClassifier("projectAssets\haarcascade_frontalface_default.xml")

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert all images to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[2]) # From the path, get the ID of position 2 ex: img.imgNum.idNum
        print ("Trainging the face with id: " + str(id))
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print("\n Training faces is established. It will take a few seconds. Please wait")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the trained model in projectAssets/trainer.yml
# In line below, you need to specify the location of the file
recognizer.write('projectAssets\\trainer.yml')
# print how many faces has been trained and end program
print("\n{0} faces trained. Exiting Program".format(len(np.unique(ids))))
os.system('CLS')