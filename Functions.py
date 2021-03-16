import math
import os

import cv2
import keyboard as keyboard
from keras.models import load_model
import pickle
import multiprocessing
import time
import pyttsx3
from gtts import gTTS
import playsound


voiceEngin = pyttsx3.init()

# the measurements below are in cm
objectDec = {'person': 42, 'bicycle': 45, 'car': 157, 'motorcycle': 85, 'airplane': 4500, 'bus': 255, 'train': 144,
             'truck': 260, 'Fishing Boat': 130, 'traffic light': 27, 'fire hydrant': 30, 'street sign': 68,
             'stop sign': 68, 'parking meter': 20, 'bench': 129, 'bird': 10, 'cat': 22, 'dog': 25, 'horse': 85,
             'sheep': 60, 'cow': 140, 'elephant': 350, 'bear': 122, 'zebra': 75, 'giraffe': 110, 'hat': 56,
             'backpack': 35, 'umbrella': 95, 'shoe': 12, 'eye glasses': 15, 'handbag': 25, 'tie': 8, 'suitcase': 45,
             'frisbee': 25, 'skis': 12, 'snowboard': 25, 'sports ball': 27, 'kite': 100, 'baseball bat': 7,
             'baseball glove': 25, 'skateboard': 22, 'surfboard': 60, 'tennis racket': 32, 'bottle': 7, 'plate': 16,
             'wine glass': 10, 'cup': 10, 'fork': 3, 'knife': 6, 'spoon': 4, 'bowl': 18, 'banana': 4, 'apple': 8,
             'sandwich': 13, 'orange': 7, 'broccoli': 4, 'carrot': 4, 'hot dog': 6, 'pizza': 30, 'donut': 12,
             'cake': 30, 'chair': 70, 'couch': 200, 'potted plant': 17, 'bed': 150, 'mirror': 30, 'dining table': 97,
             'window': 150, 'desk': 150, 'toilet': 50, 'door ': 92, 'tv': 88, 'laptop': 38, 'mouse': 9, 'remote': 5,
             'keyboard': 46, 'cell phone': 8, 'microwave': 76, 'oven': 76, 'toaster': 30, 'sink': 83,
             'refrigerator': 83, 'blender': 30, 'book': 35, 'clock': 30, 'vase': 19, 'scissors': 17, 'teddy bear': 15,
             'hair drier': 5, 'toothbrush': 1, 'hair brush': 7, 'Table': 100}


# the function below calculate the actual distance of human faces using face dimensions
def faceDes(x, y, width, height, actual_width):
    pixel_distance = math.sqrt(((x + width)) ** 2 + ((y + height)) ** 2)
    actual_distance = (pixel_distance / width) * actual_width
    distFace = '%.2f' % (actual_distance)
    return distFace


classFile = 'projectAssets/Object.names'
with open(classFile, 'r') as items:
    classNames = items.read().rstrip('\n').split('\n')


# the function below calculate the actual distance of objects using object dimensions
def obDes(x, y, width, height, id):
    pixel_distance = math.sqrt(((x + width)) ** 2 + ((y + height)) ** 2)
    actual_distance = (pixel_distance / width) * objectDec[classNames[id - 1]]
    dist = '%.2f' % (actual_distance * 1.5)
    return dist


names = pickle.load(open("names.dat", "rb"))

# the files below are very important in order to start the program
class_labels = ["Angry", "Happy", "Normal", "Sad", "Surprise"]
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('projectAssets/trainer.yml')
cascadePath = "projectAssets/haarcascade_frontalface_default.xml"
configPath = 'projectAssets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'projectAssets/frozen_inference_graph.pb'
classFile = 'projectAssets/Object.names'
face_classifier = cv2.CascadeClassifier('projectAssets\haarcascade_frontalface_default.xml')
classifier = load_model('projectAssets\Emotion_Detection.h5')

#
# def voicesid(id):
#     if __name__ == '__main__':
#         p = multiprocessing.Process(target=voices, name="voice", args=(10,))
#         p.start()
#         # Wait 10 seconds for foo
#         time.sleep(5)
#         # Terminate foo
#         p.terminate()
#         # Cleanup
#         p.join()
#
#

def voices(id):
    for i in range(1):
        if id != 0:
            sayit = '"{}"'.format(id)
            voiceEngin.say(sayit)
            print(sayit)
            voiceEngin.runAndWait()

def objectvoic(id):
    for i in range(1):
        if id != 0:
            sayit = '"{}"'.format(id)
            voiceEngin.say(sayit)
            print(sayit)
            voiceEngin.runAndWait()


# def names():
#     xname = id
#     print(xname)
#
# def speak(text):
#     i = 0
#     while i < 5:
#
#         tts = gTTS(text=text, lang="en")
#         namevoice = str("voice" + str(i) + ".mp3")
#         tts.save(namevoice)
#         playsound.playsound(namevoice, True)
#         os.remove(namevoice)
#
#
#
# def voicespeak(id):
#     p = multiprocessing.Process(target=speak, args=(10,))
#     p.start()
#     p.join(5)
#     if p.is_alive():
#         p.terminate()
#
#
