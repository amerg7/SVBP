import math
import shapely
from keras.models import load_model
import pickle
from shapely.geometry import Polygon
import cv2
import pyttsx3


# the function below are used for tracking the objects of any type using ROI (Region Of Interest)
classFile = 'projectAssets/Object.names'
with open(classFile, 'r') as items:
    classNames = items.read().rstrip('\n').split('\n')

# the variable below are used to open the encrypted names list of users.
names = pickle.load(open("names.dat", "rb"))

# Files path below are very important in order to start the program.
# ******************************************************************* #
# In line below, label list of the facial emotions.
class_labels = ["Angry", "Happy", "Normal", "Sad", "Surprised"]

# The variable below used to initiate the CV2 face recognition.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# In line below, the path for the trainer file depends on the CV2 LBHFaceRecognizer.
recognizer.read('projectAssets/trainer.yml')
cascadePath = "projectAssets/haarcascade_frontalface_default.xml"

# In line below, the path for objects weights and configuration files.
configPath = 'projectAssets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'projectAssets/frozen_inference_graph.pb'

# In line below, the path for objects names file.
classFile = 'projectAssets/Object.names'

# In line below, the path for Faces classifier.
face_classifier = cv2.CascadeClassifier('projectAssets\haarcascade_frontalface_default.xml')

# In line below, the path for Facial Emotion Model.
classifier = load_model('projectAssets\Emotion_Detection.h5')

# In line below, the path for Object detector.
detector = cv2.CascadeClassifier("projectAssets\haarcascade_frontalface_default.xml")

# the measurement of objects below are in cm (most common objects are used)
objectDec = {'person': 42, 'bicycle': 45, 'car': 157, 'motorcycle': 85, 'airplane': 4500,
             'bus': 255, 'train': 144, 'truck': 260, 'Fishing Boat': 130, 'traffic light': 27,
             'fire hydrant': 30, 'street sign': 68, 'stop sign': 68, 'parking meter': 20,
             'bench': 129, 'bird': 10, 'cat': 22, 'dog': 25, 'horse': 85, 'sheep': 60, 'cow': 140,
             'elephant': 350, 'bear': 122, 'zebra': 75, 'giraffe': 110, 'hat': 56, 'backpack': 35,
             'umbrella': 95, 'shoe': 12, 'eye glasses': 15, 'handbag': 25, 'tie': 8, 'suitcase': 45,
             'frisbee': 25, 'skis': 12, 'snowboard': 25, 'sports ball': 27, 'kite': 100, 'baseball bat': 7,
             'baseball glove': 25, 'skateboard': 22, 'surfboard': 60, 'tennis racket': 32, 'bottle': 7,
             'plate': 16, 'wine glass': 10, 'cup': 10, 'fork': 3, 'scissors': 9, 'spoon': 4, 'bowl': 18, 'banana': 4,
             'apple': 8, 'sandwich': 13, 'orange': 7, 'broccoli': 4, 'carrot': 4, 'hot dog': 6, 'pizza': 30,
             'donut': 12, 'cake': 30, 'chair': 70, 'couch': 200, 'potted plant': 17, 'bed': 150, 'mirror': 30,
             'dining table': 97, 'window': 150, 'desk': 150, 'toilet': 50, 'door ': 92, 'tv': 88, 'laptop': 38,
             'mouse': 9, 'remote': 5, 'keyboard': 46, 'cell phone': 8, 'microwave': 76, 'oven': 76, 'toaster': 30,
             'sink': 83, 'refrigerator': 83, 'blender': 30, 'book': 35, 'clock': 30, 'vase': 19, 'knife': 9,
             'teddy bear': 15, 'hair drier': 5, 'toothbrush': 1, 'hair brush': 7, 'Table': 100
             }

# the variable below are used to initialize the voice engine
voiceEngine = pyttsx3.init()


# the function below are used to calculate the actual distance of faces using face dimensions (bounding boxes)
def faceDes(x, y, width, height, actual_width):
    pixel_distance = math.sqrt((x + width) ** 2 + (y + height) ** 2)
    actual_distance = (pixel_distance / width) * actual_width
    distFace = "{:.0f}".format(actual_distance * 1.1)
    return distFace


# the function below are used to calculate the actual distance of objects using object dimensions (bounding boxes)
def objDis(x, y, width, height, id):
    pixel_distance = math.sqrt((x + width) ** 2 + (y + height) ** 2)
    actual_distance = (pixel_distance / width) * objectDec[classNames[id - 1]]
    distance = "{:.0f}".format(actual_distance * 1.1)
    return distance


# Function below are used to calculate the bounding box and the ground box.
def calculate_iou(box_1, box_2):
    polygon1 = shapely.geometry.box(*box_1, ccw=True)
    polygon2 = shapely.geometry.box(*box_2, ccw=True)
    polygon_1 = shapely.geometry.box(*box_1, ccw=True)
    polygon_2 = shapely.geometry.box(*box_2, ccw=True)
    iou = polygon1.intersection(polygon2).area / polygon1.union(polygon2).area
    if polygon1.union(polygon2).area == 0:
        iouu = polygon_1.intersection(polygon_2).area / polygon_1.intersection(polygon_2).area
        return iouu
    return iou


# Function below are used with objects to compare the ground box with estimated bounding box, it return false when the
# threshold are less than the threshold, if its grater, then it return true and set the bounding box as ground box
def isBoxMatched(trackerBbox, detectionbox):
    # the problem is different data types (tuple, float)
    threshold = 0.5
    for trackerbox in trackerBbox:
        compare_resluts = calculate_iou(trackerbox, detectionbox)
        if compare_resluts > threshold:
            return True
    return False


# Function below are used with faces to compare the ground box with estimated bounding box, it return false when the
# threshold are less than the threshold, if its grater, then it return true and set the bounding box as ground box
def isFBoxMatched(trackerBbox, detectionbox):
    threshold = 0.5
    for trackerbox in trackerBbox:
        fcompare_resluts = calculate_iou(trackerbox, detectionbox)
        if fcompare_resluts > threshold:
            return True
    return False


# Function below are used to convert the texts to voices
def voiceOutput(name):
    voiceEngine.setProperty("rate", 200)
    voiceEngine.say(name)
    voiceEngine.runAndWait()
