import sys
import os


sys.path.append("..")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/models/research/')
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/models/research/object_detection')
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/models/research/slim')
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/pynaoqi/lib')

import argparse
import time

import numpy as np
import qi
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
#import naoqi

from collections import defaultdict
from io import StringIO
import matplotlib
import vision_definitions

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import cv2

cap = cv2.VideoCapture(0)  # primaeres Videowidergaberaet = 0 also beim Laptop ist es die Webcam

# ## Object Erkennung module werden importiert

from utils import label_map_util

from utils import visualization_utils as vis_util

# # Modell

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

#Pfad zum Trainierten Modell fuer Object Detection 

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('models', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Laedt das Tensorflow Modell.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# label map wird geladen

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Erkennung

# Groesse des Bilder Outputs in Zoll(inches).
IMAGE_SIZE = (12, 8)

def naoImageToNpImage(naoImage):
    # naoimage 0 = width 1 = height 6 = bytes
    return True, Image.frombytes("RGB", (naoImage[0], naoImage[1]), bytes(naoImage[6]))

#def main(session):
def main():
    video_service = session.service("ALVideoDevice")
    tts = session.service("ALTextToSpeech")
    fps = 20#Frames pro Sekunde
    nameId = video_service.subscribe("python_GVM", vision_definitions.kVGA, vision_definitions.kYUVColorSpace, fps)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, image_np = naoImageToNpImage(video_service.getImageRemote(nameId))
                image_np_expanded = np.expand_dims(image_np, 0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Jedes Rechteck bedeutet dass das Object identifiziert wurde
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # score bedeutet die wahrscheinlichkeit dass es sich um das object handelt
                # und wird zsm mit mit dem label angezeigt
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Hier findet die eigentliche Erkennung statt
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualisierung der Ergebnisse .

                if len(classes) < 1:
                    continue
                candidate = 0
                max = 0
                personMax = 0
                for i in range(0, len(classes)):
                    if (classes[0][i] == 1): # person found
                        if (personMax < scores[0][i]):
                            personMax = scores[0][i]
                    if max < scores[0][i]:
                        max = scores[0][i]
                        candidate = i
                id = str(classes[0][candidate]).strip(".0")
                with open(os.path.dirname(os.path.realpath(__file__)) + "/models/research/object_detection/data/mscoco_label_map.pbtxt") as f:
                    content = f.readline()
                    while True:
                        while " " + id + "\n" not in content:
                            content = f.readline()

                        content = f.readline()
                        label = content.split("\"")[1]
                        print(str(max) + " " + id + " " + label)
                        break

                    f.close()
                if (personMax < 0.20 and id == 67): # if person recognized over under 20% and main "dinning_table" found stored in ID
                    tts.say("Freier Arbeitsplatzt")
                elif id == 67: # person found more than 20% so its possible that a person is there on a table so not free
                    tts.say("Arbeitsplatzt nicht frei")
                time.sleep(5)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                cv2.imshow('object detection', np.asanyarray(image_np))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    video_service.unsubscribe(nameId)


if __name__ == "__main__": # main Funktion hier wird die Verbidnung mit pepper hergestellt
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=53520,
                        help="Naoqi port number")
    args = parser.parse_args()
    print(args)
    session = qi.Session()
    try:
        print("try to connect")
        session.connect("tcp://" + args.ip + ":" + str(args.port))
        print("connected")
        main(session)
    except RuntimeError:
        print(RuntimeError.message)
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
    #main()
