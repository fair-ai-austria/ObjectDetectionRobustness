import requests
import json
from time import sleep
from pathlib import Path
import numpy as np

class CustomObjectDetectionService:
    def __init__(self):
        self.image_file_name = ""
        pass

    def update_image(self, image_file_name):
        self.image_file_name = image_file_name
        #do not care, just to have compatible interface with Jrs Object Detection service
        pass

    def detect_all_objects(self):

        #TODO: Read self.image_file_name and detect all objects in it...

        #A detected object looks like this:
        det_obj_1={}
        det_obj_1["region"]=[1,10,2,20] #fromX, fromY, toX, toY
        det_obj_1["class"]="ObjectLabel1"
        det_obj_1["confidence"]=0.342

        #And another one...
        det_obj_2={}
        det_obj_2["region"]=[5,11,23,65]
        det_obj_2["class"]="ObjectLabel2"
        det_obj_2["confidence"]=0.15247

        return [det_obj_1, det_obj_2]

    def get_score_image(self, image_file_name, wanted_label_id):
        
        #TODO: Read image_file_name, detect object type 'wanted_label_id' in it and return a 2D numpy array with each pixel's value assigned the probability/confidence of belonging to the object

        score_image = np.random.random((50,50))
        return score_image
