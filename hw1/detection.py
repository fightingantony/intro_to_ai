import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    First, it reads data from a (.txt) file in the detect folder which contains image names and the 
    number of faces in each image. For each image, it reads the image, converts it to grayscale, and 
    iterates through the detected faces. Using a trained weak classifier, it classifies each detected face 
    as either a face or a non-face. Bounding boxes are drawn around the detected faces, with green 
    indicating a classified face and red indicating a non-face. The results are displayed using matplotlib.
    """
    folderPath = 'C:\\Users\\choan\\Desktop\\intro_ai\\1\\HW1\\data\\detect'
    with open(dataPath, 'r') as file:
      lines = file.readlines()
      line_index = 0
      while line_index < len(lines):
        parts = lines[line_index].strip().split()
        if len(parts) != 2:
          line_index += 1
          continue
        image_name, num_faces = parts
        num_faces = int(num_faces)
        img_path = os.path.join(folderPath, image_name)
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        line_index += 1
        for _ in range(num_faces):
          parts = lines[line_index].strip().split()
          if len(parts) != 4:
            line_index += 1
            continue
          x, y, w, h = map(int, parts)
          face_img = gray_img[y:y+h, x:x+w]
          face_resized = cv2.resize(face_img, (19, 19))
          if clf.classify(face_resized):
            color = (0, 255, 0)
          else:
            color = (0, 0, 255)
          cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = 2)
          line_index += 1
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    #raise NotImplementedError("To be implemented")
    # End your code (Part 4)
