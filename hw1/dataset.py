import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """
    # Begin your code (Part 1-1)
    """
    I initialize an empty dataset structured into two lists, one each for training and testing data. 
    Then I navigate through the dataset directory, categorizing images into subsets ('train' and 'test') 
    and labels ('face' and 'non-face'). Using OpenCV to read each image in grayscale mode. Next, I
    assign labels—faces as 1 and non-faces as 0 which enable the model to learn from the input 
    images. Finally, I append (image, label) pairs to the respective subsets to construct the dataset.
    """
    dataset = [], []
    data_path = 'C:\\Users\\choan\\Desktop\\intro_ai\\1\\HW1\\data\\data_small'
    subsets = ['train', 'test']
    for subset in subsets:
        labels = ['face', 'non-face']
        for label in labels:
            label_path = os.path.join(data_path, subset, label) #os.path.join拼接文件路徑
            label_val = 1 if label == 'face' else 0
            for root, dirs, files in os.walk(label_path): #os.walk可以遍歷目錄樹
                for file in files:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #imread可以把圖片轉為numpy array
                    dataset[subsets.index(subset)].append((img, label_val))
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1-1)
    return dataset


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_FDDB'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("C:\\Users\\choan\\Desktop\\intro_ai\\1\\HW1\\data\\data_FDDB\\FDDB-folds\\FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("C:\\Users\\choan\\Desktop\\intro_ai\\1\\HW1\\data\\data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        
        # Begin your code (Part 1-2)
        """
        I iterate through each detected face bounding box and generate non-face regions with these face 
        regions. I first calculate the width and height of each face region. Then randomly select a specified 
        number of non-face regions within each face region. For each non-face region, a random top-left 
        point is chosen within the image dimensions, and the bottom-right point is calculated based on the 
        width and height of the face region. Next, I check for intersection between the non-face region and 
        each face bounding box to avoid overlap with face regions. If no intersection is found, the non-face 
        region is considered valid and cropped from the grayscale image. This non-face region is then 
        resized to a fixed size of 19x19 pixels and added to the non-face dataset with label 0. Finally, I set a 
        break to check only one non-face region per face region.
        """
        for face_box in face_box_list:
            left_top_face, right_bottom_face = face_box
            face_width = right_bottom_face[0] - left_top_face[0]
            face_height = right_bottom_face[1] - left_top_face[1]
            # randomly select non-face region
            for _ in range(num_faces): 
                #randomly select top-left point for non-face region
                nonface_left_top_x = np.random.randint(0, img_gray.shape[1] - face_width)
                nonface_left_top_y = np.random.randint(0, img_gray.shape[0] - face_height)
                nonface_left_top = (nonface_left_top_x, nonface_left_top_y)
                #calculate bottom-right point for non-face region
                nonface_right_bottom_x = nonface_left_top_x + face_width
                nonface_right_bottom_y = nonface_left_top_y + face_height
                nonface_right_bottom = (nonface_right_bottom_x, nonface_right_bottom_y)
                #check for intersection with face region
                intersection = False
                for face_box in face_box_list:
                    left_top, right_bottom = face_box
                    if nonface_right_bottom[0] > left_top[0] and nonface_left_top[0] < right_bottom[0] \
                            and nonface_right_bottom[1] > left_top[1] and nonface_left_top[1] < right_bottom[1]:
                        intersection = True
                        break
                if not intersection:
                    nonface_img_crop = img_gray[nonface_left_top[1]:nonface_right_bottom[1],
                                     nonface_left_top[0]:nonface_right_bottom[0]].copy()
                    nonface_dataset.append((cv2.resize(nonface_img_crop, (19, 19)), 0))
                    break #break to only select one non-face region per face region
        #for i in range(num_faces):
        #raise NotImplementedError("To be implemented")
        # End your code (Part 1-2)
        #nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))
        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()
