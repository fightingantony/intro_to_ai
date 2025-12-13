import cv2
# use cascadeclassifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#read the image
img = cv2.imread('C:\\Users\\choan\\Desktop\\intro_ai\\1\\HW1\\data\\detect\\p110912sh-0083.jpg')
#turn the picture into gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#face detect
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#open file to save face coordinates
with open('C:\\Users\\choan\\Desktop\\intro_ai\\1\\HW1\\data\\detect\\testing.txt', 'w') as file:
    #write the picture name and the number of detected faces
    file.write(f'{len(faces)}\n')
    for (x, y, w, h) in faces:
        #write the coordinates of each face
        file.write(f'{x} {y} {w} {h}\n')
