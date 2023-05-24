#opencv doc
#https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html?highlight=detectmultiscale
# dataset
#https://www.researchgate.net/figure/Car-dataset-taken-by-Brad-Philip-and-Paul-Updike-California-Institute-of-Technology-It_fig5_267863282

from random import random, randrange
import cv2
from scipy.misc import face
from random import randomrange
# supervise - we(human) train model
# un-supervise - computer train itself
#this algoritham works on positive image(face) and nagative image(non face)

# Load some pre-trained data on face frontals from open (haar cascade algorithm
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image procesiing --we detect face in single image
# Choose an image to detect faces in
img = cv2.imread('test_img2.jpg') #read img

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces -- it retruns cordinates of rectengle suranding the face 
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)
#[556 177 287 287] -- (556,177) x, y point 
# x y width hight  

# Draw rectangles around the faces
# [x,y,w,h]=face_coordinates[0]
for (x, y, w, h) in face_coordinates:
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle

#show img
cv2.imshow("yash",img)
 
cv2.waitKey()

# video procesiing

print("code complete")  