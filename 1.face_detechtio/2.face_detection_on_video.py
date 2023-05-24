from random import random, randrange
import cv2
from scipy.misc import face
# from random2 import randomrange


# Load some pre-trained data on face frontals from open (haar cascade algorithm
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video procesiing --we detect face in video stream
# Choose an image to detect faces in
webcam=cv2.VideoCapture(0) #read from webcam #use webcam
# webcam=cv2.VideoCapture('yash.mp4') #use video also

# Iterate forever over frames
while True:
    #### Read the current frame
    is_successful_frame_read, frame = webcam.read() #is_successful_frame_read is true or flase //frame contain frame
    
     # Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
    for (x, y, w, h) in face_coordinates:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
        #we put rectengle on frame(colorfull ) and not on grayscaled_frame
    
    cv2.imshow('yash', frame) 
    # cv2.waitKey() #when we press a key at a time frame will change
    key= cv2.waitKey(1) #every one milisecond fframe will change
     ####Stop if Q key is pressed 
    if key==81 or key==113: #press q to exit
         break
     
#### Release the VideoCapture obiect 
webcam. release ()   
print("code complete")  



# # Detect faces -- it retruns cordinates of rectengle suranding the face 
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)
# #[556 177 287 287] -- (556,177) x, y point 
# # x y width hight  

# # Draw rectangles around the faces
# # [x,y,w,h]=face_coordinates[0]
# for (x, y, w, h) in face_coordinates:
#     # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle

# #show img
# cv2.imshow("yash",img)
 
# cv2.waitKey()

# video procesiing

