from random import random, randrange
import cv2
from scipy.misc import face

#get path
video_file_path =   '/Users/yash/playground/AI_clever_programer/2.car_human_detection/tesla_car.mp4'
car_tracker_file_path = '/Users/yash/playground/AI_clever_programer/2.car_human_detection/car_detection.xml'
fullbody_tracker_file_path='/Users/yash/playground/AI_clever_programer/2.car_human_detection/haarcascade_fullbody.xml'

#create car Classifier
# Load some pre-trained data on face frontals from open (haar cascade algorithm
car_tracker=cv2.CascadeClassifier(car_tracker_file_path)
#create people Classifier(we track full body of human) -- pedestrian : people
pedestrian_tracker = cv2.CascadeClassifier (fullbody_tracker_file_path)


# video procesiing 
#we detect car in video stream
# Choose an image to detect car in
# webcam=cv2.VideoCapture(0) #read from webcam #use webcam
webcam=cv2.VideoCapture(video_file_path) #use video also

# Iterate forever over frames
while True:
    # Read the current frame
    is_successful_frame_read, frame = webcam.read() #is_successful_frame_read is true or flase //frame contain frame
    
    if is_successful_frame_read:
         # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Detect car
    car_coordinates_list = car_tracker.detectMultiScale(grayscaled_frame)#list of car
    pedestrian_coordinates_list = pedestrian_tracker.detectMultiScale(grayscaled_frame) #list of human
    print(car_coordinates_list)
    for (x, y, w, h) in car_coordinates_list:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
    for (x, y, w, h) in pedestrian_coordinates_list:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)#yellow#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
     
    cv2.imshow('car detection', frame) 
    # .waitKey() #when we press a key at a time frame will change
    key= cv2.waitKey(1) #every one milisecond fframe will change
     ####Stop if Q key is pressed 
    if key==81 or key==113: #press q to exit
         break
     
#### Release the VideoCapture obiect 
webcam. release ()  

 
print("code complete")  


