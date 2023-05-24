from random import random, randrange
import cv2
from scipy.misc import face

#get path
# video_file_path =   '/Users/yash/playground/AI_clever_programer/2.car_human_detection/bike_ride.mp4'
video_file_path =   '/Users/yash/playground/AI_clever_programer/2.car_human_detection/tesla_car.mp4'
car_tracker_file_path = '/Users/yash/playground/AI_clever_programer/2.car_human_detection/car_detection.xml'

#create car Classifier
# Load some pre-trained data on face frontals from open (haar cascade algorithm
car_tracker=cv2.CascadeClassifier(car_tracker_file_path)

# video procesiing --we detect car in video stream
# Choose an image to detect car in
# webcam=cv2.VideoCapture(0) #read from webcam #use webcam
webcam=cv2.VideoCapture(video_file_path) #use video also

# Iterate forever over frames
while True:
    #### Read the current frame
    is_successful_frame_read, frame = webcam.read() #is_successful_frame_read is true or flase //frame contain frame
    
    if is_successful_frame_read:
         # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Detect car
    car_coordinates = car_tracker.detectMultiScale(grayscaled_frame)
    print(car_coordinates)
    for (x, y, w, h) in car_coordinates:
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
    #we put rectengle on frame(colorfull ) and not on grayscaled_frame
    
    cv2.imshow('car detection', frame) 
    # cv2.waitKey() #when we press a key at a time frame will change
    key= cv2.waitKey(1) #every one milisecond fframe will change
     ####Stop if Q key is pressed 
    if key==81 or key==113: #press q to exit
         break
     
#### Release the VideoCapture obiect 
webcam. release ()  

 
print("code complete")  


