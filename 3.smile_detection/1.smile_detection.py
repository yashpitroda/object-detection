import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('/Users/yash/playground/AI_clever_programer/3.smile_detection/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('/Users/yash/playground/AI_clever_programer/3.smile_detection/haarcascade_smile.xml')

# Grab Webcam feed
webcam = cv2. VideoCapture(0)

# Show the current frame
while True:
    is_successful_frame_read, frame = webcam.read()
    if not is_successful_frame_read:
        break
    
    # Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = face_detector.detectMultiScale(grayscaled_frame)
    smile_coordinates = smile_detector.detectMultiScale(grayscaled_frame,scaleFactor=1.7,minNeighbors=20)#save this paramiter for smile
    #scaleFactor=1.7,minNeighbors=20 -- both are use for detecting smile 
    #scaleFactor -- how much do u blur the image
    print(face_coordinates)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5) #(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
        #we put rectengle on frame(colorfull ) and not on grayscaled_frame
    for (x, y, w, h) in smile_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) #(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
   
    #show current freme
    cv2.imshow ('Smile Detector', frame)
    
    key= cv2.waitKey(1) #every one milisecond fframe will change
     ####Stop if Q key is pressed 
    if key==81 or key==113: #press q to exit
         break
    
#### Release the VideoCapture obiect 
webcam. release ()   
print("code complete") 