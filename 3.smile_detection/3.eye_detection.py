import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('/Users/yash/playground/AI_clever_programer/3.smile_detection/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('/Users/yash/playground/AI_clever_programer/3.smile_detection/haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('/Users/yash/playground/AI_clever_programer/3.smile_detection/haarcascade_eye.xml')

# Grab Webcam feed
webcam = cv2. VideoCapture(0)

# Show the current frame
while True:
    is_successful_frame_read, frame = webcam.read() #frame is numpy array 
    if not is_successful_frame_read:
        break
    
    # Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces_coordinates = face_detector.detectMultiScale(grayscaled_frame) #it hold all faces (multiple)
    
    
    print(faces_coordinates)
    for (x, y, w, h) in faces_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5) 
        #now we extrect each individual face to perform smile function
        #smile is ateach on the face -- so we reduce search area faces(faces_coordinates) to each_indiavudual_face for smile_detector 
        
        # (x,y,w,h) -- has one face(indivual face) -- but it has only cordinate 
        #now we has coordinates of face -- where face is
        #how we extect face_image(the_face ) form the cordinate
   
        the_face=frame[y:y+h,x:x+w]#the_face hold only image of face,not hole frame  #  the_face=(x,y,w,h) # the_face has (x,y,w,h) image  #the_face -- is only one face in faces_cordiantes # so we find smile the_face
        #chage to grayscale
        # =cv2.cvtColor(,cv2.COLOR_BGR2GRAY) #covert only_one_face(each) to gray
        the_face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        # Detect smile
        #now we appy smile_detector in only on face
        #we find smile on the face(the_face) not in hole frame
        smile_coordinates = smile_detector.detectMultiScale(the_face_grayscale,scaleFactor=1.7,minNeighbors=20)#save this paramiter for smile
        eye_coordinates = eye_detector.detectMultiScale(the_face_grayscale,scaleFactor=1.1,minNeighbors=10)
        #scaleFactor=1.7,minNeighbors=20 -- both are use for detecting smile 
        #scaleFactor -- how much do u blur the image
        
         #draw rectangle -- on eye 
        for (x_, y_, w_, h_) in eye_coordinates:
            cv2.rectangle(the_face , (x_, y_), (x_+w_, y_+h_), (0, 255, 255), 2)
        
        #draw rectangle -- on smile 
        for (x__, y__, w__, h__) in smile_coordinates:
            cv2.rectangle(the_face , (x__, y__), (x__+w__, y__+h__), (0, 0, 255), 2)
        # Label this face as smiling
        if len(smile_coordinates) > 0:
            cv2. putText (frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2. FONT_HERSHEY_PLAIN, color= (255, 255, 255))
        if len(eye_coordinates) > 0:
            cv2. putText (frame, 'eye open', (x, y+h+90), fontScale=3, fontFace=cv2. FONT_HERSHEY_PLAIN, color= (255, 255, 255))
        
        
        
        
        
        
        
        
        
        
        
   
    #show current freme
    cv2.imshow ('Smile Detector', frame)
    
    key= cv2.waitKey(1) #every one milisecond fframe will change
     ####Stop if Q key is pressed 
    if key==81 or key==113: #press q to exit
         break
    
#### Release the VideoCapture obiect 
webcam. release ()   
print("code complete") 