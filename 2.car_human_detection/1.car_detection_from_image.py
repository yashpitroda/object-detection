import cv2
from sklearn.model_selection import train_test_split

#get path
img_file_path =   '/Users/yash/playground/AI_clever_programer/2.car_human_detection/3.jpg'
classifier_file_path = '/Users/yash/playground/AI_clever_programer/2.car_human_detection/car_detection.xml'

# create classifier  -- use CascadeClassifier
trained_car_data = cv2.CascadeClassifier(classifier_file_path)

# create openc image1
img = cv2.imread(img_file_path)

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect car -- it retruns cordinates of rectengle suranding the face 
car_coordinates = trained_car_data.detectMultiScale(grayscaled_img)
print(car_coordinates)

for (x, y, w, h) in car_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)#(x, y) --uper left ,(x+w, y+h) : lowerright  -- #bgr color #green # last 2 is theekness of the rectangle
    # cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)#(x
 


# Display the image with the faces spotted
cv2.imshow('yash Car Detector', img)
# Dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey ()
 
print ("Code Completed") 