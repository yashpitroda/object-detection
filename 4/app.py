# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
''''
Label           Description
1               T-shirt/top (if image is T-shirt/top then it reurn 1)
2               Trouser
3               Pullover
4               Coat
5               Sandal
6               Shirt
7               Sneaker
8               Bag
9               Ankle boot
'''


productmap={
1        :       "T-shirt/top ",
2    :           "Trouser",
3     :          "Pullover",
4      :         "Coat",
5       :        "Sandal",
6        :       "Shirt",
7         :      "Sneaker",
8          :     "Bag",
9           :    "Ankle boot",
}
import tensorflow as tf 
import numpy as np 
from tensorflow import keras
# Printing stuff
import matplotlib.pyplot as plt

# Load a pre-defined dataset (70k of 28×28 )
fashion_mnist = keras.datasets.fashion_mnist #use data form karas website database

# Pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()#60k in train #10l in tesst

# Show data
# print(train_labels [0]) #model will retrun this 

#print(train_images [0])
plt.imshow(train_images [0], cmap='gray',vmin=0,vmax=255)
plt.show ()

# Define our neural net structure
model = keras.Sequential( [
    #input layer -- # input is a 28x28 image (I'Flatten" flattens the 28×28 into a single 784×1 input layer)
    keras. layers.Flatten(input_shape= (28,28)),
    
    # hidden layer is 128 dee 128: int turns the value, or 0 (works good enough. much faster)
    keras. layers.Dense(units =124, activation=tf.nn. relu),
    
    # output is 0-10 (depending on what piece of clothing it is). return maximum
    keras. layers. Dense ( units=10, activation=tf.nn.softmax) 
])

# Compile our model
model.compile(optimizer=tf.optimizers.Adam() , loss='sparse_categorical_crossentropy',metrics= ['accuracy'])

# Train our model, using our training data
model. fit (train_images, train_labels, epochs=5)

# Test our model, using our testing data
test_loss = model. evaluate(test_images, test_labels)

# Make predictions
predictions = model.predict (test_images)

print (predictions [1])
# Print out prediction
print (list(predictions [1]). index (max (predictions [1]) ))
print('===========')
# Print the correct answer
print(test_labels [1])
print ("Done.")

print('code complete')