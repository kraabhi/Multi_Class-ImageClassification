# Import necessary libraries

import keras
from keras import Sequential
from keras.layers import MaxPooling2D , Conv2D
from keras.layers import Dense , Flatten ,Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tqdm import tqdm

# reading train file with labels for image ,  give the path where you save your images. In my case it is in digits/train.
train = pd.read_csv('digits/train.csv')
train.head()
train.label.nunique()
range(train.shape[0])

# read train image 
train_img = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img("digits/Images/train/" + train['filename'][i], target_size = (28,28,1) , color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    train_img.append(img)
X =np.array(train_img)

# Target variable is not binary so need to do one hot encoding
y = train['label'].values
y = to_categorical(y)

# Split data into train and validation set

X_train , X_test , y_train ,y_test =train_test_split(X,y , test_size = 0.2 , random_state = 42 )

# Building model

model = Sequential()
model.add(Conv2D(32 , kernel_size= (3,3) , activation= 'relu' , input_shape = (28,28,1)))
model.add(Conv2D(64 , kernel_size =(3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128 ,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10 , activation = 'softmax'))

model.compile(loss= 'categorical_crossentropy' , optimizer = 'adam' , metrics =['accuracy'])

model.fit(X_train , y_train , epochs= 5 , validation_data= (X_test , y_test))

# making predictions for the test data
test = pd.read_csv('digits/Test_fCbTej3.csv')
test.head()

# loading test image
test_img = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('digits/Images/test/' + test['filename'][i] ,target_size=(28,28,1) , color_mode='grayscale')
    img = image.img_to_array(img)
    img = img/255
    test_img.append(img)
test_image1 = np.array(test_img)

prediction = model.predict_classes(test_image1)

