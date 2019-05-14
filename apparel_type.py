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
train = pd.read_csv("Fashion/train.csv")
train.head()

# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('Fashion/train/'+train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

# As it is a multi-class classification problem (10 classes), we will one-hot encode the target variable.

y = train['label'].values

y = to_categorical(y)
X_train , X_test ,y_train , y_test = train_test_split(X ,y , random_state = 42 , test_size = 0.2)

# We define the model structure , here we use a simple architecture with two convolution layer

model = Sequential()
model.add(Conv2D(32 , kernel_size =(3,3) , activation = 'relu' , input_shape = (28,28,1) ))
model.add(Conv2D(64 , kernel_size =(3,3) , activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128 ,activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

# compiling the model 

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy' , metrics =['accuracy'])

X_train.shape
X_test.shape
model.fit(X_train , y_train , epochs= 5 , validation_data=(X_test , y_test))

# we will predict the results for the test data

test = pd.read_csv('Fashion/test.csv')
test_image = []

for i in tqdm(range(test.shape[0])):
    img = image.load_img("Fashion/test/" + test['id'][i].astype(str) +'.png' , color_mode= "grayscale",target_size=(28,28,1))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test_image1 = np.array(test_image)
predictions = model.predict_classes(test_image1)
