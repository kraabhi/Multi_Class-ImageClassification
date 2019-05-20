from keras.layers import MaxPooling2D , Conv2D
from keras.layers import Dense , Flatten ,Dropout , Activation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tqdm import tqdm
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

train =pd.read_csv('train_csv.csv')
train.head()

# Reading images and converting it into array

train_img = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('images/train/' + train['filename'][i], target_size = (224,224) )
    img = image.img_to_array(img)
    train_img.append(img)
X= np.array(train_img)
X = preprocess_input(X)
# same process for test data

test = pd.read_csv('test_csv.csv')
test.head()
test_img = []

for i in tqdm(range(test.shape[0])):
    img = image.load_img('images/test/' + test['filename'][i], target_size= (224 , 224))
    img = image.img_to_array(img)
    #img = img/255  # normalizing image that is converting0-255 into 0-1 range
    test_img.append(img)
test_img1 = np.array(test_img)
test_img1 = preprocess_input(test_img1)   # for each architecture there is fix shape of input that model accepts.
                                          # some model accepts input inpot in ranges (0 ,1) , others (-1, 1)  or values which are not normalized at all.

# Load model weights

model = VGG16(weights = 'imagenet' , include_top = False)
# Extracting features from the train dataset using the VGG16 pre-trained model
features_train = model.predict(X)
# Extracting features from the test dataset using the VGG16 pre-trained model
features_test=model.predict(test_img1)

#features_train.shape
#Reshaping the output of feature_train =[60000,7,7,512] to make it compatible to the MLP input shape

# flattening the layers to conform to MLP input
  
train_x=features_train.reshape(60000,25088) # 25088 = 7 * 7 * 512
# converting target variable to categorical and performing onr hot encoding

train_y = train['label'].values
train_y = to_categorical(train_y)

# creating train and validation set

X_train, X_val, Y_train, Y_val=train_test_split(train_x,train_y,test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(1000,input_dim=25088 , activation = 'relu' ,kernel_initializer = 'uniform'))                 # we use 25088 because we have flattened the data now each observation will go and its shape will be  25088
model.add(Dropout(0.3))
model.add(Dense(500 ,input_dim = 1000 , activation = 'sigmoid' ))
model.add(Dropout(0.5))
model.add(Dense(200 , activation = 'sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(units = 10))
model.add(Activation('softmax'))
        

model.compile(loss ='categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

model.fit(X_train , Y_train , epochs= 10 , batch_size= 128, validation_data=(X_val , Y_val))

# Making prediction on test data

test_data = features_test.reshape(10000, 25088)
test_y = test['label']
test_y = to_categorical(test_y)
predcitions = model.predict_classes(test_data)
#predcitions.shape[0]
#test_y.shape[0]

import sklearn
#from sklearn.metrics import accuracy_score
#accuracy_score(test_y , predcitions)
from sklearn import metrics
print(metrics.accuracy_score(test_y,predcitions))




