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
test_img1 = preprocess_input(test_img1)


