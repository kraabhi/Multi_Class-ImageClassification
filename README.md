# Multi Class Image Classification

We have is a training dataset that consists of number of images, each labeled with one of K different classes.We use this training data set to train our classifier and then evaluate our model performance to predict the labels for a new set of images and compare the predicted results with actual labels.

### Transfer Learning
Before going into deep we must have knowledge of transfer learning which can be explained with the example of professor and students perspective. A Professor learned through out his life and share this information to students.SImilarly a model is trained on lots of data and the information is complied in form of weights .These weights can be extracted and then transferred to any other neural network. Instead of training the other neural network from scratch, we “transfer” the learned features.One more example of transfer learning is transfer of knowledge from one generation to another.

### Pre Trained Model
Simply put, a pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, you use the model trained on other problem as a starting point.

When we are trying to use pre-trained models we would not have to train entire architecture but only a few layers.

### How to use Pre trained Model
When we train a neural network we wish to identify correct weights by using multiple forward and backward iterations.So intead to doing again andagain Forward and Back iterations we use pre trained models which ahve already been trained on large data sets , we can directly use their weights and architecture and apply the learning on our problem statement. This is known as transfer learning. We “transfer the learning” of the pre-trained model to our specific problem statement.Since we assume that the pre-trained network has been trained quite well, we would not want to modify the weights too soon and too much. While modifying we generally use a learning rate smaller than the one used for initially training the model.
### Scenario 1 – 
Size of the Data set is small while the Data similarity is very high – In this case, since the data similarity is very high, we do not need to retrain the model. All we need to do is to customize and modify the output layers according to our problem statement. 
### Scenario 2 –
Size of the data is small as well as data similarity is very low – We freeze the initial layers and traint he remaining layers.The top layers would be customized to the new data set.The small size of the data set is compensated by the fact that the initial layers are kept pretrained and the weights for those layers are frozen.
### Scenario 3 – 
Size of the data set is large however the Data similarity is very low - We need to train the model from scratch as data is large and there is lwess similarity so use of pre trained model weights would not be effective.
### Scenario 4 – 
Size of the data is large as well as there is high data similarity : This is best situaltion wher we can use our pretrained model as it is.

### These are few previously trained model’s that can be used for transfer learning.

1. Imagenet
2. VGG16 ,VGG19
3. ResNet
4. Inception V3
5. Xception

### Our data needs to be in a particular format in order to solve an image classification problem. 
We will see this in action in a couple of sections but just keep these pointers in mind till we get there.
You should have 2 folders, one for the train set and the other for the test set. In the training set, you will have a .csv file and an image folder:

1. The .csv file contains the names of all the training images and their corresponding true labels
2. The image folder has all the training images.

## Task 1 :Apparel classification 
We have a total of 70,000 images (28 x 28 dimension), out of which 60,000 are from the training set and 10,000 from the test one. The training images are pre-labelled according to the apparel type with 10 total classes. The test images are, of course, not labelled. The challenge is to identify the type of apparel present in all the test images.

Data can be found at :

https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-apparels

I tried to train model from scrach using small layer network first then I will see the improvement in performance by using transfer learning


