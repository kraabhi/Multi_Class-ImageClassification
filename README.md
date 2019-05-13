# Dogs-Breed-Classification
# Dog breed Image classification model

We have is a training dataset that consists of number of images, each labeled with one of K different classes.We use this training data set to train our classifier and then evaluate our model performance to predict the labels for a new set of images and compare the predicted results with actual labels.

### Transfer Learning
Before going into deep we must have knowledge of transfer learning which can be explained with the example of professor and students perspective. A Professor learned through out his life and share this information to students.SImilarly a model is trained on lots of data and the information is complied in form of weights .These weights can be extracted and then transferred to any other neural network. Instead of training the other neural network from scratch, we “transfer” the learned features.One more example of transfer learning is transfer of knowledge from one generation to another.

### Pre Trained Model
Simply put, a pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, you use the model trained on other problem as a starting point.

When we are trying to use pre-trained models we would not have to train entire architecture but only a few layers.

### How to use Pre trained Model
When we train a neural network we wish to identify correct weights by using multiple forward and backward iterations.So intead to doing again andagain Forward and Back iterations we use pre trained models which ahve already been trained on large data sets , we can directly use their weights and architecture and apply the learning on our problem statement. This is known as transfer learning. We “transfer the learning” of the pre-trained model to our specific problem statement.Since we assume that the pre-trained network has been trained quite well, we would not want to modify the weights too soon and too much. While modifying we generally use a learning rate smaller than the one used for initially training the model.

