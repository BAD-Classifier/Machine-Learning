# Machine-Learning
### Convolutional Neural Network (CNN)
![](https://i.imgur.com/m4WHJl1.png)  

This project uses a CNN to learn the MFCCs for each of the bird species. The CNN architecture is made up of:
- conv2d - 32 filters
- conv2d - 32 filters
- maxpooling of 3 x 3
- conv2d - 64 filters
- conv2d - 64 filters
- maxpooling 2 x 2
- conv2d - 128 filters
- conv2d - 128 filters
- maxpooling 2 x 2
- flatten
- dense 1024
- dense 10 (for 10 birds)
- softmax

Each conv2d layer uses relu activations and batch normalization after the activations. Dropout of 50% is used on each conv2d layer while 80% drop is used on the dense 1024 layer. Adam optimizer is used. Data is augmented to increase the sample size and to add some regularization of the data.
