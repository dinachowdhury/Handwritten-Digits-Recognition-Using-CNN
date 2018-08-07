# Handwritten-Digits-Recognition-Using-CNN
I try to explore a model build by convolutional neural network (CNN), a popular and modern technique for handwritten digits’ recognition. And get 96% accuracy with small loss.

The dataset I used here has total 70,000 data and they are divided into three segments: 55,000 data for training (mnist. train), 10,000 data for testing (mnist. test) and 5,000 data for validation (mnist. Validation). 
Each data form MNIST has two parts: an image of a handwritten digits and its corresponding label. Each image is 28 pixels by 28 pixels. And it can be represented in an array vector 28 X 28 = 728 pixels.

Dataset Link: http://yann.lecun.com/exdb/mnist/

https://en.wikipedia.org/wiki/MNIST_database

The coding is done using Python. At the very beginning of my experiment, first convolution layer has 32 feature maps, each of which with a resolution of 28x28 and respective file 5x5. The tensor I setup is [5,5,1,32]. Means, first two values are patch size, the next number is input channels and last number is for each output channel. It is mentionable that the stride size =1 in my experiment along with this zero padding also used here. After reshaping the input from 2d tensor to a 4d tensor, then I convolute the image with weight tensor, add bias and apply rectifier function to minimize the noise. Then applied the 2x2 max pooling and the image size reduced to 14x14. To build a deep network I used second convolution layer which has 64 feature for each 5x5 patch. In this stage image size reduced to 7x7. With the reduced image size a fully connected layer with 1024 neurons included, which will process the entire image. Another rectifier used here again. During my experiment, I used dropout technique to reduce the overfitting. Finally, last layer is SoftMax regression layer. 
