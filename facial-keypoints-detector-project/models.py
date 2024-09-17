## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
       
        ## 1. This network takes in a square (same width and height), grayscale image as input
        self.conv1 = nn.Conv2d(1, 16, 5) # 1 input image
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization for 16 feature maps
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2  = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization for 32 feature maps
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        #self.conv3  = nn.Conv2d(32, 64, 2)
        #self.bn3 = nn.BatchNorm2d(128)  # Batch normalization for 64 feature maps
        #self.pool3 = nn.MaxPool2d(2, stride=2)
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 64*53*53 -> depth*length*width of image
        self.fc1 = nn.Linear(54*54*32, 512)  # Flattened size: 387200 -> 512
        self.fc2 = nn.Linear(512, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


        
     
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        #x = self.pool3(F.leaky_relu(self.conv3(x)))
        #x = self.drop(x)
        x = self.flat(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
