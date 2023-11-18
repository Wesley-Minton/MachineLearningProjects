### Author: Wesley Minton
### Date: 11/09/2023
###
### followed tutorial from: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# device is set to cpu bc cuda is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# downloading and setting datasets for train/test
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

# printing sample data
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(mnist_trainset), size=(1,)).item()
    img, label = mnist_trainset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# creating loaders to iteratively take batches of data from sets
loaders = {
    'train': torch.utils.data.DataLoader(
        mnist_trainset,
        batch_size=100,
        shuffle=True,
        num_workers=0  # Set num_workers to 0 to disable multiprocessing
    ),
    'test': torch.utils.data.DataLoader(
        mnist_testset,
        batch_size=100,
        shuffle=True,
        num_workers=0  # Set num_workers to 0 to disable multiprocessing
    ),
}


# Defining the convolutional Neural Network (CNN)

# Parameters:
    # in_channels (int) — Number of channels in the input image (one because its grayscale)
    # out_channels (int) — Number of channels produced by the convolution
    # kernel_size (int or tuple) — Size of the convolving kernel
    # stride (int or tuple, optional) — Stride of the convolution. Default: 1
    # padding (int or tuple, optional) — Zero-padding added to both sides of the input. Default: 0
    # padding_mode (string, optional) — ‘zeros’, ‘reflect’, ‘replicate’ or ‘circular’. Default: ‘zeros’
    # dilation (int or tuple, optional) — Spacing between kernel elements. Default: 1
    # groups (int, optional) — Number of blocked connections from input channels to output channels. Default: 1
    # bias (bool, optional) — If True, adds a learnable bias to the output. Default: True

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # fully connected 32x7x7 input to 10 output
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Running through first convolution (conv kernel --> activation fxn --> pooling)
        x = self.conv1(x)
        
        # Running through second convolution (conv kernel --> activation fxn --> pooling)
        x = self.conv2(x)
        
        # flattening output to (batch size, 32 * 7 * 7) to be fed into the input 
        # layer of classification MLP
        x = x.view(x.size(0), -1)
        
        # Running through our fully connected output layer
        output = self.out(x)
        
        # returning our conv output and x for visualization
        return output, x
        
cnn = CNN()

# Training our model:
loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

num_epochs = 10

def train(num_epochs, cnn, loaders):
    cnn.train()

    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            b_x = Variable(images) # batch x
            b_y = Variable(labels) # batch y

            # making predictions about batch data
            output = cnn(b_x)[0]

            # comparing our output labels with actual labels to get loss
            loss = loss_func(output, b_y)

            # important to clear gradients for this training step
            optimizer.zero_grad()

            # backpropogation, compute gradients
            loss.backward()

            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print( 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

train(num_epochs, cnn, loaders)

def test():
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        print('Test Accuracy of the model on the 10000 test images: %.2f', accuracy)
test()

sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:10].numpy()

test_output, last_layer = cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f"Prediction number: {pred_y}")
print(f"Actual number: {actual_number}")