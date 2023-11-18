# Author: Wesley Minton
# Date: 11/09/2023
#

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# device is set to cpu bc cuda is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def show_image(img):
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Specify the path to the CIFAR-10 dataset
cifar10_custom_path = './data/CIFAR-10'

# Load CIFAR-10 dataset with the custom path
cifar10_trainset = CIFAR10(root=cifar10_custom_path, train=True, download=False, transform=transforms.ToTensor())
cifar10_testset = CIFAR10(root=cifar10_custom_path, train=False, download=False, transform=transforms.ToTensor())

loaders = {
    'train' : torch.utils.data.DataLoader(cifar10_trainset, batch_size=100, shuffle=True, num_workers=1),
    'test' : torch.utils.data.DataLoader(cifar10_testset, batch_size=100, shuffle=True, num_workers=1)
}

# Visualize a few images from the training set
# num_images_to_visualize = 5

# for i in range(num_images_to_visualize):
#     # Get a random index
#     random_index = np.random.randint(len(cifar10_trainset))
    
#     # Get the image and label
#     image, label = cifar10_trainset[random_index]
    
#     # Show the image
#     show_image(image)
#     print(f"Label: {label}")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

cnn = CNN()

# training model:
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
num_epochs = 5
def train(num_epochs, cnn, loaders):
    cnn.train()

    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)[0]
            loss = loss_function(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

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

if __name__ == '__main__':
    train(num_epochs, cnn, loaders)

    test()
