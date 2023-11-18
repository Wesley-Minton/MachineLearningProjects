import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random


from tqdm import tqdm

import matplotlib.pyplot as plt;  

# device is set to cpu bc cuda is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Hyperparameters:
batch_size = 128
num_epochs = 20
noise_factor = 0.3
learning_rate = 1e-4
num_disc_iters = 10
num_gen_iters = 5

# downloading and setting datasets for train/test
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

# dataloader for training
data = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size = batch_size,
    shuffle = True
)

def add_image_noise(img_vector, noise_factor=0.5):
    noise = np.random.normal(0, 1, img_vector.shape)
    noisy_img_vector = img_vector + noise_factor * noise
    return np.clip(noisy_img_vector, 0.0, 1.0)  # Clip values to the valid range [0, 1]

original_image, _ = mnist_testset[random.randint(0, len(mnist_testset)-1)]
original_image = original_image.numpy()

# Add noise to the image
noisy_image = add_image_noise(original_image, noise_factor)

# Plot the original and noisy images side by side
plt.subplot(1, 2, 1)
plt.imshow(original_image.squeeze(), cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image.squeeze(), cmap='gray')
plt.title('Noisy Image (Factor: {})'.format(noise_factor))

plt.show()

# Creating GAN (Generative Adversarial Network)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 784)
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
        
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


generator = Generator()
discriminator = Discriminator()

gen_optimizer = optim.Adam(generator.parameters(), lr = learning_rate)
disc_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate)

loss_function = nn.BCELoss()

# training the model:
pbar = tqdm(range(num_epochs))
disc_losses = []
gen_losses = []

for epoch in pbar:
    for _ in range(num_disc_iters):
        # optimizing discriminator for real data
        disc_optimizer.zero_grad()

        real_data, _ = next(iter(data))  # Get a batch of real data
        real_data = real_data.view(real_data.size(0), -1).to(device)  # Flatten and move to device
        real_output = discriminator(real_data)
        real_labels = torch.ones_like(real_output)

        disc_real_loss = loss_function(real_output, real_labels)
        disc_real_loss.backward()

        # optimizing discriminator for generated data
        z = torch.randn(batch_size, 784).to(device)  # Random noise
        fake_data = generator(z)
        fake_output = discriminator(fake_data.detach())  # Detach to prevent generator gradients from flowing
        fake_labels = torch.zeros_like(fake_output)

        disc_fake_loss = loss_function(fake_output, fake_labels)
        disc_fake_loss.backward()

        disc_optimizer.step()

    for _ in range(num_gen_iters):
        # gen_optimizer.zero_grad()

        # z = torch.randn(batch_size, 784).to(device)  # Random noise
        # fake_data = generator(z)
        # gen_output = discriminator(fake_data)
        # gen_labels = torch.ones_like(gen_output)

        # gen_loss = loss_function(gen_output, gen_labels)
        # gen_loss.backward()

        # gen_optimizer.step()
        gen_optimizer.zero_grad()

        # Get a batch of noisy images from the dataloader
        noisy_data, _ = next(iter(data))
        noisy_data = noisy_data.view(noisy_data.size(0), -1).to(device)

        # Generate denoised images from the generator
        fake_data = generator(noisy_data)
        
        # Pass the generated denoised images through the discriminator
        gen_output = discriminator(fake_data)
        
        # Label all generated images as real (since the generator is trying to mimic real images)
        gen_labels = torch.ones_like(gen_output)

        # Calculate generator loss and perform backpropagation
        gen_loss = loss_function(gen_output, gen_labels)
        gen_loss.backward()
        
        # Update generator parameters
        gen_optimizer.step()
    
    with torch.no_grad():  # To avoid computation of gradients
        # Calculate discriminator loss
        disc_loss = (disc_real_loss + disc_fake_loss).item()
        disc_losses.append(disc_loss)

        # Calculate generator loss
        gen_loss = gen_loss.item()
        gen_losses.append(gen_loss)


plt.figure(figsize=(10, 5))
plt.plot(disc_losses, label='Discriminator Loss')
plt.plot(gen_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Discriminator and Generator Loss over Epochs')
plt.legend()
plt.show()


original_image, _ = mnist_testset[random.randint(0, len(mnist_testset)-1)]
original_image = original_image.numpy()
noisy_image = add_image_noise(original_image, noise_factor)
# Convert noisy_image to PyTorch tensor and add batch dimension
noisy_image_tensor = torch.from_numpy(noisy_image).float().unsqueeze(0)
# noisy_image_tensor = torch.randn(1, 784)

# Pass the noisy image through the generator
denoised_image = generator(noisy_image_tensor)

# Reshape denoised_image to match the shape of the image
denoised_image = denoised_image.reshape(-1, 1, 28, 28)

# Plot the original, noisy, and denoised images side by side
plt.subplot(1, 3, 1)
plt.imshow(original_image.squeeze(), cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image.squeeze(), cmap='gray')
plt.title('Noisy Image (Factor: {})'.format(noise_factor))

plt.subplot(1, 3, 3)
plt.imshow(denoised_image.squeeze().detach().numpy(), cmap='gray')  # Convert to NumPy for plotting
plt.title('Denoised Image')

plt.show()
