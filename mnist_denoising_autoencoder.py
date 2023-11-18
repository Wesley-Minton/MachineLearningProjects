from enum import auto
from json.encoder import py_encode_basestring
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm

########################################################################
# FIXME NEED TO SWITCH AWAY FROM MSE TO MAKE RECONSTRUCTION LESS BLURRY #
########################################################################

# Hyperparameters:
batch_size = 128
latent_dims = 4
num_epochs = 20
noise_factor = 0.3

# dataloader for training
data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size = batch_size,
    shuffle = True
)

singleData = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size=1,
    shuffle=True
)

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.midLinear =nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten starting from the second dimension
        x = F.relu(self.linear1(x))
        x = F.relu(self.midLinear(x))
        return self.linear2(x)

    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.midLinear =nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 784)
        
    def forward(self, z):
        z = torch.relu(self.linear1(z))
        z = F.relu(self.midLinear(z))
        z = self.linear2(z)
        return z.reshape(-1, 1, 28, 28)

class DenoisingAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def add_image_noise(img_vector, noise_factor=0.5):
    noise = np.random.normal(0, 1, img_vector.shape)
    noisy_img_vector = img_vector + noise_factor * noise
    return np.clip(noisy_img_vector, 0.0, 1.0)  # Clip values to the valid range [0, 1]

# Get a single data point
iterator = iter(data)
original_image, _ = next(iterator)
original_image = original_image.squeeze().numpy()  # Convert to numpy

# # Add noise to the image
# noisy_image = add_image_noise(original_image, noise_factor)

# # Plot the original and noisy images side by side
# plt.subplot(1, 2, 1)
# plt.imshow(original_image, cmap='gray')
# plt.title('Original Image')

# plt.subplot(1, 2, 2)
# plt.imshow(noisy_image, cmap='gray')
# plt.title('Noisy Image (Factor: {})'.format(noise_factor))

# plt.show()

denoising_autoencoder = DenoisingAutoencoder(latent_dims)

def train_autoencoder(autoencoder, data, num_epochs=20):
    losses = []
    pbar = tqdm(range(num_epochs))
    optimizer = torch.optim.Adam(autoencoder.parameters())
    
    for epoch in pbar:
        running_loss = 0
        for x, _ in data:
            corrupted_x = add_image_noise(x, noise_factor=noise_factor)
            optimizer.zero_grad()

            # Explicitly convert input data to float
            x_hat = autoencoder(corrupted_x.float())
            
            loss = nn.MSELoss()(x_hat, x.float())
            running_loss += loss.item()  # Use .item() to get the loss value as a Python number
            loss.backward()
            optimizer.step()

        losses.append(torch.tensor(running_loss / len(data.dataset)))  # Convert to tensor before appending

    return autoencoder, losses


def plot_losses_per_epoch(losses):
    # Detach tensors before converting to numpy
    losses = [loss.detach().numpy() for loss in losses]

    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()

denoising_autoencoder, losses = train_autoencoder(denoising_autoencoder, data, num_epochs)
plot_losses_per_epoch(losses)


# Get a single data point
iterator = iter(singleData)
for i in range(3):
    original_image, _ = next(iterator)
    original_image = original_image.squeeze().numpy()  # Convert to numpy and remove batch dimension

    # Add noise to the image
    noisy_image = add_image_noise(original_image, noise_factor)

    # Send the noisy image through the denoising autoencoder
    denoised_image = denoising_autoencoder(torch.Tensor([noisy_image]))

    # Plot the original, noisy, and denoised images side by side
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image (Factor: {})'.format(noise_factor))

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image.squeeze().detach().numpy(), cmap='gray')
    plt.title('Denoised Image')

    plt.show()


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784,512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0,1)
        # self.N.loc = self.N.loc.cuda
        # self.N.scale = self.N.scale.cuda
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def trainVAE(autoencoder, data, epochs=20):
    optimizer = torch.optim.Adam(autoencoder.parameters())
    losses = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        run_loss = 0
        for x, y in data:
            corrupted_x = add_image_noise(x, noise_factor=noise_factor)
            optimizer.zero_grad()

            # Explicitly convert input data to float
            x_hat = autoencoder(corrupted_x.float())
            
            loss = nn.MSELoss()(x_hat, x.float()) + autoencoder.encoder.kl
            run_loss = run_loss + loss
            loss.backward()
            optimizer.step()
        losses.append(run_loss / batch_size)
    return autoencoder, losses

# variational_autoencoder = VariationalAutoencoder(latent_dims)
# variational_autoencoder, losses = trainVAE(variational_autoencoder, data, num_epochs)

# iterator = iter(singleData)
# for i in range(3):
#     original_image, _ = next(iterator)
#     original_image = original_image.squeeze().numpy()  # Convert to numpy and remove batch dimension

#     # Add noise to the image
#     noisy_image = add_image_noise(original_image, noise_factor)

#     # Send the noisy image through the denoising autoencoder
#     denoised_image = variational_autoencoder(torch.Tensor([noisy_image]))

#     # Plot the original, noisy, and denoised images side by side
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_image, cmap='gray')
#     plt.title('Original Image')

#     plt.subplot(1, 3, 2)
#     plt.imshow(noisy_image, cmap='gray')
#     plt.title('Noisy Image (Factor: {})'.format(noise_factor))

#     plt.subplot(1, 3, 3)
#     plt.imshow(denoised_image.squeeze().detach().numpy(), cmap='gray')
#     plt.title('Denoised Image')

#     plt.show()