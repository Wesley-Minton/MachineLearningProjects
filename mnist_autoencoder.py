from enum import auto
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from tqdm import tqdm
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
latent_dims = 2
num_epochs = 20
batch_size = 128

# Defining the model

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape(-1, 1, 28, 28)

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims=latent_dims)
        self.decoder = Decoder(latent_dims=latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Training the Model
def train(autoencoder, data, epochs=20):
    training_losses = []
    opt = torch.optim.Adam(autoencoder.parameters())
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        running_loss = 0
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            running_loss = running_loss + loss
            loss.backward()
            opt.step()
        training_losses.append(running_loss / batch_size)
    return autoencoder, training_losses

autoencoder = Autoencoder(latent_dims).to(device)

data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size = batch_size,
    shuffle = True
)

# plotting latent space and reconstructed data

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:,0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat

    plt.imshow(img, extent=[*r0, *r1])
    plt.show()

def plot_losses_per_epoch(losses):
    # Detach tensors before converting to numpy
    losses = [loss.detach().numpy() for loss in losses]

    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)

    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)

autoencoder, autoencoder_losses = train(autoencoder, data)
plot_latent(autoencoder, data)
plot_reconstructed(autoencoder)
plot_losses_per_epoch(autoencoder_losses)


# defining a better autoencoder (Variational AutoEncoder)

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
        self.encoder = VariationalEncoder(latent_dims=latent_dims)
        self.decoder = Decoder(latent_dims=latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
# new train to include the error for distance from normal dist
def trainVAE(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    losses = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        run_loss = 0
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            run_loss = run_loss + loss
            loss.backward()
            opt.step()
        losses.append(run_loss / batch_size)
    return autoencoder, losses

vae = VariationalAutoencoder(latent_dims).to(device)
vae, vae_losses = trainVAE(vae, data)

plot_losses_per_epoch(vae_losses)
plot_latent(vae, data)
plot_reconstructed(vae, r0=(-3,3), r1=(-3,3))


# interpolate(vae, x_1, x_2, n=20)
# interpolate_gif(vae, x_1, x_2)
