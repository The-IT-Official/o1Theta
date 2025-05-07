import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)
batch_size = 64
lr = 0.0002
n_epochs = 50
sample_interval = 500

# Data Loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
    
# Initialize generator and discriminator
G = Generator().to(device)
D = Discriminator().to(device)
adversarial_loss = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real = imgs.to(device)
        batch_size = real.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        g_loss = adversarial_loss(D(fake_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(D(real), real_labels)
        fake_loss = adversarial_loss(D(fake_imgs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        if i % sample_interval == 0:
            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

def sample_image(n_row=5):
    z = torch.randn(n_row ** 2, latent_dim).to(device)
    gen_imgs = G(z)
    gen_imgs = gen_imgs.view(-1, *img_shape)
    gen_imgs = gen_imgs.cpu().detach().numpy()
    fig, axs = plt.subplots(n_row, n_row, figsize=(5,5))
    for i in range(n_row):
        for j in range(n_row):
            axs[i, j].imshow(gen_imgs[i * n_row + j][0], cmap='gray')
            axs[i, j].axis('off')
    plt.show()
sample_image()