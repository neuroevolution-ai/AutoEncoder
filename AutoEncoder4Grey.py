import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Auto-Encoder 4 - kann mit Graustufen-Bildern umgehen
"""
class AutoEncoder4Grey(nn.Module):
    def __init__(self):
        super(AutoEncoder4Grey, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=(2,2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)

        # Latent View

        self.lv1 = nn.Linear(5200, 400)
        self.lv2 = nn.Linear(400, 30)

        self.fc_mu = nn.Linear(30, 30)
        self.fc_logsigma = nn.Linear(30, 30)

        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 5200)

        #Decoder
        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=7, stride=1)
        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, indices1 = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x, indices2 = self.maxpool2(x)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        x = eps.mul(sigma).add_(mu)
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = self.unmaxpool1(x, indices2)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        x = self.unmaxpool2(x, indices1, output_size=torch.Size([x.size(0), x.size(1), 210, 160]))
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        return x