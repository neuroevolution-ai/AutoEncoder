"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zum Testen einzelner Bilder mit einem trainierten Auto-Encoder.
Dafür kann manuell ein Bild übergeben werden, oder mehrere Bilder ab einem bestimmten Index im Datensatz.

Implementiert sind vier Auto-Encoder Architekturen.
AutoEncoder1 orientiert sich an:
Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631.
https://github.com/ctallec/world-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from numpy import asarray
from visdom import Visdom
import utils
import cv2

"""
Auto-Encoder 1 - kann mit RGB-Bildern umgehen
"""
class AutoEncoder1(nn.Module):


    def __init__(self):
        super(AutoEncoder1, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        # Latent View
        self.lv1 = nn.Linear(22528, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 22528)

        #Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, output_padding=(1,0))
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=(0,1))
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = x[:, :, :210, :160]
        return x

"""
Auto-Encoder 1 - kann mit Graustufen-Bildern umgehen
"""
class AutoEncoder1Grey(nn.Module):


    def __init__(self):
        super(AutoEncoder1Grey, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        # Latent View
        self.lv1 = nn.Linear(22528, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 22528)

        #Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, output_padding=(1,0))
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=(0,1))
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = x[:, :, :210, :160]
        return x

"""
Auto-Encoder 2 - kann mit RGB-Bildern umgehen
"""
class AutoEncoder2(nn.Module):


    def __init__(self):
        super(AutoEncoder2, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=8, stride=4)

        # Latent View
        self.lv1 = nn.Linear(7680, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 7680)

        #Decoder
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=8, stride=4, output_padding=(1, 0))
        self.deconv2 = nn.ConvTranspose2d(20, 3, kernel_size=5, stride=3, output_padding=(1, 2))



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return x

"""
Auto-Encoder 2 - kann mit Graustufen-Bildern umgehen
"""
class AutoEncoder2Grey(nn.Module):


    def __init__(self):
        super(AutoEncoder2Grey, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=8, stride=4)

        # Latent View
        self.lv1 = nn.Linear(7680, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 7680)

        #Decoder
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=8, stride=4, output_padding=(1, 0))
        self.deconv2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=3, output_padding=(1, 2))



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return x

"""
Auto-Encoder 3 - kann mit RGB-Bildern umgehen
"""
class AutoEncoder3(nn.Module):

    def __init__(self):
        super(AutoEncoder3, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=(2,2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=(3,3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)

        # Latent View
        self.lv1 = nn.Linear(5200, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 5200)

        #decoder
        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=7, stride=1)
        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(20, 3, kernel_size=5, stride=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x,indices1 = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x,indices2 = self.maxpool2(x)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = self.unmaxpool1(x,indices2)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        x = self.unmaxpool2(x,indices1, output_size=torch.Size([x.size(0),x.size(1),210,160]))
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        return x

"""
Auto-Encoder 3 - kann mit Graustufen-Bildern umgehen
"""
class AutoEncoder3Grey(nn.Module):
    def __init__(self):
        super(AutoEncoder3Grey, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=(2,2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=(3,3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)

        # Latent View
        self.lv1 = nn.Linear(5200, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 5200)

        #Decoder
        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=7, stride=1)
        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=1)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x,indices1 = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x,indices2 = self.maxpool2(x)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = self.unmaxpool1(x,indices2)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        x = self.unmaxpool2(x,indices1, output_size=torch.Size([x.size(0),x.size(1),210,160]))
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        return x

"""
Auto-Encoder 4 - kann mit RGB-Bildern umgehen
"""
class AutoEncoder4(nn.Module):

    def __init__(self):
        super(AutoEncoder4, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=(2, 2))
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
        self.deconv2 = nn.ConvTranspose2d(20, 3, kernel_size=5, stride=1)


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

"""
Funktion zeigt ein Bild
"""
def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor)
    plt.show()

"""
Funktion normalisiert einen Pixelwert
"""
def normalize(v):
    return v / 255

"""
Funktion liefert ein RGB-Bild in einem Datensatz mit einem bestimmten Index
"""
def getFrame(dataset, index):
    frame = normalize(dataset[index:index+1, :, :, :])
    torchFrame = torch.from_numpy(frame).type(torch.FloatTensor)
    torchFrame = torchFrame.permute(0, 3, 1, 2)
    return torchFrame[:, :, :, :]

"""
Funktion liefert ein Graustufen-Bild in einem Datensatz mit einem bestimmten Index
"""
def getFrameGrey(dataset, index):
    dataset = dataset.reshape(len(dataset), 210, 160, 1)
    frame = normalize(dataset[index:index + 1, :, :, :])
    torchFrame = torch.from_numpy(frame).type(torch.FloatTensor)
    torchFrame = torchFrame.permute(0, 3, 1, 2)
    return torchFrame[:, :, :, :]


if __name__ == '__main__':
    """
    Hier kann manuell die Auto-Encoder-Architektur gewählt werden 
    Zur Auswahl stehen nur die Auto-Encoder, die RGB-Bilder verarbeiten können.
    """
    ae = AutoEncoder1()
    # ae = AutoEncoder1Grey()
    # ae = AutoEncoder2()
    # ae = AutoEncoder2Grey()
    # ae = AutoEncoder3()
    # ae = AutoEncoder3Grey()
    # ae = AutoEncoder4()
    # ae = AutoEncoder4Grey()

    # Hier wird der Pfad für den gespeicherten Auto-Encoder geladen
    #ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_RGB_1/Q*Bert_JustQBert/!Episode1/ae.pt'))
    ae.eval()

    # Hier muss manuell der Datensatz ausgewählt werden, auf dem der Auto-Encoder getestet werden soll
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_JustQBert/smallDatasetTest_JustQBert.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders/smallDatasetTest_SpaceInvaders.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTest_SpaceInvaders_Greyscale.npy')

    """
    Hier wird manuell ein RGB-Bild in den Auto-Encoder gegeben und der Loss ermittelt.
    Eingabe- und Ausgabe-Bild werden nach Ausführung angezeigt.
    """
    imgArray = []
    img = Image.open('/home/annika/Bachelorarbeit/Vergleichsbilder/EingabeSI.png')
    imgArray.append(asarray(img))
    print(len(imgArray))
    print(imgArray[0].shape)
    img = Image.open('/home/annika/Bachelorarbeit/Vergleichsbilder/Eingabe_Schwierig_Q.png')
    imgArray.append(asarray(img))
    print(len(imgArray))
    print(imgArray[0].shape)
    frame = getFrame(np.array(imgArray), 0)
    pred = ae(frame)
    pred = pred.permute(0, 2, 3, 1)

    print(pred.shape)
    frame = frame.permute(0, 2, 3, 1)
    show_torch_image(frame[0].detach())
    show_torch_image(pred[0].detach())

    loss = 0
    for l in range(210):
        for w in range(160):
            fst = np.square(np.subtract(frame[0, l, w, 0].detach(), pred[0, l, w, 0].detach())).mean()
            snd = np.square(np.subtract(frame[0, l, w, 1].detach(), pred[0, l, w, 1].detach())).mean()
            trd = np.square(np.subtract(frame[0, l, w, 2].detach(), pred[0, l, w, 2].detach())).mean()
            loss += (fst + snd + trd) / 3
    loss = loss / (210 * 160)
    print('Bild {} - Loss: {}'.format(0, loss))

    """
    Hier wird manuell ein Graustufen-Bild in den Auto-Encoder gegeben und der Loss ermittelt.
    Eingabe- und Ausgabe-Bild werden nach Ausführung angezeigt.
    """

    imgArray = []
    img = cv2.imread('/home/annika/Bachelorarbeit/Vergleichsbilder/EingabeQ.png')
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgArray.append(asarray(imgGrey))
    print(len(imgArray))
    print(imgArray[0].shape)
    img = cv2.imread('/home/annika/Bachelorarbeit/Vergleichsbilder/Eingabe_Schwierig_Q.png')
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgArray.append(asarray(imgGrey))
    print(len(imgArray))
    print(imgArray[0].shape)
    frame = getFrameGrey(np.array(imgArray), 1)
    pred = ae(frame)
    pred = pred.permute(0, 2, 3, 1)

    print(pred.shape)
    frame = frame.permute(0, 2, 3, 1)
    yay = torch.tensor([255], dtype=torch.int)
    imgFrame = frame * yay
    plt.imshow(imgFrame[0].reshape(210, 160).detach().cpu().numpy().astype(np.uint8), cmap=plt.get_cmap('gray'))
    plt.show()
    imgpred = pred * yay
    plt.imshow(imgpred[0].reshape(210, 160).detach().cpu().numpy().astype(np.uint8), cmap=plt.get_cmap('gray'))
    plt.show()

    loss = 0
    for l in range(210):
        for w in range(160):
            x = np.square(np.subtract(frame[0, l, w].detach(), pred[0, l, w].detach())).mean()
            loss += x
    loss = loss / (210 * 160)
    print('Bild {} - Loss: {}'.format(0, loss))

    """
    Hier werden RGB-Bilder aus dem ausgewählten Testdatensatz 
    an einem bestimmten Index ausgewählt und in den Auto-Encoder gegeben und der Loss ermittelt.
    
    start: ist der Indexstart im Datensatz
    count: ist die Anzahl an Bildern, die ab dem definierten Index in den Auto-Encoder gegeben werden sollen
    """
    start = 21
    count = 1
    for i in range(count):
        frame = getFrame(datasetTest, start + i)
        pred = ae(frame)
        pred = pred.permute(0, 2, 3, 1)

        print(pred.shape)
        frame = frame.permute(0, 2, 3, 1)
        show_torch_image(frame[0].detach())
        show_torch_image(pred[0].detach())

        loss = 0
        for l in range(210):
            for w in range(160):
                fst = np.square(np.subtract(frame[0,l,w,0].detach(),pred[0,l,w,0].detach())).mean()
                snd = np.square(np.subtract(frame[0,l,w,1].detach(),pred[0,l,w,1].detach())).mean()
                trd = np.square(np.subtract(frame[0,l,w,2].detach(),pred[0,l,w,2].detach())).mean()
                loss += (fst + snd + trd)/3
        loss =  loss/(210*160)
        print('Bild {} - Loss: {}'.format(i,loss))

    """
    Hier werden Graustufen-Bilder aus dem ausgewählten Testdatensatz 
    an einem bestimmten Index ausgewählt und in den Auto-Encoder gegeben und der Loss ermittelt.

    start: ist der Indexstart im Datensatz
    count: ist die Anzahl an Bildern, die ab dem definierten Index in den Auto-Encoder gegeben werden sollen
    """
    for i in range(count):
        frame = getFrameGrey(datasetTest, start + i)
        pred = ae(frame)
        pred = pred.permute(0, 2, 3, 1)

        print(pred.shape)
        frame = frame.permute(0, 2, 3, 1)
        yay = torch.tensor([255], dtype=torch.int)
        imgFrame = frame * yay
        plt.imshow(imgFrame[0].reshape(210, 160).detach().cpu().numpy().astype(np.uint8), cmap=plt.get_cmap('gray'))
        plt.show()
        imgpred = pred * yay
        plt.imshow(imgpred[0].reshape(210, 160).detach().cpu().numpy().astype(np.uint8), cmap=plt.get_cmap('gray'))
        plt.show()

        loss = 0
        for l in range(210):
            for w in range(160):
                x = np.square(np.subtract(frame[0, l, w].detach(), pred[0, l, w].detach())).mean()
                loss += x
        loss = loss / (210 * 160)
        print('Bild {} - Loss: {}'.format(i, loss))

