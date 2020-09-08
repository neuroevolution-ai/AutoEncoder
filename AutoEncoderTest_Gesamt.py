"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zum Testen eines Auto-Encoders.
Ein trainierter Auto-Encoder kann mit einem Testdatensatz geprüft werden.
Ermittelt wird der Median Loss, beste Loss und schlechteste Loss im gesamten Testdatensatz.

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
import statistics



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


def normalize(v):
    return v / 255

"""
Funktion führt den gesamten Testdurchlauf eines Auto-Encoders durch.
Funktioniert nur mit einem Auto-Encoder der RGB-Datensätze verarbeiten kann.
"""
def startAutoEncoder(dataset):
    global losses
    # Überprüfung welcher Datensatz als Grundlage verwendet werden soll, anschließendes Laden des Datensatzes
    if (dataset == 'SmallDataset_SpaceInvaders'):
        datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders/smallDatasetTest_SpaceInvaders.npy')
    elif (dataset == 'SmallDataset_Q*Bert_Mixed'):
        datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
    elif (dataset == 'SmallDataset_JustQbert'):
        datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')


    #Bild aus dem Datensatz laden, dient zur Kontrolle
    TestImage = Image.fromarray(datasetTest[10])
    plt.imshow(TestImage)
    plt.show()

    # Aufbau Datensatz
    print("Test Datensatz:")
    print(len(datasetTest))
    print(datasetTest[10].shape)

    ae.to(torch.device("cuda:0"))
    print(ae)

    # definiert die Loss-Funktion
    loss_func = nn.MSELoss()

    # Aufteilung des Datensatzes in kleinere Sequenzen, kompletter Datensatz zu groß für direkte Verarbeitung
    iterationsTest = len(datasetTest) // 1000

    # Testdurchlauf und Ermittlung der Losswerte
    for i in range(iterationsTest):
        startTest = i * 1000
        stopTest = ((i + 1) * 1000) - 1
        if (i + 1 < iterationsTest):
            testSetSnippet = datasetTest[startTest:stopTest, :, :]
        else:
            testSetSnippet = datasetTest[startTest:, :, :]

        testSetSnippet = normalize(testSetSnippet)
        test_torch = torch.from_numpy(testSetSnippet).type(torch.cuda.FloatTensor)
        test_torch = test_torch.permute(0, 3, 1, 2)
        test_torch = test_torch[:, :, :, :]
        test = TensorDataset(test_torch, test_torch)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

        ae.eval()
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data = torch.autograd.Variable(data)

            pred = ae(data)

            loss_val = loss_func(pred, data)
            losses.append(loss_val.cpu().data.item())
        print('Testsequenz {} beendet'.format(i))

"""
Funktion führt den gesamten Testdurchlauf eines Auto-Encoders durch.
Funktioniert nur mit einem Auto-Encoder der Graustufen-Datensätze verarbeiten kann.
"""
def startAutoEncoderGrey(dataset):
    global losses
    global ae
    # Überprüfung welcher Datensatz als Grundlage verwendet werden soll, anschließendes Laden des Datensatzes
    if (dataset == 'SmallDataset_Q*Bert_Mixed_Greyscale'):
        datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy')

    elif (dataset == 'SmallDataset_SpaceInvaders_Greyscale'):
        datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTest_SpaceInvaders_Greyscale.npy')

    # Bild aus dem Datensatz laden, dient zur Kontrolle
    TestImage = Image.fromarray(datasetTest[10])
    plt.imshow(TestImage, cmap=plt.get_cmap('gray'))
    plt.show()

    ae.to(torch.device("cuda:0"))
    print(ae)

    # definiert die Loss-Funktion
    loss_func = nn.MSELoss()

    # Aufteilung des Datensatzes in kleinere Sequenzen, kompletter Datensatz zu groß für direkte Verarbeitung
    iterationsTest =len(datasetTest) // 1000
    if (len(datasetTest) % 1000 != 0):
        iterationsTest += 1

    # Testdurchlauf und Ermittlung der Losswerte
    for i in range(iterationsTest):
        startTest = i * 1000
        stopTest = ((i + 1) * 1000) - 1
        if (i + 1 < iterationsTest):
            testSetSnippet = datasetTest[startTest:stopTest, :, :]
        else:
            testSetSnippet = datasetTest[startTest:, :, :]

        testSetSnippet = testSetSnippet.reshape(len(testSetSnippet), 210, 160, 1)
        testSetSnippet = normalize(testSetSnippet)
        test_torch = torch.from_numpy(testSetSnippet).type(torch.cuda.FloatTensor)
        test_torch = test_torch.permute(0, 3, 1, 2)
        test_torch = test_torch[:, :, :, :]
        test = TensorDataset(test_torch, test_torch)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

        ae.eval()
        for batch_idx, (data, target) in enumerate(test_dataloader):

            data = torch.autograd.Variable(data)

            pred = ae(data)

            loss_val = loss_func(pred, data)
            losses.append(loss_val.cpu().data.item())
        print('Testsequenz {} beendet'.format(i))


if __name__ == '__main__':

    global losses
    losses = []
    global ae

    """
    Auswahl an Datensätzen:

    SmallDataset_JustQbert
    SmallDataset_Q*Bert_Mixed
    SmallDataset_Q*Bert_Mixed_Greyscale
    SmallDataset_SpaceInvaders (Training Split in two seperate Arrays)
    SmallDataset_SpaceInvaders_Greyscale (Training Split in two seperate Arrays)

    """
    datasetChoice = 'SmallDataset_SpaceInvaders'

    if (datasetChoice == 'SmallDataset_SpaceInvaders'):
        grey = False
    elif (datasetChoice == 'SmallDataset_Q*Bert_Mixed'):
        grey = False
    elif (datasetChoice == 'SmallDataset_JustQbert'):
        grey = False
    elif (datasetChoice == 'SmallDataset_Q*Bert_Mixed_Greyscale'):
        grey = True
    elif (datasetChoice == 'SmallDataset_SpaceInvaders_Greyscale'):
        grey = True


    # Hier kann manuell die Auto-Encoder-Architektur gewählt werden
    #ae = AutoEncoder1()
    #ae = AutoEncoder1Grey()
    #ae = AutoEncoder2()
    #ae = AutoEncoder2Grey()
    #ae = AutoEncoder3()
    #ae = AutoEncoder3Grey()
    ae = AutoEncoder4()
    #ae = AutoEncoder4Grey()

    # Hier wird der trainierte Auto-Encoder geladen
    ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_RGB_4/SpaceInvaders/ae.pt'))

    if (grey):
        startAutoEncoderGrey(datasetChoice)
    else:
        startAutoEncoder(datasetChoice)

    # Ausgabe der Losswerte
    print(losses)
    print(len(losses))
    print('Median Loss: {}'.format(statistics.median(losses)))
    print('Worst Loss: {}'.format(max(losses)))
    print('Best Loss: {}'.format(min(losses)))

