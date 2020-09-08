"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zum Training eines Auto-Encoders,
trainierter Auto-Encoder kann danach gespeichert werden.

Implementiert sind vier Auto-Encoder Architekturen.
AutoEncoder1 orientiert sich an:
Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631.
https://github.com/ctallec/world-models

Zur Orientierung diente das Tutorial https://www.kaggle.com/jagadeeshkotra/autoencoders-with-pytorch
Teile des Codes wurden aus dem Tutorial übernommen.
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
import statistics
import csv

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
Funktion lädt den Datensatz
"""
def load_dataset(path):
    return load(path)

"""
Funktion zeigt ein RGB-Bild
"""
def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor.cpu().numpy().astype(np.uint8))
    plt.show()

"""
Funktion zeigt ein Graustufen-Bild
"""
def show_torch_image_Grey(torch_tensor):
    plt.imshow(torch_tensor.cpu().numpy().astype(np.uint8), cmap = plt.get_cmap('gray'))
    plt.show()

"""
Funktion normalisiert einen Pixelwert
"""
def normalize(v):
    return v / 255


def reshape(dataset):
    nData = []
    for a in dataset:
        b = np.delete(a,0,0)
        nData.append(np.delete(b,0,0))
    return asarray(nData)

"""
Grundstruktur für den Trainingsplot/Validierungsplot
"""
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y, ):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


"""
Funktion trainiert einen Auto-Encoder mit RGB-Datensatz.
Der Name des Datensatzes muss übergeben werden.
Die Auto-Encoder-Architektur muss einmalig manuell im Code geändert werden.
Trainierter Auto-Encoder wird am Ende der Funktion gespeichert (Pfad muss manuell angepasst werden).
"""
def startAutoEncoder(dataset):
    global myDataYTrain
    global myDataXTrain
    global myDataYTest
    global myDataXTest

    #Überprüfung welcher Datensatz als Grundlage verwendet werden soll, anschließendes Laden des Datensatzes
    if (dataset == 'SmallDataset_SpaceInvaders'):
        split = True
        grey = False
        datasetTrain = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders/smallDatasetTraining1_SpaceInvaders.npy')
        datasetTrain2 = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders/smallDatasetTraining2_SpaceInvaders.npy')
        datasetTest = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders/smallDatasetTest_SpaceInvaders.npy')
        title = 'Auto-Encoder mit SmallDataset_SpaceInvaders'
    elif(dataset == 'SmallDataset_Q*Bert_Mixed'):
        split = False
        grey = False
        datasetTrain = load_dataset('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTraining.npy')
        datasetTest = load_dataset('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
        title = 'Auto-Encoder mit SmallDataset_Q*Bert_Mixed'
    elif(dataset == 'SmallDataset_JustQbert'):
        split = False
        grey = False
        datasetTrain = load_dataset('/home/annika/BA-Datensaetze/SmallDataset_JustQBert/smallDatasetTraining_JustQBert.npy')
        datasetTest = load_dataset('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
        title = 'Auto-Encoder mit SmallDataset_Q*Bert_JustQBert'

    #Bild aus dem Trainingsdatensatz laden, dient zur Kontrolle
    TrainingImage = Image.fromarray(datasetTrain[10])
    plt.imshow(TrainingImage)
    plt.show()

    #Bild aus dem Testdatensatz laden, dient zur Kontrolle
    TestImage = Image.fromarray(datasetTest[10])
    plt.imshow(TestImage)
    plt.show()

    #Aufbau Trainingsdatensatz
    print("Training Datensatz:")
    print(len(datasetTrain))
    print(datasetTrain[10].shape)

    #Aufbau Testdatensatz
    print("Test Datensatz:")
    print(len(datasetTest))
    print(datasetTest[10].shape)

    """
    Hier kann manuell die Auto-Encoder-Architektur gewählt werden 
    Zur Auswahl stehen nur die Auto-Encoder, die RGB-Bilder verarbeiten können.
    """
    #ae = AutoEncoder1()
    #ae = AutoEncoder2()
    #ae = AutoEncoder3()
    ae = AutoEncoder4()
    ae.to(torch.device("cuda:0"))
    print(ae)

    #definiert die Loss-Funktion und den Optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(ae.parameters(), lr=4e-4)

    #startet den Plot
    global plotter
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    #Aufteilung des Datensatzes in kleinere Sequenzen, kompletter Datensatz zu groß für direkte Verarbeitung
    if (split):
        iterationsTrain = ((len(datasetTrain) + len(datasetTrain2)) // 1000)
        firstIterationTrain = (len(datasetTrain) // 1000)
    else:
        iterationsTrain = (len(datasetTrain) // 1000)
        firstIterationTrain = iterationsTrain
    rest = False
    if (len(datasetTrain) % 1000 != 0):
        iterationsTrain += 1
        rest = True

    predictions = []

    #Wahl der Epochen (Wie oft der komplette Trainingsdatensatz für das Training verwendet werden soll)
    epochs = 2

    for e in range(epochs):
        for i in range(iterationsTrain):
            train_snippet = i + (e * iterationsTrain)
            losses = []
            startTrain = i * 1000
            stopTrain = ((i + 1) * 1000) - 1
            if(split):
                if (i + 1 < firstIterationTrain):
                    trainSetSnippet = datasetTrain[startTrain:stopTrain, :, :, :]
                else:
                    if (i + 1 == firstIterationTrain):
                        trainSetSnippet = datasetTrain[startTrain:, :, :, :]
                    else:
                        startTrain = (i - firstIterationTrain) * 1000
                        stopTrain = ((i - firstIterationTrain + 1) * 1000) - 1
                        if (i + 1 == iterationsTrain):
                            trainSetSnippet = datasetTrain2[startTrain:, :, :, :]
                        else:
                            trainSetSnippet = datasetTrain2[startTrain:stopTrain, :, :, :]
            else:
                if (i + 1 < firstIterationTrain):
                    trainSetSnippet = datasetTrain[startTrain:stopTrain, :, :, :]
                else:
                    trainSetSnippet = datasetTrain[startTrain:, :, :, :]

            trainSetSnippet = normalize(trainSetSnippet)
            trn_torch = torch.from_numpy(trainSetSnippet).type(torch.cuda.FloatTensor)
            trn_torch = trn_torch.permute(0, 3, 1, 2)
            trn_torch = trn_torch[:, :, :, :]
            trn = TensorDataset(trn_torch, trn_torch)
            trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=1, shuffle=False, num_workers=0)

            startTest = i * 430
            stopTest = ((i + 1) * 430) - 1
            if (i + 1 == iterationsTrain):
                testSetSnippet = datasetTest[startTest:, :, :, :]
            else:
                testSetSnippet = datasetTest[startTest:stopTest, :, :, :]

            testSetSnippet = normalize(testSetSnippet)
            test_torch = torch.from_numpy(testSetSnippet).type(torch.cuda.FloatTensor)
            test_torch = test_torch.permute(0, 3, 1, 2)
            test_torch = test_torch[:, :, :, :]
            test = TensorDataset(test_torch, test_torch)
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False, num_workers=0)

            for batch_idx, (data, target) in enumerate(trn_dataloader):

                data = torch.autograd.Variable(data)

                optimizer.zero_grad()

                pred = ae(data)

                loss = loss_func(pred, data)

                losses.append(loss.cpu().data.item())

                # Backpropagation
                loss.backward()

                optimizer.step()

                # Display
                if batch_idx % 25 == 1:
                    number = (((i + 1) * 1000))
                    if (i + 1 == iterationsTrain):
                        number = len(datasetTrain)

                    numberAll = number * (e + 1)
                    print('\r Images trained: {}/{} epochs: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i,
                        iterationsTrain,
                        e + 1,
                        epochs,
                        batch_idx * len(data),
                        len(trn_dataloader.dataset),
                        100. * batch_idx / len(trn_dataloader),
                        loss.cpu().data.item()),
                        end='')

            median_loss_train = statistics.median(losses)
            plotter.plot('loss', 'train', title, train_snippet + 1, median_loss_train)
            if (i == 0 and e == 0):
                myDataXTrain.append(0)
                myDataYTrain.append(losses[0])
            myDataXTrain.append(train_snippet + 1)
            myDataYTrain.append(median_loss_train)

            ae.eval()

            loss_func_val = nn.MSELoss()
            losses_val = []
            for batch_idx, (data, target) in enumerate(test_dataloader):

                data = torch.autograd.Variable(data)

                pred = ae(data)

                for prediction in pred:
                    predictions.append(prediction)

                loss_val = loss_func_val(pred, data)
                losses_val.append(loss_val.cpu().data.item())
            print('\ntestLossSum = {}'.format(loss_val.cpu().data.item()))
            median_loss_test = statistics.median(losses_val)
            plotter.plot('loss', 'validation', title, train_snippet +1 , median_loss_test)
            if (i == 0 and e == 0):
                myDataXTest.append(0)
                myDataYTest.append(losses_val[0])
            myDataXTest.append(train_snippet+1)
            myDataYTest.append(median_loss_test)


            if ((i == (iterationsTrain - 1)) and (e == 0 or e == (epochs - 1))):
                test_torch = test_torch.permute(0, 2, 3, 1)
                show_torch_image(
                    test_torch[2] * torch.tensor([255, 255, 255], dtype=torch.int, device=torch.device("cuda:0")))
                show_torch_image(predictions[2].permute(1, 2, 0).detach() * torch.tensor([255, 255, 255],
                                                                                         dtype=torch.int,
                                                                                         device=torch.device("cuda:0")))
                test_torch = test_torch.permute(0, 3, 1, 2)
            predictions = []

    global episode
    global evaluationsfolder
    pathEvaluation = evaluationsfolder + "/" + 'Episode{}/ae.pt'.format(episode)

    #Speichern des trainierten Auto-Encoders
    torch.save(ae.state_dict(), pathEvaluation)

"""
Funktion trainiert einen Auto-Encoder mit Graustufen-Datensatz.
Der Name des Datensatzes muss übergeben werden.
Die Auto-Encoder-Architektur muss einmalig manuell im Code geändert werden.
Trainierter Auto-Encoder wird am Ende der Funktion gespeichert (Pfad muss manuell angepasst werden).
"""
def startAutoEncoderGrey(dataset):
    global myDataYTrain
    global myDataXTrain
    global myDataYTest
    global myDataXTest

    # Überprüfung welcher Datensatz als Grundlage verwendet werden soll, anschließendes Laden des Datensatzes
    if (dataset == 'SmallDataset_Q*Bert_Mixed_Greyscale'):
        split = False
        datasetTrain = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTraining_Q*Bert_Mixed_Greyscale.npy')
        datasetTest = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy')
        title = 'Auto-Encoder mit SmallDatasetTest_Q*Bert_Mixed_Greyscale'

    elif (dataset == 'SmallDataset_SpaceInvaders_Greyscale'):
        split = True
        datasetTrain = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTraining1_SpaceInvaders_Greyscale.npy')
        datasetTrain2 = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTraining2_SpaceInvaders_Greyscale.npy')
        datasetTest = load_dataset(
            '/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTest_SpaceInvaders_Greyscale.npy')
        title = 'Auto-Encoder mit SmallDataset_SpaceInvaders_Greyscale'


    # Bild aus dem Trainingsdatensatz laden, dient zur Kontrolle
    TrainingImage = Image.fromarray(datasetTrain[10])
    plt.imshow(TrainingImage,cmap = plt.get_cmap('gray'))
    plt.show()

    # Bild aus dem Testdatensatz laden, dient zur Kontrolle
    TestImage = Image.fromarray(datasetTest[10])
    plt.imshow(TestImage,cmap = plt.get_cmap('gray'))
    plt.show()

    """
    Hier kann manuell die Auto-Encoder-Architektur gewählt werden 
    Zur Auswahl stehen nur die Auto-Encoder, die Graustufen-Bilder verarbeiten können.
    """
    #ae = AutoEncoder1Grey()
    #ae = AutoEncoder2Grey()
    #ae = AutoEncoder3Grey()
    ae = AutoEncoder4Grey()
    ae.to(torch.device("cuda:0"))
    print(ae)

    # definiert die Loss-Funktion und den Optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(ae.parameters(), lr=4e-4)

    # startet den Plot
    global plotter
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    # Aufteilung des Datensatzes in kleinere Sequenzen, kompletter Datensatz zu groß für direkte Verarbeitung
    if (split):
        iterationsTrain = ((len(datasetTrain) + len(datasetTrain2)) // 1000)
        firstIterationTrain = (len(datasetTrain) // 1000)
    else:
        iterationsTrain = (len(datasetTrain) // 1000)
        firstIterationTrain = iterationsTrain
    rest = False
    if (len(datasetTrain) % 1000 != 0):
        iterationsTrain += 1
        rest = True

    predictions = []

    # Wahl der Epochen (Wie oft der komplette Trainingsdatensatz für das Training verwendet werden soll)
    epochs = 4

    for e in range(epochs):
        for i in range(iterationsTrain):
            train_snippet = i + (e * iterationsTrain)
            losses = []
            startTrain = i * 1000
            stopTrain = ((i + 1) * 1000) - 1
            if (split):
                if (i + 1 < firstIterationTrain):
                    trainSetSnippet = datasetTrain[startTrain:stopTrain, :, :]
                else:
                    if (i + 1 == firstIterationTrain):
                        trainSetSnippet = datasetTrain[startTrain:, :, :]
                    else:
                        startTrain = (i - firstIterationTrain) * 1000
                        stopTrain = ((i - firstIterationTrain + 1) * 1000) - 1
                        if (i + 1 == iterationsTrain):
                            trainSetSnippet = datasetTrain2[startTrain:, :, :]
                        else:
                            trainSetSnippet = datasetTrain2[startTrain:stopTrain, :, :]
            else:
                if (i + 1 < firstIterationTrain):
                    trainSetSnippet = datasetTrain[startTrain:stopTrain, :, :]
                else:
                    trainSetSnippet = datasetTrain[startTrain:, :, :]

            trainSetSnippet = trainSetSnippet.reshape(len(trainSetSnippet),210,160,1)
            trainSetSnippet = normalize(trainSetSnippet)
            trn_torch = torch.from_numpy(trainSetSnippet).type(torch.cuda.FloatTensor)
            trn_torch = trn_torch.permute(0, 3, 1, 2)
            trn_torch = trn_torch[:, :, :, :]
            trn = TensorDataset(trn_torch, trn_torch)
            trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=1, shuffle=False, num_workers=0)

            startTest = i * 430
            stopTest = ((i + 1) * 430) - 1
            if (i + 1 == iterationsTrain):
                testSetSnippet = datasetTest[startTest:, :, :]
            else:
                testSetSnippet = datasetTest[startTest:stopTest, :, :]

            testSetSnippet = testSetSnippet.reshape(len(testSetSnippet), 210, 160, 1)
            testSetSnippet = normalize(testSetSnippet)
            test_torch = torch.from_numpy(testSetSnippet).type(torch.cuda.FloatTensor)
            test_torch = test_torch.permute(0, 3, 1, 2)
            test_torch = test_torch[:, :, :, :]
            test = TensorDataset(test_torch, test_torch)
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False, num_workers=0)

            for batch_idx, (data, target) in enumerate(trn_dataloader):

                data = torch.autograd.Variable(data)

                optimizer.zero_grad()

                pred = ae(data)

                loss = loss_func(pred, data)

                losses.append(loss.cpu().data.item())

                # Backpropagation
                loss.backward()

                optimizer.step()

                # Display
                if batch_idx % 25 == 1:
                    number = (((i + 1) * 1000))
                    if (i + 1 == iterationsTrain):
                        number = len(datasetTrain)

                    numberAll = number * (e + 1)
                    print('\r Images trained: {}/{} epochs: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i,
                        iterationsTrain,
                        e + 1,
                        epochs,
                        batch_idx * len(data),
                        len(trn_dataloader.dataset),
                        100. * batch_idx / len(trn_dataloader),
                        loss.cpu().data.item()),
                        end='')

            median_loss_train = statistics.median(losses)
            plotter.plot('loss', 'train', title, train_snippet + 1, median_loss_train)
            if (i == 0 and e == 0):
                myDataXTrain.append(0)
                myDataYTrain.append(losses[0])
            myDataXTrain.append(train_snippet + 1)
            myDataYTrain.append(median_loss_train)

            ae.eval()

            loss_func_val = nn.MSELoss()
            losses_val = []
            for batch_idx, (data, target) in enumerate(test_dataloader):

                data = torch.autograd.Variable(data)

                pred = ae(data)

                for prediction in pred:
                    predictions.append(prediction)

                loss_val = loss_func_val(pred, data)
                losses_val.append(loss_val.cpu().data.item())
            print('\ntestLossSum = {}'.format(loss_val.cpu().data.item()))
            median_loss_test = statistics.median(losses_val)
            plotter.plot('loss', 'validation', title, train_snippet+1, median_loss_test)
            if (i == 0 and e == 0):
                myDataXTest.append(0)
                myDataYTest.append(losses_val[0])
            myDataXTest.append(train_snippet+1)
            myDataYTest.append(median_loss_test)
            if ((i == (iterationsTrain - 1)) and (e == 0 or e == (epochs - 1))):
                test_torch = test_torch.permute(0, 2, 3, 1)
                yay = torch.tensor([255], dtype=torch.int, device=torch.device("cuda:0"))
                testImg = test_torch[2] * yay
                show_torch_image_Grey(testImg.reshape(210,160))

                predImg = predictions[2].permute(1, 2, 0).detach() * torch.tensor([255], dtype=torch.int,device=torch.device("cuda:0"))
                show_torch_image_Grey(predImg.reshape(210,160))
                test_torch = test_torch.permute(0, 3, 1, 2)
            predictions = []

    global episode
    global evaluationsfolder
    pathEvaluation = evaluationsfolder + "/" + 'Episode{}/ae.pt'.format(episode)

    #Speichern des trainierten Auto-Encoders
    torch.save(ae.state_dict(), pathEvaluation)

if __name__ == '__main__':

    global myDataYTrain
    global myDataXTrain
    global myDataYTest
    global myDataXTest
    myDataYTrain = []
    myDataXTrain = []
    myDataYTest = []
    myDataXTest = []
    myDataYTrain.append(0)
    myDataXTrain.append(0)
    myDataYTest.append(0)
    myDataXTest.append(0)
    """
    Auswahl an Datensätzen:
    
    SmallDataset_JustQbert
    SmallDataset_Q*Bert_Mixed
    SmallDataset_Q*Bert_Mixed_Greyscale
    SmallDataset_SpaceInvaders (Training Split in two seperate Arrays)
    SmallDataset_SpaceInvaders_Greyscale (Training Split in two seperate Arrays)
    
    """
    datasetChoice = 'SmallDataset_JustQbert'
    global evaluationsfolder
    evaluationsfolder = '/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_RGB_4/Q*Bert_JustQBert'

    if (datasetChoice == 'SmallDataset_SpaceInvaders'):
        grey = False
    elif(datasetChoice == 'SmallDataset_Q*Bert_Mixed'):
        grey = False
    elif(datasetChoice == 'SmallDataset_JustQbert'):
        grey = False
    elif(datasetChoice == 'SmallDataset_Q*Bert_Mixed_Greyscale'):
        grey = True
    elif (datasetChoice == 'SmallDataset_SpaceInvaders_Greyscale'):
        grey = True

    global episode
    episode = 1

    #Auto-Encoder-Training speichern trainierten Auto-Encoder speichern. Verfahren wird 3x ausgeführt.
    for i in range(3):
        if (grey):
            startAutoEncoderGrey(datasetChoice)
        else:
            startAutoEncoder(datasetChoice)

        myData = np.array([myDataXTrain, myDataYTrain, myDataYTest]).transpose()

        pathEvaluation = evaluationsfolder + "/" + 'Episode{}/LossAndVal.csv'.format(episode)
        myFile = open(pathEvaluation, 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(myData)

        episode = episode + 1
        myDataYTrain = []
        myDataXTrain = []
        myDataYTest = []
        myDataXTest = []