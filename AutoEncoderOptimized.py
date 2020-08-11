"""
Ein AutoEncoder mit ...
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

class AutoEncoderWM(nn.Module):


    def __init__(self):
        super(AutoEncoderWM, self).__init__()

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
class AutoEncoderWMGrey(nn.Module):


    def __init__(self):
        super(AutoEncoderWMGrey, self).__init__()

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
class AutoEncoderMF(nn.Module):


    def __init__(self):
        super(AutoEncoderMF, self).__init__()

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
class AutoEncoderMFGrey(nn.Module):


    def __init__(self):
        super(AutoEncoderMFGrey, self).__init__()

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
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        #print(x.size(1))
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        #print(x.shape)
        #print(x.shape)
        x = x.view(x.size(0), originalC, originalH, originalW)
        x = F.relu(self.deconv1(x))
        #print(x.shape)
        x = F.relu(self.deconv2(x))
        return x
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

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

        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x,indices1 = self.maxpool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x,indices2 = self.maxpool2(x)
        #print(x.shape)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        #print(x.size(1))
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        #print(x.shape)
        #print(x.shape)
        x = x.view(x.size(0), originalC, originalH, originalW)
        #print(x.shape)
        x = self.unmaxpool1(x,indices2)
        #print(x.shape)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        #print(x.shape)
        x = self.unmaxpool2(x,indices1, output_size=torch.Size([x.size(0),x.size(1),210,160]))
        #print(x.shape)
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        #print(x.shape)
        return x
class AutoEncoderGrey(nn.Module):
    def __init__(self):
        super(AutoEncoderGrey, self).__init__()
        """
        # encoder
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=8, stride=4)
        """
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=(2,2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=7, stride=1, padding=(3,3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)

        # Latent View
        """
        self.lv1 = nn.Linear(7680, 200)
        self.lv2 = nn.Linear(200, 80)
        self.lv3 = nn.Linear(80, 80)
        self.lv4 = nn.Linear(80, 7680)

        """
        self.lv1 = nn.Linear(5200, 400)
        self.lv2 = nn.Linear(400, 30)
        self.lv3 = nn.Linear(30, 400)
        self.lv4 = nn.Linear(400, 5200)
        """
        #Decoder
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=8, stride=4, output_padding=(1,0))
        self.deconv2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=3,  output_padding=(1,2))

        """
        self.unmaxpool1 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv1 = nn.ConvTranspose2d(40, 20, kernel_size=7, stride=1)
        self.unmaxpool2 = nn.MaxUnpool2d(kernel_size=4, stride=4)
        self.deconv2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=1)



    def forward(self, x):
        """
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        #print(x.size(1))
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        #print(x.shape)
        #print(x.shape)
        x = x.view(x.size(0), originalC, originalH, originalW)

        x = F.relu(self.deconv1(x))
        #print(x.shape)
        x = F.relu(self.deconv2(x))
        return x
        """
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x,indices1 = self.maxpool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x,indices2 = self.maxpool2(x)
        #print(x.shape)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        #print(x.size(1))
        x = torch.sigmoid(self.lv1(x))
        x = torch.sigmoid(self.lv2(x))
        x = torch.sigmoid(self.lv3(x))
        x = torch.sigmoid(self.lv4(x))
        #print(x.shape)
        #print(x.shape)
        x = x.view(x.size(0), originalC, originalH, originalW)
        #print(x.shape)
        x = self.unmaxpool1(x,indices2)
        #print(x.shape)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        #print(x.shape)
        x = self.unmaxpool2(x,indices1, output_size=torch.Size([x.size(0),x.size(1),210,160]))
        #print(x.shape)
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        #print(x.shape)
        return x
class AutoEncoderVAE(nn.Module):

    def __init__(self):
        super(AutoEncoderVAE, self).__init__()

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
        #print("start:")
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x, indices1 = self.maxpool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x, indices2 = self.maxpool2(x)
        #print(x.shape)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        #print(x.size(1))
        x = torch.sigmoid(self.lv1(x))
        #print(x.size(1))
        x = torch.sigmoid(self.lv2(x))
        #print(x.size(1))
        mu = self.fc_mu(x)
        #print("mu:")
        #print(mu.size(1))
        logsigma = self.fc_logsigma(x)
        #print("logsigma")
        #print(logsigma.size(1))
        sigma = logsigma.exp()
        #print("sigma:")
        #print(sigma.shape)
        eps = torch.randn_like(sigma)
        #print("eps:")
        #print(eps.shape)
        x = eps.mul(sigma).add_(mu)
        #print(x.size(1))
        x = torch.sigmoid(self.lv3(x))
        #print(x.size(1))
        x = torch.sigmoid(self.lv4(x))
        #print(x.shape)
        x = x.view(x.size(0), originalC, originalH, originalW)
        #print(x.shape)
        x = self.unmaxpool1(x, indices2)
        #print(x.shape)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        #print(x.shape)
        x = self.unmaxpool2(x, indices1, output_size=torch.Size([x.size(0), x.size(1), 210, 160]))
        #print(x.shape)
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        #print(x.shape)
        return x
class AutoEncoderVAEGrey(nn.Module):
    def __init__(self):
        super(AutoEncoderVAEGrey, self).__init__()

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
        #print("start:")
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x, indices1 = self.maxpool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x, indices2 = self.maxpool2(x)
        #print(x.shape)
        originalC = x.size(1)
        originalH = x.size(2)
        originalW = x.size(3)
        x = x.view(x.size(0), -1)
        #print(x.size(1))
        x = torch.sigmoid(self.lv1(x))
        #print(x.size(1))
        x = torch.sigmoid(self.lv2(x))
        #print(x.size(1))
        mu = self.fc_mu(x)
        #print("mu:")
        #print(mu.size(1))
        logsigma = self.fc_logsigma(x)
        #print("logsigma")
        #print(logsigma.size(1))
        sigma = logsigma.exp()
        #print("sigma:")
        #print(sigma.shape)
        eps = torch.randn_like(sigma)
        #print("eps:")
        #print(eps.shape)
        x = eps.mul(sigma).add_(mu)
        #print(x.size(1))
        x = torch.sigmoid(self.lv3(x))
        #print(x.size(1))
        x = torch.sigmoid(self.lv4(x))
        #print(x.shape)
        x = x.view(x.size(0), originalC, originalH, originalW)
        #print(x.shape)
        x = self.unmaxpool1(x, indices2)
        #print(x.shape)
        x = F.relu(self.deconv1(x))
        x = x[:, :, :52, :40]
        #print(x.shape)
        x = self.unmaxpool2(x, indices1, output_size=torch.Size([x.size(0), x.size(1), 210, 160]))
        #print(x.shape)
        x = F.relu(self.deconv2(x))
        x = x[:, :, :210, :160]
        #print(x.shape)
        return x



def load_dataset(path):
    return load(path)

def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor.cpu().numpy().astype(np.uint8))
    plt.show()

def show_torch_image_Grey(torch_tensor):
    plt.imshow(torch_tensor.cpu().numpy().astype(np.uint8), cmap = plt.get_cmap('gray'))
    plt.show()
def normalize(v):
    return v / 255


def reshape(dataset):
    nData = []
    for a in dataset:
        b = np.delete(a,0,0)
        nData.append(np.delete(b,0,0))
    return asarray(nData)

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



def startAutoEncoder(dataset):
    global myDataYTrain
    global myDataXTrain
    global myDataYTest
    global myDataXTest
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

    TrainingImage = Image.fromarray(datasetTrain[10])
    # firstImage.show()
    plt.imshow(TrainingImage)
    plt.show()

    TestImage = Image.fromarray(datasetTest[10])
    # firstImage.show()
    plt.imshow(TestImage)
    plt.show()
    print("Training Datensatz:")
    print(len(datasetTrain))
    print(datasetTrain[10].shape)

    print("Test Datensatz:")
    print(len(datasetTest))
    print(datasetTest[10].shape)

    #ae = AutoEncoderWM()
    ae = AutoEncoderMF()
    #ae = AutoEncoder()
    #ae = AutoEncoderVAE()
    ae.to(torch.device("cuda:0"))
    print(ae)

    # define our optimizer and loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(ae.parameters(), lr=4e-4)

    # losses = []

    global plotter
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

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

    epochs = 4

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

            # last_loss = 1
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

            # if(loss.cpu().data.item() <= last_loss):
            #    last_loss = loss.cpu().data.item()
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

                # * torch.tensor([255,255,255])
                show_torch_image(predictions[2].permute(1, 2, 0).detach() * torch.tensor([255, 255, 255],
                                                                                         dtype=torch.int,
                                                                                         device=torch.device("cuda:0")))
                test_torch = test_torch.permute(0, 3, 1, 2)
            predictions = []

    global episode
    global evaluationsfolder
    pathEvaluation = evaluationsfolder + "/" + 'Episode{}/ae.pt'.format(episode)
    torch.save(ae.state_dict(), pathEvaluation)

def startAutoEncoderGrey(dataset):
    global myDataYTrain
    global myDataXTrain
    global myDataYTest
    global myDataXTest
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



    TrainingImage = Image.fromarray(datasetTrain[10])
    # firstImage.show()
    plt.imshow(TrainingImage,cmap = plt.get_cmap('gray'))
    plt.show()

    TestImage = Image.fromarray(datasetTest[10])
    # firstImage.show()
    plt.imshow(TestImage,cmap = plt.get_cmap('gray'))
    plt.show()

    ae = AutoEncoderWMGrey()
    #ae = AutoEncoderMFGrey()
    #ae = AutoEncoderGrey()
    #ae = AutoEncoderVAEGrey()
    ae.to(torch.device("cuda:0"))
    print(ae)

    # define our optimizer and loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(ae.parameters(), lr=4e-4)

    # losses = []

    global plotter
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

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
            #print(trainSetSnippet.shape)
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
            #print(testSetSnippet.shape)
            testSetSnippet = normalize(testSetSnippet)
            test_torch = torch.from_numpy(testSetSnippet).type(torch.cuda.FloatTensor)
            test_torch = test_torch.permute(0, 3, 1, 2)
            test_torch = test_torch[:, :, :, :]
            test = TensorDataset(test_torch, test_torch)
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False, num_workers=0)

            # last_loss = 1
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

            # if(loss.cpu().data.item() <= last_loss):
            #    last_loss = loss.cpu().data.item()
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

                # * torch.tensor([255,255,255])
                predImg = predictions[2].permute(1, 2, 0).detach() * torch.tensor([255], dtype=torch.int,device=torch.device("cuda:0"))
                show_torch_image_Grey(predImg.reshape(210,160))
                test_torch = test_torch.permute(0, 3, 1, 2)
            predictions = []

    global episode
    global evaluationsfolder
    pathEvaluation = evaluationsfolder + "/" + 'Episode{}/ae.pt'.format(episode)
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
    Possibilities for Datasets:
    
    SmallDataset_JustQbert
    SmallDataset_Q*Bert_Mixed
    SmallDataset_Q*Bert_Mixed_Greyscale
    SmallDataset_SpaceInvaders (Training Split in two seperate Arrays)
    SmallDataset_SpaceInvaders_Greyscale (Training Split in two seperate Arrays)
    
    """
    datasetChoice = 'SmallDataset_Q*Bert_Mixed'
    global evaluationsfolder
    evaluationsfolder = '/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_GS_1/Q*Bert_Mixed'

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