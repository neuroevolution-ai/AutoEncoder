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

def show_torch_image(torch_tensor):
    plt.imshow(torch_tensor)
    plt.show()

def normalize(v):
    return v / 255

def getFrame(dataset, index):
    frame = normalize(dataset[index:index+1, :, :, :])
    torchFrame = torch.from_numpy(frame).type(torch.FloatTensor)
    torchFrame = torchFrame.permute(0, 3, 1, 2)
    return torchFrame[:, :, :, :]

def getFrameGrey(dataset, index):
    dataset = dataset.reshape(len(dataset), 210, 160, 1)
    frame = normalize(dataset[index:index + 1, :, :, :])
    torchFrame = torch.from_numpy(frame).type(torch.FloatTensor)
    torchFrame = torchFrame.permute(0, 3, 1, 2)
    return torchFrame[:, :, :, :]

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



if __name__ == '__main__':
    ae = AutoEncoderWMGrey()
    #ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_RGB_1/Q*Bert_JustQBert/!Episode1/ae.pt'))
    #ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_RGB_1/Q*Bert_Mixed/!!Episode1/ae.pt'))
    #ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_RGB_1/SpaceInvaders/!Episode2/ae.pt'))
    ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_GS_1/Q*Bert_Mixed/!Episode1/ae.pt'))
    #ae.load_state_dict(torch.load('/home/annika/Bachelorarbeit/Evaluationsergebnisse/AutoEncoder_GS_1/SpaceInvaders/!Episode3/ae.pt'))
    ae.eval()

    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_JustQBert/smallDatasetTest_JustQBert.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders/smallDatasetTest_SpaceInvaders.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy')
    #datasetTest = load('/home/annika/BA-Datensaetze/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTest_SpaceInvaders_Greyscale.npy')

    """RGB
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
    """Grey"""

    [210,160]
    [210,160,3]

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
    start = 21
    count = 1
    For RGB Images
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
    """For Greyscale Images
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
    """
    """
    JustQBert: 41
    MixedQBert: 21
    SpaceInvaders: 3
    QBertGreyscale: 16
    SpaceInvadersGrey: 10
    """
