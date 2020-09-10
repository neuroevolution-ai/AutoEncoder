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
import sys
import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from numpy import asarray, os, load
import cv2

#Auto-Encoder importieren
import AutoEncoder1
import AutoEncoder1Grey
import AutoEncoder2
import AutoEncoder2Grey
import AutoEncoder3
import AutoEncoder3Grey
import AutoEncoder4
import AutoEncoder4Grey

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

    autoencoder = input("\nWelche Auto-Encoder-Architektur soll gewählt werden?\n\n"
                 "Geben Sie das Kürzel hinter der gewünschten Architektur an.\n"
                 "Auto-Encoder 1: AE1\n"
                 "Auto-Encoder 1: AE2\n"
                 "Auto-Encoder 1: AE3\n"
                 "Auto-Encoder 1: AE4\n"
                 "Ihre Wahl ist: ")
    if(not(autoencoder == 'AE1' or autoencoder == 'AE2' or autoencoder == 'AE3' or autoencoder == 'AE4')):
        print("\n\nSie haben keine gültige Architektur gewählt. Das Programm bricht ab.")
        sys.exit()
    print('\nSie haben die Architektur {} gewählt.'.format(autoencoder))
    datasetChoice = input("\n\nMit welchem Datensatz soll der Auto-Encoder getestet werden?\n\n"
                        "Geben Sie das Kürzel hinter dem gewünschten Datensatz an.\n"
                        "Q*Bert - RGB JustQbert: 1\n"
                        "Q*Bert - RGB Mixed: 2\n"
                        "Q*bert - Graustufen: 3\n"
                        "SPACE INVADERS - RGB: 4\n"
                        "SPACE INVADERS - Graustufen: 5\n"
                        "Ihre Wahl ist: ")
    if (not (datasetChoice == '1' or datasetChoice == '2' or datasetChoice == '3' or datasetChoice == '4' or datasetChoice == '5')):
        print("\n\nSie haben keinen gültigen Datensatz ausgewählt. Das Programm bricht ab.")
        sys.exit()

    if (datasetChoice == '1'):
        grey = False
        autoencoder = autoencoder + "RGB"
        dataset = 'SmallDataset_JustQbert'
    elif(datasetChoice == '2'):
        grey = False
        autoencoder = autoencoder + "RGB"
        dataset = 'SmallDataset_Q*Bert_Mixed'
    elif(datasetChoice == '3'):
        grey = True
        autoencoder = autoencoder + "GS"
        dataset = 'SmallDataset_Q*Bert_Mixed_Greyscale'
    elif(datasetChoice == '4'):
        grey = False
        autoencoder = autoencoder + "RGB"
        dataset = 'SmallDataset_SpaceInvaders'
    elif (datasetChoice == '5'):
        grey = True
        autoencoder = autoencoder + "GS"
        dataset = 'SmallDataset_SpaceInvaders_Greyscale'

    path = input("\n\nGeben Sie bitte den Pfad der trainierten Auto-Encoder Datei mit dem Namen ae.pt ein: ")
    if not os.path.exists(path):
        print("Dieser Pfad existiert nicht. Das Programm wird beendet")
        sys.exit()

    if (grey):
        if (autoencoder == 'AE1GS'):
            ae = AutoEncoder1Grey.AutoEncoder1Grey()
        elif (autoencoder == 'AE2GS'):
            ae = AutoEncoder2Grey.AutoEncoder2Grey()
        elif (autoencoder == 'AE3GS'):
            ae = AutoEncoder3Grey.AutoEncoder3Grey()
        elif (autoencoder == 'AE4GS'):
            ae = AutoEncoder4Grey.AutoEncoder4Grey()
        else:
            print('Der Auto-Encoder {} kann nicht mit dem Datensatz {} trainiert werden.'.format(autoencoder, dataset))
    else:
        if (autoencoder == 'AE1RGB'):
            ae = AutoEncoder1.AutoEncoder1()
        elif (autoencoder == 'AE2RGB'):
            ae = AutoEncoder2.AutoEncoder2()
        elif (autoencoder == 'AE3RGB'):
            ae = AutoEncoder3.AutoEncoder3()
        elif (autoencoder == 'AE4RGB'):
            ae = AutoEncoder4.AutoEncoder4()
        else:
            print('Der Auto-Encoder {} kann nicht mit dem Datensatz {} trainiert werden.'.format(autoencoder, dataset))

    # Hier wird der trainierte Auto-Encoder geladen
    ae.load_state_dict(torch.load(path))
    ae.eval()

    # Überprüfung welcher Datensatz als Grundlage verwendet werden soll, anschließendes Laden des Datensatzes
    if (dataset == 'SmallDataset_SpaceInvaders'):
        datasetTest = load('datasets/SmallDataset_SpaceInvaders/smallDatasetTest_SpaceInvaders.npy')
    elif (dataset == 'SmallDataset_Q*Bert_Mixed'):
        datasetTest = load('datasets/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
    elif (dataset == 'SmallDataset_JustQbert'):
        datasetTest = load('datasets/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
    elif (dataset == 'SmallDataset_Q*Bert_Mixed_Greyscale'):
        datasetTest = load('datasets/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy')
    elif (dataset == 'SmallDataset_SpaceInvaders_Greyscale'):
        datasetTest = load('datasets/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTest_SpaceInvaders_Greyscale.npy')

    answer = input("\n\nMöchten Sie ein einzelnes Bild laden? Antwortmöglichkeiten ja und nein: ")
    if (not(answer == 'ja' or answer == 'nein')):
        print("Keine erlaubte Antwort. Das Programm wird beendet.")
        sys.exit()

    if(answer == 'ja'):
        imgpath = input("\n\nGeben Sie bitte den Pfad des Bildes ein: ")
        if not os.path.exists(path):
            print("Dieser Pfad existiert nicht. Das Programm wird beendet")
            sys.exit()

        answerGrey = input("\n\nIst das Bild in einer Graustufenrepräsentation? Antwortmöglichkeiten ja und nein: ")
        if (not (answerGrey == 'ja' or answerGrey == 'nein')):
            print("Keine erlaubte Antwort. Das Programm wird beendet.")
            sys.exit()

        if(answerGrey == 'ja'):
            """
            Hier wird manuell ein Graustufen-Bild in den Auto-Encoder gegeben und der Loss ermittelt.
            Eingabe- und Ausgabe-Bild werden nach Ausführung angezeigt.
            """

            imgArray = []
            img = cv2.imread(path)
            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgArray.append(asarray(imgGrey))
            print(len(imgArray))
            print(imgArray[0].shape)
            frame = getFrameGrey(np.array(imgArray), 0)
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

        else:
            """
            Hier wird manuell ein RGB-Bild in den Auto-Encoder gegeben und der Loss ermittelt.
            Eingabe- und Ausgabe-Bild werden nach Ausführung angezeigt.
            """
            imgArray = []
            img = Image.open(imgpath)
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
    else:
        startS = input("\n\nGeben Sie den Startindex an: ")
        if (not (str.isnumeric(startS))):
            print("Keine gültige Zahl.")
            sys.exit()
        countS = input("\n\nGeben Sie die Anzahl an Bildern an: ")
        if (not (str.isnumeric(countS))):
            print("Keine gültige Zahl.")
            sys.exit()

    """
    start: ist der Indexstart im Datensatz
    count: ist die Anzahl an Bildern, die ab dem definierten Index in den Auto-Encoder gegeben werden sollen
    """
    start = int(startS)
    count = int(countS)

    answerGrey = input("\n\nIst das Bild in einer Graustufenrepräsentation? Antwortmöglichkeiten ja und nein: ")
    if (not (answerGrey == 'ja' or answerGrey == 'nein')):
        print("Keine erlaubte Antwort. Das Programm wird beendet.")
        sys.exit()

    if (answerGrey == 'ja'):
        """
        Hier werden Graustufen-Bilder aus dem ausgewählten Testdatensatz 
        an einem bestimmten Index ausgewählt und in den Auto-Encoder gegeben und der Loss ermittelt.
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

    else:
        """
        Hier werden RGB-Bilder aus dem ausgewählten Testdatensatz 
        an einem bestimmten Index ausgewählt und in den Auto-Encoder gegeben und der Loss ermittelt.
        """
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



