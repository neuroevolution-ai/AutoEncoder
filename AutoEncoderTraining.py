"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zum Training eines Auto-Encoders,
trainierter Auto-Encoder wird danach in einem (neuen) Ordner gespeichert.

Implementiert sind vier Auto-Encoder Architekturen.
Es können 5 verschiedene Datensätze verwendet werden.

Folgende Datensätze stehen zur Verfügung:

    Q*Bert - RGB JustQbert
    Q*Bert - RGB Mixed
    Q*bert - Graustufen
    SPACE INVADERS - RGB
    SPACE INVADERS - Graustufen

AutoEncoder1 orientiert sich an:
Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631.
https://github.com/ctallec/world-models

Zur Orientierung diente das Tutorial https://www.kaggle.com/jagadeeshkotra/autoencoders-with-pytorch
Teile des Codes wurden aus dem Tutorial übernommen.
"""
import sys
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from numpy import load
from torch.utils.data import TensorDataset
from numpy import asarray
import os
from visdom import Visdom
import statistics
import csv

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
Funktion trainiert einen Auto-Encoder und speichert den trainierten Auto-Encoder
dann relativ zum Programmcode in einem (neuen) Ordner Trainingsergebnis.
Der Kürzer für die Auto-Encoder-Architektur, der Name des Datensets, die Anzahl der Epochen muss übergeben werden.
Außerdem muss der Parameter "grey" übergeben werden:
    grey ist True, wenn der Auto-Encoder auf einem Graustufendatensatz trainiert werden soll.
"""
def startAE(autoencoder, dataset, grey, epochs):

    myDataYTrain = []
    myDataXTrain = []
    myDataYTest = []
    myDataXTest = []
    myDataYTrain.append(0)
    myDataXTrain.append(0)
    myDataYTest.append(0)
    myDataXTest.append(0)

    # Überprüfung welcher Datensatz als Grundlage verwendet werden soll, anschließendes Laden des Datensatzes
    if(grey):
        if (dataset == 'SmallDataset_Q*Bert_Mixed_Greyscale'):
            split = False
            datasetTrain = load_dataset(
                'datasets/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTraining_Q*Bert_Mixed_Greyscale.npy')
            datasetTest = load_dataset(
                'datasets/SmallDataset_Q*Bert_Mixed_Greyscale/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy')
            title = 'Auto-Encoder mit SmallDatasetTest_Q*Bert_Mixed_Greyscale'

        elif (dataset == 'SmallDataset_SpaceInvaders_Greyscale'):
            split = True
            datasetTrain = load_dataset(
                'datasets/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTraining1_SpaceInvaders_Greyscale.npy')
            datasetTrain2 = load_dataset(
                'datasets/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTraining2_SpaceInvaders_Greyscale.npy')
            datasetTest = load_dataset(
                'datasets/SmallDataset_SpaceInvaders_Greyscale/smallDatasetTest_SpaceInvaders_Greyscale.npy')
            title = 'Auto-Encoder mit SmallDataset_SpaceInvaders_Greyscale'

        # Bild aus dem Trainingsdatensatz laden, dient zur Kontrolle
        TrainingImage = Image.fromarray(datasetTrain[10])
        plt.imshow(TrainingImage, cmap=plt.get_cmap('gray'))
        plt.show()

        # Bild aus dem Testdatensatz laden, dient zur Kontrolle
        TestImage = Image.fromarray(datasetTest[10])
        plt.imshow(TestImage, cmap=plt.get_cmap('gray'))
        plt.show()

    else:
        if (dataset == 'SmallDataset_SpaceInvaders'):
            split = True
            grey = False
            datasetTrain = load_dataset(
                'datasets/SmallDataset_SpaceInvaders/smallDatasetTraining1_SpaceInvaders.npy')
            datasetTrain2 = load_dataset(
                'datasets/SmallDataset_SpaceInvaders/smallDatasetTraining2_SpaceInvaders.npy')
            datasetTest = load_dataset(
                'datasets/SmallDataset_SpaceInvaders/smallDatasetTest_SpaceInvaders.npy')
            title = 'Auto-Encoder mit SmallDataset_SpaceInvaders'
        elif (dataset == 'SmallDataset_Q*Bert_Mixed'):
            split = False
            grey = False
            datasetTrain = load_dataset('datasets/SmallDataset_Q*Bert_Mixed/smallDatasetTraining.npy')
            datasetTest = load_dataset('datasets/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
            title = 'Auto-Encoder mit SmallDataset_Q*Bert_Mixed'
        elif (dataset == 'SmallDataset_JustQbert'):
            split = False
            grey = False
            datasetTrain = load_dataset(
                'datasets/SmallDataset_JustQBert/smallDatasetTraining_JustQBert.npy')
            datasetTest = load_dataset('datasets/SmallDataset_Q*Bert_Mixed/smallDatasetTest.npy')
            title = 'Auto-Encoder mit SmallDataset_Q*Bert_JustQBert'

        # Bild aus dem Trainingsdatensatz laden, dient zur Kontrolle
        TrainingImage = Image.fromarray(datasetTrain[10])
        plt.imshow(TrainingImage)
        plt.show()

        # Bild aus dem Testdatensatz laden, dient zur Kontrolle
        TestImage = Image.fromarray(datasetTest[10])
        plt.imshow(TestImage)
        plt.show()

    # Aufbau Trainingsdatensatz
    print("Training Datensatz:")
    print(len(datasetTrain))
    print(datasetTrain[10].shape)

    if(split):
        # Aufbau Trainingsdatensatz2
        print("Training Datensatz 2:")
        print(len(datasetTrain2))
        print(datasetTrain2[10].shape)

    # Aufbau Testdatensatz
    print("Test Datensatz:")
    print(len(datasetTest))
    print(datasetTest[10].shape)

    #Auto-Encoder wird geladen
    if(grey):
        if (autoencoder == 'AE1GS'):
            ae = AutoEncoder1Grey.AutoEncoder1Grey()
        elif (autoencoder == 'AE2GS'):
            ae = AutoEncoder2Grey.AutoEncoder2Grey()
        elif (autoencoder == 'AE3GS'):
            ae = AutoEncoder3Grey.AutoEncoder3Grey()
        elif (autoencoder == 'AE4GS'):
            ae = AutoEncoder4Grey.AutoEncoder4Grey()
        else:
            print('Der Auto-Encoder {} kann nicht mit dem Datensatz {} trainiert werden.'.format(autoencoder,dataset))
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

    if torch.cuda.is_available():
        ae.to(torch.device("cuda:0"))
    print(ae)

    # definiert die Loss-Funktion und den Optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(ae.parameters(), lr=4e-4)

    # startet den Plot
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')



    # Aufteilung des Datensatzes in kleinere Sequenzen, kompletter Datensatz zu groß für direkte Verarbeitung
    if (split):
        iterationsTrain = ((len(datasetTrain) + len(datasetTrain2)) // 1000)
        firstIterationTrain = (len(datasetTrain) // 1000)
    else:
        iterationsTrain = (len(datasetTrain) // 1000)
        firstIterationTrain = iterationsTrain
    if (len(datasetTrain) % 1000 != 0):
        iterationsTrain += 1

    predictions = []

    for e in range(epochs):
        for i in range(iterationsTrain):

            #Datensatz wird hier in Trainingssequenzen von 1000 Bildern aufgeteilt
            train_snippet = i + (e * iterationsTrain)
            losses = []
            startTrain = i * 1000
            stopTrain = ((i + 1) * 1000) - 1
            #falls der Trainingsdatensatz in zwei Teilen vorliegt
            if (split):
                #Teil 1
                if (i + 1 < firstIterationTrain):
                    trainSetSnippet = datasetTrain[startTrain:stopTrain, :, :]
                else:
                    if (i + 1 == firstIterationTrain):
                        trainSetSnippet = datasetTrain[startTrain:, :, :]
                    # Teil 2
                    else:
                        startTrain = (i - firstIterationTrain) * 1000
                        stopTrain = ((i - firstIterationTrain + 1) * 1000) - 1
                        if (i + 1 == iterationsTrain):
                            trainSetSnippet = datasetTrain2[startTrain:, :, :]
                        else:
                            trainSetSnippet = datasetTrain2[startTrain:stopTrain, :, :]
            #falls der Trainingsdatensatz in einem Teil vorliegt
            else:
                #die nächsten 1000 Bilder aus dem Datensatz nehmen
                if (i + 1 < firstIterationTrain):
                    trainSetSnippet = datasetTrain[startTrain:stopTrain, :, :]
                #die letzten Bilder aus dem Datensatz nehmen
                else:
                    trainSetSnippet = datasetTrain[startTrain:, :, :]

            if(grey):
                trainSetSnippet = trainSetSnippet.reshape(len(trainSetSnippet),210,160,1)
            trainSetSnippet = normalize(trainSetSnippet)
            trn_torch = torch.from_numpy(trainSetSnippet).type(torch.cuda.FloatTensor)
            trn_torch = trn_torch.permute(0, 3, 1, 2)
            trn_torch = trn_torch[:, :, :, :]
            trn = TensorDataset(trn_torch, trn_torch)
            trn_dataloader = torch.utils.data.DataLoader(trn, batch_size=1, shuffle=False, num_workers=0)

            # in einer Trainingssequenzen werden immer 430 Bilder aus dem Testdatensatz zur Validierung entnommen
            startTest = i * 430
            stopTest = ((i + 1) * 430) - 1
            if (i + 1 == iterationsTrain):
                testSetSnippet = datasetTest[startTest:, :, :]
            else:
                testSetSnippet = datasetTest[startTest:stopTest, :, :]

            if(grey):
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
                if(grey):
                    x = torch.tensor([255], dtype=torch.int, device=torch.device("cuda:0"))
                    testImg = test_torch[2] * x
                    show_torch_image_Grey(testImg.reshape(210,160))

                    predImg = predictions[2].permute(1, 2, 0).detach() * torch.tensor([255], dtype=torch.int,device=torch.device("cuda:0"))
                    show_torch_image_Grey(predImg.reshape(210,160))
                else:
                    show_torch_image(
                        test_torch[2] * torch.tensor([255, 255, 255], dtype=torch.int, device=torch.device("cuda:0")))
                    show_torch_image(predictions[2].permute(1, 2, 0).detach() * torch.tensor([255, 255, 255],
                                                                                             dtype=torch.int,
                                                                                             device=torch.device(
                                                                                                 "cuda:0")))

                test_torch = test_torch.permute(0, 3, 1, 2)
            predictions = []

    # Speichern des trainierten Auto-Encoders
    if not os.path.exists('Trainingsergebnis'):
        os.mkdir('Trainingsergebnis')

    pathAE = 'Trainingsergebnis/ae.pt'
    torch.save(ae.state_dict(), pathAE)

    myData = np.array([myDataXTrain, myDataYTrain, myDataYTest]).transpose()

    pathEvaluation = 'Trainingsergebnis/LossAndVal.csv'
    myFile = open(pathEvaluation, 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)



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
    datasetChoice = input("\n\nMit welchem Datensatz soll der Auto-Encoder trainiert werden?\n\n"
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

    epochsS = input("\n\nWie viele Epochen soll trainiert werden?\n\n"
                          "Ihre Wahl ist: ")
    if (not (str.isnumeric(epochsS))):
        print("\n\nSie haben keine gültige Zahl eingegeben. Das Programm bricht ab.")
        sys.exit()

    epochs = int(epochsS)
    startAE(autoencoder,dataset,grey,epochs)