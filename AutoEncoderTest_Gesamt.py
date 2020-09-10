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
import sys

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from numpy import load, os
from torch.utils.data import TensorDataset
import statistics

#Auto-Encoder importieren
import AutoEncoder1
import AutoEncoder1Grey
import AutoEncoder2
import AutoEncoder2Grey
import AutoEncoder3
import AutoEncoder3Grey
import AutoEncoder4
import AutoEncoder4Grey


def normalize(v):
    return v / 255


"""
Funktion führt den gesamten Testdurchlauf eines Auto-Encoders durch.
"""
def startAE(dataset, grey, losses, ae):

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

    if(grey):
        # Bild aus dem Datensatz laden, dient zur Kontrolle
        TestImage = Image.fromarray(datasetTest[10])
        plt.imshow(TestImage, cmap=plt.get_cmap('gray'))
        plt.show()
    else:
        # Bild aus dem Datensatz laden, dient zur Kontrolle
        TestImage = Image.fromarray(datasetTest[10])
        plt.imshow(TestImage)
        plt.show()

    # Aufbau Datensatz
    print("Test Datensatz:")
    print(len(datasetTest))
    print(datasetTest[10].shape)

    if torch.cuda.is_available():
        ae.to(torch.device("cuda:0"))
    print(ae)

    # definiert die Loss-Funktion
    loss_func = nn.MSELoss()

    # Aufteilung des Datensatzes in kleinere Sequenzen, kompletter Datensatz zu groß für direkte Verarbeitung
    iterationsTest = len(datasetTest) // 1000
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

        if(grey):
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

    return losses

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

    losses = []

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

    losses = startAE(dataset, grey, ae)

    # Ausgabe der Losswerte
    print(losses)
    print(len(losses))
    print('Median Loss: {}'.format(statistics.median(losses)))
    print('Worst Loss: {}'.format(max(losses)))
    print('Best Loss: {}'.format(min(losses)))

