"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zur Erstellung von Datensätzen in einer .npy-Datei.
Datensatz liegt an sich schon in einem Ordner vor, hier wird nur die Umwandlung von Bilder in eine .npy-Datei vorgenommen.
Hier sind die Datensätze Q*Bert-Graustufen und Space Invaders - Graustufen gezeigt. 
"""

from PIL import Image
from numpy import asarray
import os
from numpy import save
import random

"""
Funktion iteriert über jedes Bild im Ordner und wandelt es in ein Array um.
Array wird dann dem Testdatensatz hinzugefügt.
"""
def create_arrayTest(folder):

    for img_path in os.listdir(folder):
        path = os.path.join(folder, img_path)
        img = Image.open(path)
        img_array = asarray(img)  # convert to array
        test_data.append(img_array)  # add this to our training_data

"""
Funktion iteriert über jedes Bild im Ordner und wandelt es in ein Array um.
Array wird dann dem Trainingsdatensatz hinzugefügt.
"""
def create_arrayTraining(folder):

    for img_path in os.listdir(folder):
        path = os.path.join(folder, img_path)
        img = Image.open(path)
        img_array = asarray(img)  # convert to array
        training_data.append(img_array)  # add this to our training_data

"""
Funktion speichert einen Datensatz, mit Pfadangabe.
"""
def save_dataset(pathName, data):
    save(pathName, data)

if __name__ == '__main__':

    #Folgend sind Beispiele für die Datensätze QBert_SmallDataset_Mixed_Greyscale und SpaceInvaders_SmallDataset_Greyscale
    """Q*Bert"""
    test_data = []
    create_arrayTest('/media/annika/Daten/QBert_SmallDataset_Mixed_Greyscale/Test')
    print(len(test_data))
    print(test_data[0].shape)

    save_dataset('/home/annika/BA-Datensaetze/SmallDatasetTest_Q*Bert_Mixed_Greyscale.npy', test_data)

    training_data = []
    create_arrayTraining('/media/annika/Daten/QBert_SmallDataset_Mixed_Greyscale/Training')
    print(len(training_data))
    print(training_data[0].shape)

    save_dataset('/home/annika/BA-Datensaetze/SmallDatasetTraining_Q*Bert_Mixed_Greyscale.npy', training_data)


    """Space Invaders"""
    test_data = []
    training_data1 = []
    training_data2 = []

    for img_path in os.listdir('/media/annika/Daten/SpaceInvaders_SmallDataset_Greyscale/All'):

        path = os.path.join('/media/annika/Daten/SpaceInvaders_SmallDataset_Greyscale/All', img_path)
        img = Image.open(path)
        img_array = asarray(img)

        if(bool(random.getrandbits(1)) and len(test_data) < 19215):
            test_data.append(img_array)
            if(len(test_data) == 19215):
                print(len(test_data))
                print(test_data[0].shape)
                save_dataset('/home/annika/BA-Datensaetze/smallDatasetTest_SpaceInvaders_Greyscale.npy', test_data)
        else:
            if(len(training_data1) < 22418):
                training_data1.append(img_array)
                if (len(training_data1) == 22418):
                    print(len(training_data1))
                    print(training_data1[0].shape)
                    save_dataset('/home/annika/BA-Datensaetze/smallDatasetTraining1_SpaceInvaders_Greyscale.npy', training_data1)
            else:
                training_data2.append(img_array)


    print(len(training_data2))
    print(training_data2[0].shape)
    save_dataset('/home/annika/BA-Datensaetze/smallDatasetTraining2_SpaceInvaders_Greyscale.npy', training_data2)