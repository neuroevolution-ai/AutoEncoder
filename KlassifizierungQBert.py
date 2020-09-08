"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zur Klassifizierung von Atari Bildern im Spiel Q*Bert.

Der Datensatz stammt von der Webseite: http://atarigrandchallenge.com/data
Grundlage bildet das Paper von Kurin, Vitaly, et al. "The atari grand challenge dataset." arXiv preprint arXiv:1705.10998 (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import os
from numpy import save
import shutil

"""
Funktion iteriert über jedes Bild in einem Ordnern und konvertiert es in ein Array
"""
# iterate over each image in one folder and convert it to a an array
def create_training_data(folder):

    for img_path in os.listdir(folder):
        path = os.path.join(folder, img_path)
        img = Image.open(path)
        img_array = asarray(img)  # convert to array
        training_data.append(img_array)  # add this to our training_data

"""
Funktion speichert einen Datensatz
"""
def save_dataset(pathName, data):
    save(pathName, data)

"""
Funktion ermittelt den Farbwert an Pixel (75,50) und bestimmt somit die Pyramidenfarbe des Atari-Bildes
Farbcode wird dem Array boxColoursAndQuantity angefügt.
"""
def countNumberOfDifferentColours(folder):

    for img_path in os.listdir(folder):
        path = os.path.join(folder, img_path)
        img = Image.open(path)
        pix = img.load()
        actualColour = pix[75, 50]
        found = 0
        if (len(boxColoursAndQuantity) != 0):
            for col in boxColoursAndQuantity:
                if(col[0] == actualColour[0]):
                    if(col[1] == actualColour[1]):
                        if (col[2] == actualColour[2]):
                            col[3]= col[3] + 1
                            found = 1

        if(found == 0):
            boxColoursAndQuantity.append([actualColour[0],actualColour[1],actualColour[2],1])

        #searchQbert(pix)

"""
Funktion erhöht die Menge an Farben, wenn die Farbe schon existiert.
Wenn die Farbe noch nicht existiert, wird eine neue Farbnummer hinzugefügt.
"""
def addColourToboxColoursAndQuantity(actualColour):
    found = 0
    if (len(boxColoursAndQuantity) != 0):
        for col in boxColoursAndQuantity:
            if (col[0] == actualColour[0]):
                if (col[1] == actualColour[1]):
                    if (col[2] == actualColour[2]):
                        col[3] = col[3] + 1
                        found = 1

    if (found == 0):
        boxColoursAndQuantity.append([actualColour[0], actualColour[1], actualColour[2], 1])

"""
Funktion gibt die Farbklasse der Pyramide, abhängig von Pixel (75,50) zurück
"""
def getColourClass(imgPath):
    img = Image.open(imgPath)
    pix = img.load()
    return pix[75, 50]

"""
Funktion checkt, ob Q*bert im Bild vorhanden ist, seine Farbe ist: (172,80,48)
Gibt False zurück, wenn Q*Bert fehlt und sonst True
"""
def checkQBert(imgPath):
    img = Image.open(imgPath)
    pix = img.load()
    for pW in range(160):
        for pH in range(210):
            p = pix[pW,pH]
            if (p[0] == 172) & (p[1] == 80) & (p[2] == 48):
                return True

    return False

"""
Funktion checkt, ob die Farbklasse schon gefunden wurde, basierend auf boxColoursAndQuantity
Gibt True oder False zurück.
"""
def checkColourClass(imgPath):
    colour = getColourClass(imgPath)
    if (len(boxColoursAndQuantity) != 0):
        for col in boxColoursAndQuantity:
            if (col[0] == colour[0]):
                if (col[1] == colour[1]):
                    if (col[2] == colour[2]):
                        return True

    return False


    return False

"""
Funktion gibt alle Farbklassen und ihre Häufigkeit auf der Konsole aus.
"""
def printColours():

    i = 1
    for col in boxColoursAndQuantity:
        print('Colour Nr.{}: ({},{},{}) Quantity:{}'.format(
            i,
            col[0],
            col[1],
            col[2],
            col[3]))
        i = i + 1

"""
Funktion checkt, ob Q*bert im Bild vorhanden ist, seine Farbe ist: (172,80,48)
Wenn ja, wird die Häufigkeit von Q*Bert erhöht: qBertQuantity + 1
"""
def searchQbert(pixel):
    for pW in range(160):
        for pH in range(210):
            p = pixel[pW,pH]
            if (p[0] == 172) & (p[1] == 80) & (p[2] == 48):
                global qBertQuantity
                qBertQuantity = qBertQuantity + 1
                return 1

    return 0



"""
Funktion lädt Beispielbilder und schneidet Q*Bert aus.
Q*Bert wird als Bild ausgegeben.
"""
def exampleQberts():
    qbert = Image.open('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/qbert/141/3962.png')
    plt.imshow(qbert)
    plt.show()
    plt.imshow(qbert.crop((35, 145, 50, 160)))
    plt.show()
    pix1 = qbert.crop((35, 145, 50, 160)).load()
    qBertColour1 = pix1[7, 8]
    print(qBertColour1)

    qbert = Image.open('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/qbert/141/2234.png')
    plt.imshow(qbert)
    plt.show()
    plt.imshow(qbert.crop((75, 70, 92, 90)))
    plt.show()
    pix2 = qbert.crop((75, 70, 92, 90)).load()
    qBertColour2 = pix2[8, 11]
    print(qBertColour2)

    qbert = Image.open('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/qbert/141/935.png')
    plt.imshow(qbert)
    plt.show()
    plt.imshow(qbert.crop((60, 100, 75, 125)))
    plt.show()
    pix3 = qbert.crop((60, 100, 75, 125)).load()
    qBertColour3 = pix3[6, 13]
    print(qBertColour3)

"""
Funktion generiert einen Namen für das Bild:
    colourClass: Sagt aus, welche Farbklasse ein Bild erhält, abhängig von der Pyramidenfarbe in (r,g,b)
    qBert:       False, wenn Q*Bert nicht im Bild ist -> "NoQbert"
                 True, wenn Q*bert im Bild ist -> "WithQbert"
    Number:      Nummer des Bildes
"""
def createNameImg(colourClass,qBert,number):
    if qBert:
        return '{}_({},{},{})_WithQbert.png'.format(number, colourClass[0], colourClass[1], colourClass[2])
    else:
        return '{}_({},{},{})_NoQbert.png'.format(number,colourClass[0],colourClass[1],colourClass[2])

"""
Funktion generiert einen Namen für den Ordner:
    colourClass: Sagt aus, welche Farbklasse ein Bild erhält, abhängig von der Pyramidenfarbe in (r,g,b)
    qBert:       False, wenn Q*Bert nicht im Bild ist -> "NoQbert"
                 True, wenn Q*bert im Bild ist -> "WithQbert"
    Number:      Nummer des Bildes
"""
def createNameFolder(colourClass,qBert):
    if qBert:
        return 'ColourClass:({},{},{})__WithQbert.png'.format(colourClass[0], colourClass[1], colourClass[2])

    else:
        return 'ColourClass:({},{},{})_NoQbert'.format(colourClass[0],colourClass[1],colourClass[2])
"""
Funktion generiert einen neuen Ordner mit dem gegebenen Pfad
"""
def createFolder(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

"""
Start der Klassifizierung.
Grundlage ist der Datensatz von der Webseite: http://atarigrandchallenge.com/data
"""
if __name__ == '__main__':
    """training_data = []
    create_training_data('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/qbert/7')

    print(len(training_data))
    print(training_data[0].shape)

    save_dataset('/home/annika/BA-Datensaetze/datasetQBertFolder7.npy', training_data)
    """
    global boxColoursAndQuantity
    boxColoursAndQuantity = []
    global qBertQuantity
    qBertQuantity = 0
    #countNumberOfDifferentColours('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/qbert/141')

    pathQBertPicFolder = '/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/qbert'
    pathNewQBertDataset = '/media/annika/Daten/QBert_Datensatz'

    amountOfImages = 0
    for f in os.listdir(pathQBertPicFolder):
        folder = os.path.join(pathQBertPicFolder, f)
        for img in os.listdir(folder):

            img_path = os.path.join(folder, img)
            colourClass = getColourClass(img_path)
            qBert = checkQBert(img_path)

            folderpath = os.path.join(pathNewQBertDataset, createNameFolder(colourClass, qBert))
            createFolder(folderpath)
            number = len(os.listdir(folderpath))
            pathName = os.path.join(folderpath, createNameImg(colourClass, qBert, number))
            shutil.copy(img_path, pathName)
            amountOfImages = amountOfImages + 1

            addColourToboxColoursAndQuantity(colourClass)
            if qBert:
                qBertQuantity = qBertQuantity + 1

        print('Folder: {} is finished.'.format(folder))
        printColours()
        print('Quantity of qBerts: {} of {} images.'.format(qBertQuantity, amountOfImages))
