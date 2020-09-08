"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient zur Erstellung von Datensätzen.
Es werden RGB-Datensätze und Graustufen-Datensätze erstellt.

Der Datensatz stammt von der Webseite: http://atarigrandchallenge.com/data
Grundlage bildet das Paper von Kurin, Vitaly, et al. "The atari grand challenge dataset." arXiv preprint arXiv:1705.10998 (2017).

In diesem Code wird im Falle von Q*Bert schon von einem klassifizierten Datensatz ausgegangen.
"""

import os
from numpy import save
import shutil
import cv2

"""
Funktion wählt Bilder aus einem Ordner aus und teilt diese entweder dem Trainings- oder Testdatensatz zu.
"""
def create_datasets(folder,numberTraining,numberTest):
    #nTraining = 0
    #nTest = 0
    print('Starting:{}'.format(folder))

    """Q*Bert"""
    for img_path in os.listdir(folder):
        if((nTraining == numberTraining) and (nTest == numberTest)):
            return

        if(nTraining < numberTraining):
            path = os.path.join(folder, img_path)
            #shutil.copy(path, '/media/annika/Daten/QBert_SmallDataset/Training')
            #shutil.copy(path, '/media/annika/Daten/QBert_SmallDataset_JustQBert/Training')
            nTraining = nTraining + 1

        if ((nTraining == numberTraining) and (nTest < numberTest)):
            path = os.path.join(folder, img_path)
            #shutil.copy(path, '/media/annika/Daten/QBert_SmallDataset/Test')
            #shutil.copy(path, '/media/annika/Daten/QBert_SmallDataset_JustQBert/Test')
            nTest = nTest + 1


    """Space Invaders"""
    """
    for img_path in os.listdir(folder):
        global numberFramesAll
        if(numberFramesAll == 64050):
            break

        path = os.path.join(folder, img_path)
        newName = os.path.join('/media/annika/Daten/SpaceInvaders_SmallDataset/All','SpaceInvaders_{}.png'.format(numberFramesAll))
        shutil.copy(path, newName)
        numberFramesAll += 1
    """

"""
Funktion wählt Bilder aus einem Trainings- oder Testordner und wandelt diese in ein Graustufen-Bild um.
Das umgewandelte Bild wird dann in einem neuen Datensatzordner gespeichert.
Hier für Space Invaders.
"""
def create_datasets_grey(folder):
    for img_path in os.listdir(folder):
        global numberFramesAll
        if(folder == '/media/annika/Daten/SpaceInvaders_SmallDataset/Test'):
            path = os.path.join(folder, img_path)
            newName = os.path.join('/media/annika/Daten/SpaceInvaders_SmallDataset_Greyscale/Test',
                                   'SpaceInvaders_SmallDataset_Greyscale_Test_{}.png'.format(testFrames))
            testFrames += 1
        elif(folder == '/media/annika/Daten/SpaceInvaders_SmallDataset/Training'):
            path = os.path.join(folder, img_path)
            newName = os.path.join('/media/annika/Daten/SpaceInvaders_SmallDataset_Greyscale/Training',
                                   'SpaceInvaders_SmallDataset_Greyscale_Training_{}.png'.format(trainingFrames))
            trainingFrames+= 1

        path = os.path.join(folder, img_path)
        newName = os.path.join('/media/annika/Daten/SpaceInvaders_SmallDataset_Greyscale/All',
                               'SpaceInvaders_SmallDataset_Greyscale_All_{}.png'.format(numberFramesAll))
        numberFramesAll += 1
        img = cv2.imread(path)
        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(newName, imgGrey)



def save_dataset(pathName, data):
    save(pathName, data)

if __name__ == '__main__':
    #QBert_Mixed
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(0,0,0)__WithQbert', 1400, 600)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(104,180,148)_NoQbert', 3500, 1500)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(104,180,148)__WithQbert', 14000, 6000)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(160,160,52)__WithQbert', 1400, 600)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(172,80,48)__WithQbert', 2100, 900)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(172,120,60)_NoQbert', 6300, 2700)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(172,120,60)__WithQbert', 7000, 3000)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(176,176,176)_NoQbert', 2100, 900)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(176,176,176)__WithQbert', 7000, 3000)

    #QBert__JustQBert
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(0,0,0)__WithQbert', 1435, 615)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(104,180,148)__WithQbert', 17500, 7500)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(160,160,52)__WithQbert', 1400, 600)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(172,80,48)__WithQbert', 2100, 900)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(172,120,60)__WithQbert', 13300, 5700)
    #create_datasets('/media/annika/Daten/QBert_Datensatz_All/ColourClass:(176,176,176)__WithQbert', 9100, 3900)

    #SpaceInvaders 64050 aus den Top Episoden wählen
    #global numberFramesAll
    #numberFramesAll = 0
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/893', 0, 0)
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/1028', 0, 0)
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/739', 0, 0)
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/678', 0, 0)
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/622', 0, 0)
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/436', 0, 0)
    #create_datasets('/media/annika/Daten/BA-Datensaetze/atari_v2_release/screens/spaceinvaders/199', 0, 0)
    

    #Graustufen Datensätze erstellen, hier für SpaceInvaders
    global testFrames
    global trainingFrames
    numberFramesAll = 0
    print("Test")
    create_datasets_grey('/media/annika/Daten/SpaceInvaders_SmallDataset/All')
    print("Training")
    create_datasets_grey('/media/annika/Daten/SpaceInvaders_SmallDataset/Training')
