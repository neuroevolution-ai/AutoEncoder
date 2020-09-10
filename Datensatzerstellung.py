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
def create_datasets(folder, numberTraining, numberTest, qBert, qBertmixed, numberFramesAll):
    nTraining = 0
    nTest = 0
    print('Starting:{}'.format(folder))

    if(qBert):
        """Q*Bert"""
        for img_path in os.listdir(folder):
            if((nTraining == numberTraining) and (nTest == numberTest)):
                return

            if(nTraining < numberTraining):
                path = os.path.join(folder, img_path)
                if(qBertmixed):
                    shutil.copy(path, 'datasets/QBert_SmallDataset/Training')
                else:
                    shutil.copy(path, 'datasets/QBert_SmallDataset_JustQBert/Training')
                nTraining = nTraining + 1

            if ((nTraining == numberTraining) and (nTest < numberTest)):
                path = os.path.join(folder, img_path)
                if (qBertmixed):
                    shutil.copy(path, 'datasets/QBert_SmallDataset/Test')
                else:
                    shutil.copy(path, 'datasets/QBert_SmallDataset_JustQBert/Test')
                nTest = nTest + 1
    else:
        """Space Invaders"""
        for img_path in os.listdir(folder):
            if(numberFramesAll == 64050):
                break

            path = os.path.join(folder, img_path)
            newName = os.path.join('datasets/SpaceInvaders_SmallDataset/All','SpaceInvaders_{}.png'.format(numberFramesAll))
            shutil.copy(path, newName)
            numberFramesAll += 1

    return numberFramesAll

"""
Funktion wählt Bilder aus einem Trainings- oder Testordner und wandelt diese in ein Graustufen-Bild um.
Das umgewandelte Bild wird dann in einem neuen Datensatzordner gespeichert.
Hier für Space Invaders.
"""
def create_datasets_grey(folder, numberFramesAll, testFrames, trainingFrames):
    for img_path in os.listdir(folder):
        if(folder == 'SpaceInvaders_SmallDataset/Test'):
            path = os.path.join(folder, img_path)
            newName = os.path.join('datasets/SpaceInvaders_SmallDataset_Greyscale/Test',
                                   'SpaceInvaders_SmallDataset_Greyscale_Test_{}.png'.format(testFrames))
            testFrames += 1
            img = cv2.imread(path)
            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(newName, imgGrey)
        elif(folder == '/media/annika/Daten/SpaceInvaders_SmallDataset/Training'):
            path = os.path.join(folder, img_path)
            newName = os.path.join('datasets/SpaceInvaders_SmallDataset_Greyscale/Training',
                                   'SpaceInvaders_SmallDataset_Greyscale_Training_{}.png'.format(trainingFrames))
            trainingFrames+= 1
            img = cv2.imread(path)
            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(newName, imgGrey)

        path = os.path.join(folder, img_path)
        newName = os.path.join('datasets/SpaceInvaders_SmallDataset_Greyscale/All',
                               'SpaceInvaders_SmallDataset_Greyscale_All_{}.png'.format(numberFramesAll))
        numberFramesAll += 1
        img = cv2.imread(path)
        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(newName, imgGrey)

    return numberFramesAll, testFrames, trainingFrames



def save_dataset(pathName, data):
    save(pathName, data)

if __name__ == '__main__':


    #QBert - RGB Mixed Datensatz erstellen

    if not os.path.exists('datasets/QBert_SmallDataset/Training'):
        os.mkdir('datasets/QBert_SmallDataset/Training')
    if not os.path.exists('datasets/QBert_SmallDataset/Test'):
        os.mkdir('datasets/QBert_SmallDataset/Test')

    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(0,0,0)__WithQbert', 1400, 600, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(104,180,148)_NoQbert', 3500, 1500, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(104,180,148)__WithQbert', 14000, 6000, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(160,160,52)__WithQbert', 1400, 600, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(172,80,48)__WithQbert', 2100, 900, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(172,120,60)_NoQbert', 6300, 2700, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(172,120,60)__WithQbert', 7000, 3000, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(176,176,176)_NoQbert', 2100, 900, True, True, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(176,176,176)__WithQbert', 7000, 3000, True, True, 0)

    #QBert - RGB JustQBert Datensatz erstellen

    if not os.path.exists('datasets/QBert_SmallDataset_JustQBert/Training'):
        os.mkdir('datasets/QBert_SmallDataset_JustQBert/Training')
    if not os.path.exists('datasets/QBert_SmallDataset_JustQBert/Test'):
        os.mkdir('datasets/QBert_SmallDataset_JustQBert/Test')

    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(0,0,0)__WithQbert', 1435, 615, True, False, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(104,180,148)__WithQbert', 17500, 7500, True, False, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(160,160,52)__WithQbert', 1400, 600, True, False, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(172,80,48)__WithQbert', 2100, 900, True, False,0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(172,120,60)__WithQbert', 13300, 5700, True, False, 0)
    create_datasets('datasets/QBert_Datensatz_All/ColourClass:(176,176,176)__WithQbert', 9100, 3900, True, False, 0)

    #Space Invaders - RGB 64050 aus den Top Episoden wählen und Datensatz erstellen
    #wenn sich die besten Spielepisoden geändert haben, müssen die Pfade angepasst werden und andere Spielepisoden bestimmt werden

    if not os.path.exists('datasets/SpaceInvaders_SmallDataset/All'):
        os.mkdir('datasets/SpaceInvaders_SmallDataset/All')

    numberFramesAll = 0
    create_datasets('atari_v2_release/screens/spaceinvaders/893', 0, 0, False, False, numberFramesAll)
    numberFramesAll = create_datasets('atari_v2_release/screens/spaceinvaders/1028', 0, 0, False, False, numberFramesAll)
    numberFramesAll = create_datasets('atari_v2_release/screens/spaceinvaders/739', 0, 0, False, False, numberFramesAll)
    numberFramesAll = create_datasets('atari_v2_release/screens/spaceinvaders/678', 0, 0, False, False, numberFramesAll)
    numberFramesAll = create_datasets('atari_v2_release/screens/spaceinvaders/622', 0, 0, False, False, numberFramesAll)
    numberFramesAll = create_datasets('atari_v2_release/screens/spaceinvaders/436', 0, 0, False, False, numberFramesAll)
    numberFramesAll = create_datasets('atari_v2_release/screens/spaceinvaders/199', 0, 0, False, False, numberFramesAll)
    

    #Graustufen Datensätze erstellen, hier für SpaceInvaders

    if not os.path.exists('datasets/SpaceInvaders_SmallDataset_Greyscale/Test'):
        os.mkdir('datasets/SpaceInvaders_SmallDataset_Greyscale/Test')
    if not os.path.exists('datasets/SpaceInvaders_SmallDataset_Greyscale/Training'):
        os.mkdir('datasets/SpaceInvaders_SmallDataset_Greyscale/Training')
    if not os.path.exists('datasets/SpaceInvaders_SmallDataset_Greyscale/All'):
        os.mkdir('datasets/SpaceInvaders_SmallDataset_Greyscale/All')

    testFrames = 0
    trainingFrames = 0
    numberFramesAll = 0
    print("Test")
    numberFramesAll, testFrames, trainingFrames = create_datasets_grey('SpaceInvaders_SmallDataset/Test',numberFramesAll,testFrames,trainingFrames)
    print("Training")
    create_datasets_grey('SpaceInvaders_SmallDataset/Training', testFrames, trainingFrames)
