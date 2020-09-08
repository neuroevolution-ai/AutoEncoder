"""
Code ist im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Code dient dazu die besten Spielepisoden im Datensatz herauszufinden.

Der Datensatz stammt von der Webseite: http://atarigrandchallenge.com/data
Grundlage bildet das Paper von Kurin, Vitaly, et al. "The atari grand challenge dataset." arXiv preprint arXiv:1705.10998 (2017).
"""
import os


"""
Funktion gibt die letzte Zeile des Dokuments aus
"""
def getLastLine(trajectoryFilePath):
    with open(trajectoryFilePath, "r") as file:
        for last_line in file:
            pass

    return last_line

"""
Es werden die besten Spielepisoden aus SPACE INVADERS bestimmt und ausgegeben.
Datensatz stammt von der Webseite: http://atarigrandchallenge.com/data
"""
if __name__ == '__main__':
    listOfAll = []
    listOfTop7 = []
    folder = '/media/annika/Daten/BA-Datensaetze/atari_v2_release/trajectories/spaceinvaders'
    for textFile_path in os.listdir(folder):
        path = os.path.join(folder, textFile_path)
        last_line = getLastLine(path)
        last_line_split = last_line.split(',')
        filenumber = textFile_path.split('.')
        element = [filenumber[0],int(last_line_split[0])+1,last_line_split[2]]

        listOfAll.append(element)

        if(len(listOfTop7) < 7):
            listOfTop7.append(element)
            continue

        smallestElement = listOfTop7[0]
        for e in listOfTop7:
            number = int(smallestElement[2])
            if(int(smallestElement[2]) > int(e[2])):
                smallestElement = e

        if(int(smallestElement[2]) < int(element[2])):
            listOfTop7.remove(smallestElement)
            listOfTop7.append(element)

    print("--------------------------listOfTop7-----------------------------")
    numberimages = 0
    for e in listOfTop7:
        numberimages += e[1]
        print(e)
    print('Anzahl an Bilder: {}'.format(numberimages))
    print("--------------------------All-----------------------------")
    #for e in listOfAll:
     #   print(e)
