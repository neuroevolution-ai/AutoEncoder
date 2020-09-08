# Bachelorarbeit

Die Dateien sind im Rahmen der Bachelorarbeit von Annika Kies am FZI Karlsruhe mit dem Titel
"Entwicklung und Analyse von Auto-Encodern für intelligente Agenten zum Erlernen von Atari-Spielen"
entstanden.

Es wurden vier verschiedene Auto-Encoder-Architekturen entworfen, wobei jede Architektur entweder Atari-Bilder in RGB-Darstellung oder Graustufen-Darstellung entgegennehmen kann.
Nähere Beschreibung der Architekturen kann der Bachelorarbeit selbst entnommen werden.

Auto-Encoder 1 orientierte sich im Aufbau an dem Variational Auto-Encoder von Ha und Schmidhuber:
Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631.
https://github.com/ctallec/world-models

Die Datensätze, die zum Training und Testen verwendet wurden entstanden aus dem Datensatz "The atari grand challenge dataset" von Kurin, Vitaly, et al.:
Webseite: http://atarigrandchallenge.com/data
Grundlage bildet das Paper von Kurin, Vitaly, et al. "The atari grand challenge dataset." arXiv preprint arXiv:1705.10998 (2017).

Die Spiele Q*Bert und Space Invaders wurden für die Arbeit verwendet.

**Ordner "Trainierte AutoEncoder"**
Der Ordner "Trainierte AutoEncoder" enthält Auto-Encoder, welche mit verschiedenen Datensätzen trainiert worden sind und angepasste Parameter besitzen.
Außerdem sind die Losswerte von Trainingsdatensatz und Testdatensatz im Trainingsverlauf in einer Datei neben dem trainierten Auto-Encoder dokumentiert.
Aufbau des Ordners ist die Aufteilung der Auto-Encoder-Architektur und dann nach dem Datensatz, mit welchem trainiert worden ist.
Der Auto-Encoder 4 in Graustufen- und RGB-Form erreicht die besten Ergebnisse.

**Hinweis**
Im Code sind die Dateipfade von verwendeten Dateien manuell gesetzt. Wenn der Code wiederverwendet werden soll, müssen Dateipfade neu gesetzt werden.

## Datensatzgenerierung

Folgende PyTorch-Dateien enthalten Code für die Generierung von Trainings- und Testdatensätzen.

**KlassifizierungQBert.py**
Hier werden Q*Bert Atari-Bilder einer Farbklasse (abhängig von der Pyramidenfarbe) und der Eigenschaft "QBert im Bild" zugeordnet.
Die Bilder werden entsprechend ihrer Klasse in einem Ordner gespeichert.

**SpaceInvadersTopSpielepisoden.py**
Hier werden die besten 7 Spielepisoden von Space Invaders aus dem Datensatz "The atari grand challenge dataset." bestimmt.
Diese werden in der Datei Datensatzerstellung.py für den Aufbau der Space Invaders Datensätze verwendet.

**Datensatzerstellung.py**
Hier werden die Trainings- und Testdatensätze erstellt. 
Bei Q*bert wird aus den Klassen eine bestimmte Anzahl von Bildern entnommen und dem Trainings- oder Testdatensatz zugeordnet.
Bei Space Invaders wird aus den besten 7 Spielepisoden ein Trainings- oder Testdatensatz erstellt.
Es entstehen pro Datensatz zwei Ordner mit dem Namen "Train" und "Test", die Atari-Bilder enthalten.

**Datensatzumwandlung.py**
Hier wird für einen Datensatz zwei .npy-Dateien erstellt. 
Diese Dateien können dann direkt für das Training und das Testen des Auto-Encoders verwendet werden.

## Auto-Encoder Dateien

Folgende PyTorch-Dateien enthalten Code für das Trainieren und Testen der Auto-Encoder

**AutoEncoderTraining.py**
Hier kann eine Auto-Encoder-Architektur und ein Datensatz ausgewählt werden, damit kann ein Auto-Encoder trainiert und validiert werden.
Ist das Training abgeschlossen, wird der Auto-Encoder gespeichert.
Hinweis: Für die Generierung eines Plots für den Trainingsverlauf wurde Visdom verwendet. Siehe: https://github.com/noagarcia/visdom-tutorial

**AutoEncoderTest_Gesamt.py**
Hier kann ein trainierter Auto-Encoder und Testdatensatz geladen werden.
Es kann der Median Loss, beste Loss und schlechteste Loss mit dem geladenen Auto-Encoder im Bezug auf den Testdatensatz ermittelt werden.

**AutoEncoderTest_Stichprobe.py**
Hier kann ein trainierter Auto-Encoder und Testdatensatz geladen werden.
Es ist möglich mit diesem Code Eingabe und Ausgabe eines Auto-Encoders zu vergleichen.





