# Hundearten-Erkennung mit Transfer Learning

Dieses Projekt untersucht die automatische Erkennung von Hundearten mit Convolutional Neural Networks (CNNs) auf Basis eines eigenen Bilddatensatzes sowie einer zusätzlichen Haustier-Klasse **Garo**. Als Ausgangspunkt dient ein vortrainiertes ResNet18, das per Transfer Learning auf die Zielklassen angepasst wird; zusätzlich wird ein zweites vortrainiertes Modell als Vergleichsarchitektur trainiert, um Unterschiede in Accuracy, F1-Score und Konfusionsmatrix auszuwerten.  

## Die Projektidee

Ziel des Projekts ist die Bildklassifikation von Hunden in insgesamt 121 Klassen: 120 reguläre Hunderassen aus dem Datensatz plus die zusätzliche eigene Klasse **Garo**. Das System soll am Ende nicht nur Vorhersagen für einzelne Testbilder liefern, sondern auch nachvollziehbar zeigen, wie gut insbesondere die Garo-Klasse gegenüber den anderen Hunderassen getrennt wird.  

Die Projektidee orientiert sich an den Inhalten aus den Vorlesungen zu CNNs und Transfer Learning: Statt ein Netz vollständig neu zu trainieren, wird ein bereits auf ImageNet vortrainiertes Modell verwendet, dessen letzter Klassifikationskopf auf die Zielklassen angepasst und anschließend feinjustiert wird. Dieses Vorgehen reduziert den Datenbedarf, beschleunigt das Training und verbessert typischerweise die Generalisierungsfähigkeit bei kleineren Zieldatensätzen.   

## Related Work und Datensatz

### Related Work

Die Vorlesung zu CNN-Architekturen behandelt zentrale Bildklassifikationsmodelle wie LeNet, AlexNet, VGG, Inception und ResNet und zeigt, dass moderne Architekturen erhebliche Leistungsunterschiede hinsichtlich Genauigkeit, Parametern und Tiefe aufweisen. ResNet ist dabei besonders relevant, weil Residual-Verbindungen das Training tiefer Netze erleichtern und sich als Standard-Backbone für viele Bildklassifikationsaufgaben etabliert haben.  

Die Transfer-Learning-Vorlesung empfiehlt für spezialisierte Bildklassifikation ausdrücklich die Nutzung vortrainierter Modelle mit anschließendem Austausch des Klassifikationskopfs und Fine-Tuning auf dem Zieldatensatz. Als Auswahlkriterien für das Backbone werden insbesondere Aufgabenähnlichkeit, verfügbare vortrainierte Gewichte, Architekturgröße und vorhandene Rechenressourcen genannt.  

### Datensatz

Der Trainings- und Testdatensatz besteht aus Hundebildern, die in Klassenordnern organisiert sind, sowie einer separaten zusätzlichen Klasse **Garo**. Verwendet werden die Ordner `train/`, `test/`, `train_garo/` und `test_garo/`, sodass die regulären Hundearten und die eigene Haustier-Klasse gemeinsam verarbeitet werden können. Die Labelzuordnung erfolgt über die Ordnerstruktur beziehungsweise bei Garo über die separaten Garo-Ordner.  

Für die Experimente werden insgesamt 121 Klassen verwendet. Die Datensatz-Zusammenfassung (`dataset_summary.json`) dokumentiert die Anzahl der Hundeklassen, die Gesamtzahl der Klassen inklusive Garo sowie die Verteilung der Bilder auf Trainings- und Testdaten. Diese saubere Trennung von Train-, Validation- und Testdaten entspricht den Empfehlungen des Metriken-Skripts, das eine strikte Datentrennung für valide Evaluation fordert.  

## Vorgehen

### Datenvorbereitung

Die Bilder werden in einem benutzerdefinierten PyTorch-Dataset geladen und in ein einheitliches Format überführt. Für das Training kommen zusätzlich Data-Augmentation-Verfahren wie horizontales Spiegeln, leichte Rotation und ColorJitter zum Einsatz, während die Validierungs- und Testbilder nur skaliert und normalisiert werden. Dieses Vorgehen verbessert die Robustheit gegenüber kleinen Bildvariationen und folgt dem in der Transfer-Learning-Vorlesung empfohlenen Trainingssetup mit Augmentierung und sauberem Split.  

### Modell 1: ResNet18

Als erstes Hauptmodell wird ein auf ImageNet vortrainiertes **ResNet18** eingesetzt. Dabei wird der ursprüngliche Klassifikationskopf entfernt und durch einen neuen Fully-Connected-Layer mit 121 Ausgängen ersetzt, damit das Netz direkt auf die Zielklassen angepasst ist.   

Das Training erfolgt in zwei Phasen:
- **Head-Training:** Zunächst wird der Backbone eingefroren und nur der neue Klassifikationskopf trainiert.
- **Fine-Tuning:** Danach werden alle Layer wieder freigegeben und das gesamte Netz mit kleinerer Lernrate feinjustiert.  

### Modell 2: Vergleichsmodell

Zusätzlich wird ein zweites vortrainiertes CNN trainiert, um die Architekturfrage experimentell zu untersuchen. Dafür wird im selben Projekt ein separates Vergleichsmodell mit identischem Dateninput, identischem Split und identischer Evaluationslogik trainiert, sodass die Ergebnisse fair mit dem ResNet18 verglichen werden können. Die Metriken-Vorlesung betont ausdrücklich, dass faire Modellvergleiche nur unter gleichen Testbedingungen aussagekräftig sind.  

### Evaluation

Die Auswertung basiert auf mehreren komplementären Metriken:
- **Accuracy** zur Gesamtgenauigkeit,
- **Precision**, **Recall** und **F1-Score** pro Klasse,
- **Macro-F1** und **Weighted-F1** als aggregierte Mehrklassenmetriken,
- **Confusion Matrix** zur Analyse systematischer Verwechslungen.  

Die Konfusionsmatrix ist laut Metriken-Skript die zentrale Grundlage jeder Klassifikationsauswertung, weil sich daraus sowohl korrekte Klassifikationen auf der Diagonalen als auch typische Fehlklassifikationen zwischen ähnlichen Klassen ablesen lassen. Gerade bei vielen Hundeklassen und einer zusätzlichen Haustier-Klasse ist diese Darstellung besonders wichtig.  

## Ergebnisse und Auswertung

Die Ergebnisse werden in den Output-Ordnern gespeichert und umfassen sowohl numerische Metriken als auch grafische Darstellungen. Insbesondere werden `metrics_summary.json`, `classification_report.json`, `prediction_overview.csv`, `confusion_matrix_121.png`, `confusion_matrix_small.png` und `garo_analysis.json` erzeugt. Diese Dateien erlauben sowohl eine globale Bewertung des Modells als auch eine fokussierte Analyse der Klasse Garo.  

### Wichtige Ausgabedateien

| Datei | Inhalt |
|------|--------|
| `dataset_summary.json` | Anzahl Klassen, Bildverteilung, Datensatzübersicht |
| `train_history.json` | Verlauf von Train-Loss, Train-Accuracy, Val-Loss und Val-Accuracy |
| `metrics_summary.json` | Accuracy, Macro-F1, Weighted-F1, Anzahl Testbilder |
| `classification_report.json` | Precision, Recall, F1-Score und Support pro Klasse |
| `confusion_matrix_121.png` | Große Konfusionsmatrix über alle 121 Klassen |
| `confusion_matrix_small.png` | Reduzierte Konfusionsmatrix über tatsächlich vorkommende Testklassen |
| `prediction_overview.csv` | Einzelvorhersagen pro Bild mit Wahrheitslabel, Vorhersage und Konfidenz |
| `garo_analysis.json` | Spezifische Auswertung der Garo-Klasse |

### Interpretation der Ergebnisse

Für die Interpretation gilt: Eine hohe Accuracy allein reicht nicht aus, insbesondere bei vielen Klassen oder ungleich verteilten Daten. Deshalb werden zusätzlich Precision, Recall und F1-Score betrachtet, um sowohl die Zuverlässigkeit als auch die Vollständigkeit der Klassenerkennung zu bewerten. Das Metriken-Skript empfiehlt ausdrücklich, Accuracy nie isoliert zu berichten, sondern immer zusammen mit Konfusionsmatrix und F1-basierten Kennzahlen.  

Besonderes Augenmerk liegt auf der Frage, wie gut **Garo** von den übrigen Hunderassen unterschieden werden kann. Dazu werden gezielt die Testbilder der Garo-Klasse betrachtet und analysiert, ob das Modell diese korrekt erkennt oder mit bestimmten Hunderassen verwechselt. Diese Form der Fehleranalyse ist wichtig, um die Stärken und Schwächen des Modells transparent darzustellen.  

### Einbindung von Diagrammen und Bildern

Für die finale Git-Dokumentation sollten mindestens folgende Visualisierungen eingebunden werden:
- Trainingsverlauf aus `train_history.json` als Diagramm für Train- und Validation-Accuracy,
- die große und kleine Konfusionsmatrix,
- Beispielvorhersagen aus `prediction_overview.csv`,
- optional Beispielbilder korrekter und falscher Klassifikationen.  

Falls Poster, Screenshots oder Vergleichsgrafiken vorliegen, können sie direkt in dieses Repository eingebunden werden, um die Projektergebnisse visuell nachvollziehbar zu machen. Gerade bei Computer-Vision-Projekten sind qualitative Beispiele laut Metriken-Skript eine sinnvolle Ergänzung zu den reinen Zahlenwerten.  


## Projektstruktur

```text
.
├── dataset.py
├── train.py
├── evaluate.py
├── evaluate_small.py
├── dataset_cmp.py
├── train_cmp.py
├── evaluate_cmp.py
├── evaluate_small_cmp.py
├── train/
├── test/
├── train_garo/
├── test_garo/
├── output_dog_project/
└──  output_dog_project_compare/


## Ausführung

### ResNet18 trainieren

```bash
python train.py
```

### ResNet18 evaluieren

```bash
python evaluate.py
python evaluate_small.py
```

### Vergleichsmodell trainieren

```bash
python train_cmp.py
```

### Vergleichsmodell evaluieren

```bash
python evaluate_cmp.py
python evaluate_small_cmp.py
```

## Fazit

Das Projekt demonstriert die Nutzung von Transfer Learning für eine mehrklassige Hundearten-Erkennung mit zusätzlicher Haustier-Klasse. Durch den direkten Vergleich zweier vortrainierter CNN-Architekturen unter identischen Bedingungen lassen sich Unterschiede in Generalisierung, Fehlermustern und Erkennungsleistung systematisch analysieren. Genau diese Kombination aus Architekturvergleich, sauberer Evaluation und nachvollziehbarer Fehleranalyse entspricht den methodischen Empfehlungen der Kursunterlagen.    