ConvNeXt Bildklassificeringsmodell
Detta projekt implementerar en bildklassificeringsmodell baserad på ConvNeXt Tiny-arkitekturen från PyTorchs torchvision. Modellen är förtränad på ImageNet och finjusteras för att klassificera bilder från ett anpassat dataset. 
Projektet inkluderar databehandling, träning med early stopping och prediktion på enskilda bilder.

Funktioner
Data Augmentation: Avancerad förstärkning med rotation, flip, Gaussian blur, etc.
Träning: Använder AdamW-optimeraren och CrossEntropyLoss med schemalagd inlärningshastighet.
Validering: Utvärderar modellens prestanda på ett valideringsdataset efter varje epoch.
Early Stopping: Stoppar träningen tidigt om valideringsförlusten inte förbättras.
Prediktion: Klassificerar enskilda bilder med sannolikhetsfördelning över klasser.
Modellsparning: Sparar bästa modellen baserat på valideringsförlust.
Förutsättningar
Python 3.8+
PyTorch (torch, torchvision)
PIL (Pillow)
NumPy
CUDA (valfritt, för GPU-acceleration)
Installera beroenden med:


pip install torch torchvision pillow numpy
Dataset
Projektet förväntar sig ett dataset i följande struktur:


dataset_path/
    class1/
        image1.jpg
        image2.jpg
        ...
    class2/
        image1.jpg
        image2.jpg
        ...
Standardvägen är C:/Users/xxx/xxx/xxx/archive/train/train, men detta kan ändras i koden (dataset_path).

Användning
1. Träna modellen
Kör skriptet utan argument för att starta träning:


python script.py
Träningen körs i 15 epoker (konfigurerbart) med en batchstorlek på 16.
Datasetet delas automatiskt i 80% träning och 20% validering.
Den bästa modellen sparas som best_model.pth.
Klassnamnen sparas som class_names.pkl för senare användning (t.ex. i ett GUI).
2. Prediktera en bild
Ange en bildsökväg som argument för att klassificera en bild:


python script.py path/to/image.jpg
Modellen laddar vikterna från best_model.pth.
Utdata visar den predikterade klassen med konfidens samt sannolikheter för alla klasser.
Exempelutdata

🔮 Predicted class: dog (Confidence: 92.34%)
📊 **Sannolikhet för varje klass:**
cat: 6.12%
dog: 92.34%
bird: 1.54%
Filstruktur
script.py: Huvudskriptet för träning och prediktion.
best_model.pth: Sparade modellvikter (skapas vid träning).
class_names.pkl: Lista över klassnamn (skapas vid träning).
Konfiguration
Modell: ConvNeXt Tiny (förtränad på ImageNet).
Inlärningshastighet: 0.00008 (AdamW).
Schemaläggare: StepLR (halverar lr var 3:e epoch).
Batchstorlek: 16.
Early Stopping: Aktiveras efter 3 epoker utan förbättring.
Anpassning
Ändra dataset_path till din egen datamapp.
Justera transform för att ändra data augmentation.
Modifiera num_classes dynamiskt baserat på datasetet.
Ändra träningsparametrar som epochs, lr, eller patience i train_model().
Felsökning
"Dataset laddningsfel": Kontrollera att dataset_path är korrekt och att bilderna är i rätt format (JPG/PNG).
"Modellfil saknas": Träna modellen först för att generera best_model.pth.
GPU-problem: Om CUDA inte fungerar, kontrollerar koden automatiskt och faller tillbaka till CPU.
