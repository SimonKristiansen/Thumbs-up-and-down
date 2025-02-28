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

__________________________________________________________________________________________________________________________________


ConvNeXt Image Classification Model
This project implements an image classification model based on the ConvNeXt Tiny architecture from PyTorch's torchvision. The model is pre-trained on ImageNet and fine-tuned to classify images from a custom dataset. 
The project includes data processing, training with early stopping, and prediction on individual images.

Functions
Data Augmentation: Advanced reinforcement with rotation, flip, Gaussian blur, etc.

Training: Uses the AdamW optimizer and CrossEntropyLoss with scheduled learning rate.

Validation: Evaluates the model's performance on a validation dataset after each epoch.

Early Stopping: Stops the training early if the validation loss does not improve.

Prediction: Classifies individual images with probability distribution over classes.

Model Saving: Saves the best model based on validation loss.

Prerequisites
Python 3.8+

PyTorch (torch, torchvision)

PIL (Pillow)

NumPy

CUDA (optional, for GPU acceleration)

Install dependencies with:


pip install torch torchvision pillow numpy
Dataset
The project expects a dataset in the following structure:


dataset_path/
    class1/
        image1.jpg
        image2.jpg
        ...
    class2/
        image1.jpg
        image2.jpg
        ...
The default path is C:/Users/xxx/xxx/xxx/archive/train/train, but this can be changed in the code (dataset_path).

Usage
Train the Model
Run the script without arguments to start training:


python script.py
The training runs in 15 epochs (configurable) with a batch size of 16. The dataset is automatically divided into 80% training and 20% validation. The best model is saved as best_model.pth. The class names are saved as class_names.pkl for later use (e.g., in a GUI).

2. Predict an Image
Specify an image path as an argument to classify an image:


python script.py path/to/image.jpg
The model loads the weights from best_model.pth. The output shows the predicted class with confidence as well as probabilities for all classes.

Example Output

🔮 Predicted class: dog (Confidence: 92.34%)
📊 Probability for each class:
cat: 6.12%
dog: 92.34%
bird: 1.54%
File Structure
script.py: The main script for training and prediction.

best_model.pth: Saved model weights (created during training).

class_names.pkl: List of class names (created during training).

Configuration
Model: ConvNeXt Tiny (pre-trained on ImageNet).

Learning Rate: 0.00008 (AdamW).

Scheduler: StepLR (halves lr every 3rd epoch).

Batch Size: 16.

Early Stopping: Activated after 3 epochs without improvement.

Customization
Change dataset_path to your own data folder.

Adjust transform to change data augmentation.

Modify num_classes dynamically based on the dataset.

Change training parameters like epochs, lr, or patience in train_model().

Troubleshooting
"Dataset loading error": Check that dataset_path is correct and that the images are in the right format (JPG/PNG).

"Model file missing": Train the model first to generate best_model.pth.

GPU problems: If CUDA does not work, the code automatically checks and falls back to CPU.
