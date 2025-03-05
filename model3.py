import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import PIL.Image
import os
import torch.nn.functional as F
from collections import Counter
import numpy as np
import sys
import pickle


dataset_path = "C:/Users/xxx/xxx/model/archive/train/train"


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])


try:
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
except Exception as e:
    print(f"⚠️ Fel vid laddning av dataset: {e}")
    sys.exit(1)


with open("class_names.pkl", "wb") as f:
    pickle.dump(full_dataset.classes, f)
print("✅ Sparade klasserna till 'class_names.pkl'!")


train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


print("Classes found:", full_dataset.classes)
print("Number of images:", len(full_dataset))
label_counts = Counter([label for _, label in full_dataset.samples])
print("Label distribution:", label_counts)


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtClassifier, self).__init__()
        
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
       
        for param in self.convnext.parameters():
            param.requires_grad = False
        
        
        self.convnext.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.convnext(x)  


num_classes = len(full_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtClassifier(num_classes).to(device)


if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("✅ Laddade tidigare vikter från 'best_model.pth'!")


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00008)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


def train_model():
    epochs = 15
    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = 3

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} påbörjad...")

        
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

       
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_train_loss = running_loss / len(trainloader)
        avg_val_loss = val_loss / len(valloader)

        print(f"Epoch {epoch+1} klar! Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Sparade bästa modellen till 'best_model.pth'!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"🏁 Early stopping aktiverad efter {epoch+1} epochs!")
                break

        scheduler.step()


def predict_image(image_path):
    try:
        image = PIL.Image.open(image_path).convert("RGB")  
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"⚠️ Fel vid laddning av bild: {e}")
        return

    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    except FileNotFoundError:
        print("⚠️ Modellfilen 'best_model.pth' saknas! Träna modellen först.")
        return

    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        top_prob, top_class = torch.max(probabilities, dim=0)

    print(f"🔮 Predicted class: {full_dataset.classes[top_class.item()]} (Confidence: {top_prob.item() * 100:.2f}%)")
    print("\n📊 **Sannolikhet för varje klass:**")
    for i, prob in enumerate(probabilities):
        print(f"{full_dataset.classes[i]}: {prob.item() * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1]) 
    else:
        train_model()  
