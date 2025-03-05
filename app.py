import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image
import PIL.ImageDraw
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import pickle

)
class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtClassifier, self).__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in self.convnext.parameters():
            param.requires_grad = False
        self.convnext.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.convnext(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])


def add_rounded_corners(image, radius):
    mask = Image.new("L", image.size, 0)
    draw = PIL.ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0) + image.size, radius, fill=255)
    rounded_image = Image.new("RGBA", image.size)
    rounded_image.paste(image, (0, 0), mask)
    return rounded_image.convert("RGB")


try:
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
except FileNotFoundError:
    print("⚠️ Filen 'class_names.pkl' saknas. Kör träningsprogrammet först för att generera den.")
    class_names = ["Unknown"] 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtClassifier(num_classes=len(class_names)).to(device)


def predict_image(image_path):
    result_text.set("🔍 Analyserar bild...")
    progressbar.start()
    root.update_idletasks()
    
    image = PIL.Image.open(image_path).convert("RGB")  
    image = transform(image).unsqueeze(0).to(device)
    
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    except FileNotFoundError:
        result_text.set("⚠️ Modellfilen 'best_model.pth' saknas!")
        progressbar.stop()
        return
    
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        top_prob, top_class = torch.max(probabilities, dim=0)
    
    progressbar.stop()
    result_text.set(f"✅ Resultat: {class_names[top_class.item()]} ({top_prob.item() * 100:.2f}%)")
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize((300, 300))
    img = add_rounded_corners(img, 30)
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img 


root = ttk.Window(themename="cyborg")
root.title("Bildklassificering med ConvNeXt")
root.geometry("550x650")
root.resizable(False, False)
root.attributes('-alpha', 0.95) 

frame = ttk.Frame(root, bootstyle="dark")
frame.pack(pady=20, padx=20, fill="both", expand=True)

result_text = ttk.StringVar()
result_label = ttk.Label(frame, textvariable=result_text, font=("Poppins", 16, "bold"), bootstyle="light-inverse")
result_label.pack(pady=10)

progressbar = ttk.Progressbar(frame, mode="indeterminate", bootstyle="info-striped")
progressbar.pack(pady=5, fill="x")

img_label = ttk.Label(frame, bootstyle="secondary", relief="solid", padding=5)
img_label.pack(pady=10)


def on_enter(e):
    upload_button.config(bootstyle="success")

def on_leave(e):
    upload_button.config(bootstyle="success-outline")


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Bildfiler", "*.png *.jpg *.jpeg")])
    if file_path:
        predict_image(file_path)

upload_button = ttk.Button(root, text="📂 Ladda upp bild", command=upload_image, bootstyle="success-outline", padding=(10, 5))
upload_button.pack(pady=20)
upload_button.bind("<Enter>", on_enter)
upload_button.bind("<Leave>", on_leave)


root.mainloop()
