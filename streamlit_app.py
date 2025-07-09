import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import speech_recognition as sr

# Paths
TRAIN_DATA_PATH = "dataset/Train"
MODEL_PATH = "crop_disease_model.pth"

# ------------------ MODEL ------------------ #
class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ------------------ TRAIN OR LOAD MODEL ------------------ #
@st.cache_resource
def train_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Training model on all crop disease classes...")
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform)
        classes = dataset.classes
        train_ds, _ = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = CropDiseaseModel(len(classes))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            model.train()
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()

        torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, MODEL_PATH)
        st.success("Model training complete and saved.")

    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model = CropDiseaseModel(len(ckpt['classes']))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['classes']

# ------------------ UTILITIES ------------------ #
def normalize_class_name(cls_name):
    return cls_name.replace('___', ' ').replace('_', ' ').replace('(', '').replace(')', '').strip().title()

def generate_diagnosis_data(classes):
    # Dynamically generated solution dictionary
    generic_solutions = {
        "blight": "Apply fungicides and remove infected areas.",
        "spot": "Use copper-based sprays and maintain field hygiene.",
        "rot": "Prune and destroy infected plant parts, and improve air flow.",
        "rust": "Use resistant varieties and apply protective fungicides.",
        "mildew": "Avoid overhead watering and use sulfur-based sprays.",
        "healthy": "No issues detected. Maintain regular care."
    }

    data = []
    for cls in classes:
        norm_name = normalize_class_name(cls)
        key = norm_name.lower()

        # Dynamically assign a solution
        matched_sol = "General plant care and monitoring recommended."
        for keyword in generic_solutions:
            if keyword in key:
                matched_sol = generic_solutions[keyword]
                break
        if "healthy" in key:
            matched_sol = generic_solutions["healthy"]

        data.append({
            "condition": norm_name,
            "condition_key": key,
            "solution": matched_sol
        })
    return data

def analyze_image(img, model, classes):
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    return normalize_class_name(classes[pred])

def provide_solution(predicted_condition, diagnosis_data):
    key = predicted_condition.lower()
    for entry in diagnosis_data:
        if entry["condition_key"] == key:
            return f"**Diagnosis:** {entry['condition']}\n\n**Solution:** {entry['solution']}"
    return "**Diagnosis:** Unknown\n\n**Solution:** Please consult a local agronomist."

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as src:
        st.info("Listening...")
        try:
            audio = recognizer.listen(src, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except:
            st.warning("Speech recognition failed.")
            return ""

# ------------------ OPTIONAL FALLBACK VOICE ANALYSIS ------------------ #
def analyze_voice(text):
    fallback_conditions = [
        {"condition": "Fungal Infection", "keywords": ["fungus","mold","spots","blight","discolor"], "solution": "Use fungicide and remove diseased parts."},
        {"condition": "Pest Infestation", "keywords": ["pest","worm","aphid","bug","insect"], "solution": "Apply organic pesticide and inspect regularly."},
        {"condition": "Water Stress", "keywords": ["dry","wilting","underwatered","drought"], "solution": "Water regularly to maintain soil moisture."},
        {"condition": "Nutrient Deficiency", "keywords": ["yellow","chlorosis","browning","deficiency"], "solution": "Use balanced fertilizer; test your soil."}
    ]
    t = text.lower()
    for cond in fallback_conditions:
        if any(kw in t for kw in cond["keywords"]):
            return f"**Possible Issue (Based on description):** {cond['condition']}\n\n**Advice:** {cond['solution']}"
    return "Unable to diagnose from description."

# ------------------ STREAMLIT APP ------------------ #
def main():
    st.set_page_config(page_title="Smart Crop Disease Detector")
    st.title("🌿 AI Helper for Farmers")
    st.write("Upload an image to detect disease and get solutions.")

    model, classes = train_and_load_model()
    diagnosis_data = generate_diagnosis_data(classes)

    img_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
    voice_text = ""

    if st.button("🎙 Speak "):
        voice_text = transcribe_speech()

    description = st.text_input("📝 Or type a description (optional)", value=voice_text)

    if st.button("🔍 Diagnose"):
        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            predicted_condition = analyze_image(img, model, classes)
            st.markdown(f"🧠 **Predicted Disease:** `{predicted_condition}`")
            solution = provide_solution(predicted_condition, diagnosis_data)
            st.markdown(solution)
        elif description.strip():
            fallback_result = analyze_voice(description)
            st.markdown(fallback_result)
        else:
            st.warning("Please upload an image or describe the problem.")

if __name__ == "__main__":
    main()
