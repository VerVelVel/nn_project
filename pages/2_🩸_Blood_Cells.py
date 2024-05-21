import streamlit as st
from PIL import Image
import torch
import json
# import os
import sys
import requests
from io import BytesIO
import time
import torch.nn.functional as F
from pathlib import Path


# Путь к корню проекта относительно текущего файла
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from models.model2.model import ResNet_2
from models.model2.preprocessing import preprocess
import torch
from torch import nn
from torchvision.models import resnet18


class ResNet_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None)
        # заменяем слой
        self.model.fc = nn.Linear(512, 4)
    def forward(self, x):
        return self.model(x)

st.write("# Классификатор картинок по видам клеток крови")

@st.cache_resource
def load_model():
    model = ResNet_2()
    # weights_path = 'models/model2/weights2.pt'  # Убедитесь, что путь к файлу корректен
    state_dict = torch.load('models/model2/weights21.pt')  
    model.model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()

id_class = json.load(open('models/model2/id_class.json'))
id_class = {int(k): v for k, v in id_class.items()}

# Функция для предсказания класса изображения
def predict(image):
    img = preprocess(image)
    with torch.no_grad():
        start_time = time.time()
        preds = model(img.unsqueeze(0))
        end_time = time.time()
    pred_class = preds.argmax(dim=1).item()
    return id_class[pred_class], end_time - start_time

# Загрузка изображения по ссылке
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Загрузка изображения через загрузку файла или по ссылке
def load_image(image):
    if isinstance(image, BytesIO):
        return Image.open(image)
    else:
        return load_image_from_url(image)

# Загрузка изображений и предсказание класса
def predict_images(images):
    predictions = []
    for img in images:
        image = load_image(img)
        prediction, inference_time = predict(image)
        predictions.append((image, prediction, inference_time))
    return predictions

# Отображение изображения и результатов предсказания
def display_results(predictions):
    for img, prediction, inference_time in predictions:
        st.image(img) 
        st.write(f'Prediction: {prediction}')
        st.write(f'Inference Time: {inference_time:.4f} seconds')

# Загрузка изображений через файлы или ссылки
images = st.file_uploader('Upload file(s)', accept_multiple_files=True)

if not images:
    image_urls = st.text_area('Enter image URLs (one URL per line)', height=100).strip().split('\n')
    images = [url.strip() for url in image_urls if url.strip()]

if images:
    predictions = predict_images(images)
    display_results(predictions)