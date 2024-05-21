import streamlit as st
from PIL import Image
import torch
import json
import sys
from pathlib import Path
import requests
from io import BytesIO
import time


st.write("# Классификатор картинок по виду спорта")
st.write("Здесь вы можете загрузить картинку со своего устройства, либо при помощи ссылки")
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from models.model1.model import ResNet_1
from models.model1.preprocessing import preprocess


# Загрузка модели и словаря
@st.cache_resource
def load_model():
    model = ResNet_1()
    weights_path = 'models/model1/weights1.pt'
    state_dict = torch.load(weights_path)
    model.model.fc.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

id_class = json.load(open('models/model1/id_class.json'))
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