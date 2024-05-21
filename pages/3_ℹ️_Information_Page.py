import streamlit as st
import sys
from pathlib import Path

# Определение корневого каталога проекта и добавление его в sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Информация о первой модели
st.write("# Sport Image Classification INFO:")
st.write("Использовалась предобученная модель - ResNet152 с заменой последнего слоя")
st.write("Модель обучалась на предсказание 100 классов")
st.write("Размер тренировочного датасета - 13492 картинки")
st.image(str(project_root / 'images/image1.jpeg'))
st.write("Время обучения модели - 15 эпох = 40 минут, batch_size=32")
st.write("Значения метрики f1 на последней эпохе: 0.695-train и 0.840-valid")
st.write('Confusion matrix')
st.image(str(project_root / 'images/image2.jpeg'))

# Информация о второй модели
st.write("# Blood Cells Classification INFO:")
st.write("Использовалась модель - ResNet18 и обучалась с нуля")
st.write("Модель обучалась на предсказание 4 классов")
st.write("Размер тренировочного датасета - 9957 картинок")
st.image(str(project_root / 'images/image3.png'))
st.write("Время обучения модели - 15 эпох = 40 минут")
st.write("Значения метрики f1 на последней эпохе: 0.915-train и 0.849-valid")
st.write('Confusion matrix')
st.image(str(project_root / 'images/image4.png'))

