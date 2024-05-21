import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Добро пожаловать на страничку нашего проекта! 👋")

st.sidebar.success("Выберите нужную вам классификацию.")

st.markdown(
    """
    Приложение позволяет классифицировать ваши изображения.

    **👈 Выберите нужную вам классификацию и наша нейросеть постарается вам помочь!**
    ### Что можно найти в этом сервисе?
    - Страницу, позволяющую загрузить пользовательскую спортивную фотографию и получить класс
    - Страницу, позволяющую классифицировать изображение клеток крови
    - Страницу с информацией о:
    - - процессе обучения модели: кривые обучения и метрик
    - - времени обучения
    - - значениях метрики f1 и confusion matrix (в виде heatmap)

    ### Над проектом трудились:
    - [Даша](https://github.com/Dasha0203)
    - [Вера](https://github.com/VerVelVel)
"""
)