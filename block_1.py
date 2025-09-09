import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile

st.title("📦 Детекция штрихкодов")
st.write("Загрузи изображение, и обученная YOLOv8 модель найдёт штрихкоды.")

# Загружаем модель (замени путь на свой)
model = YOLO("runs/detect/train/weights/best.pt")

# Форма для загрузки изображения
uploaded_file = st.file_uploader("📤 Загрузите фото со штрихкодом", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Сохраняем во временный файл
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Загружаем картинку через OpenCV
    img = cv2.imread(tfile.name)

    # Запускаем детекцию
    results = model(img)

    # Отрисовываем только рамки
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Конвертируем в RGB для Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="🔎 Результат детекции штрихкодов", use_column_width=True)
