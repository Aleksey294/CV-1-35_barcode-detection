# 📦 Barcode Detection with YOLOv8

Проект для автоматической **детекции штрихкодов** на изображениях с использованием **YOLOv8**.  
Включает:
- Скрипт для обучения модели на датасете Seg-Barcode  
- Веб-приложение на **Streamlit** для загрузки и анализа изображений  

## 📂 Датасет  
Для обучения используется датасет **Seg-Barcode**:  
👉 [Seg-Barcode Dataset (Roboflow)](https://universe.roboflow.com/tafila-technichal-university/seg-barcode)  

Формат аннотаций: **YOLOv8 (txt)**  
В датасете есть изображения штрихкодов с разметкой.  

## ⚙️ Установка  

1. Клонируйте репозиторий:  
```bash
git clone https://github.com/yourusername/barcode-detection-yolo.git
cd barcode-detection-yolo
```  

2. Установите зависимости:  
```bash
pip install -r requirements.txt
```  

Минимальные зависимости:  
```
ultralytics
opencv-python
streamlit
numpy
```  

## 🏋️‍♂️ Обучение модели  

1. Скачайте датасет с Roboflow и экспортируйте его в формате **YOLOv8**.  
В папке проекта у вас должно получиться:  
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```  

2. Запустите обучение:  
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640
```  

- `yolov8n.pt` — лёгкая модель (Nano). Можно заменить на `yolov8s.pt`, `yolov8m.pt` и т.д.  
- Результаты сохраняются в папку:  
```
runs/detect/train/
```  

3. После обучения весовая модель будет находиться по пути:  
```
runs/detect/train/weights/best.pt
```  

## 🚀 Запуск приложения  

1. Убедитесь, что модель `best.pt` скопирована в папку проекта (например, `weights/best.pt`).  

2. Запустите Streamlit-приложение:  
```bash
streamlit run app.py
```  

3. Перейдите в браузере:  
```
http://localhost:8501
```  

## 🖼 Использование  

- Нажмите кнопку **"Загрузите фото со штрихкодом"**  
- YOLOv8 проанализирует изображение  
- На выходе вы получите картинку с выделенными **рамками вокруг штрихкодов**  

## 📊 Результаты  

Модель автоматически находит штрихкоды и подсвечивает их **зелёными рамками**.  

Пример работы:  

![Пример детекции](example.png)  

## 📌 Технологии  

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)  
- [OpenCV](https://opencv.org/)  
- [Streamlit](https://streamlit.io/)  
- [NumPy](https://numpy.org/)  

## 📖 Полезные ссылки  

- [Документация YOLOv8](https://docs.ultralytics.com/)  
- [Seg-Barcode Dataset (Roboflow)](https://universe.roboflow.com/tafila-technichal-university/seg-barcode)  
