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

## Обучение модели YOLOv8n
```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=52 imgsz=640
```

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

##Модель находится в 
`../runs/detect/train/weights/best.pt`
## 🚀 Запуск приложения  

1. Запустите Streamlit-приложение:  
```bash
streamlit run main.py
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

## 📌 Технологии  

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)  
- [OpenCV](https://opencv.org/)  
- [Streamlit](https://streamlit.io/)  
- [NumPy](https://numpy.org/)  

## 📖 Полезные ссылки  

- [Документация YOLOv8](https://docs.ultralytics.com/)  
- [Seg-Barcode Dataset (Roboflow)](https://universe.roboflow.com/tafila-technichal-university/seg-barcode)  
