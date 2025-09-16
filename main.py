import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = input("Введите путь к изображению: ")

# Загружаем изображение
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Считаем градиенты по x и y
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# Вычитаем (штрих-код вытянут по x → он проявится сильнее)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# Размытие и бинаризация
blurred = cv2.blur(gradient, (9, 9))
_, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# Морфологическое закрытие (соединяем полоски штрих-кода в одну область)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Немного эрозий и дилатаций (убираем шум)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# Находим контуры и выбираем самый большой
cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# Ограничивающий прямоугольник
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int32(box)

# Рисуем на копии изображения
output = image.copy()
cv2.drawContours(output, [box], -1, (0, 255, 0), 3)

# Вывод
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Оригинал")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Найденный штрих-код")
plt.axis("off")

plt.show()
