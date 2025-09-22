import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(image_path: str) -> np.ndarray:
    """Загружает изображение и конвертирует в BGR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Не удалось открыть изображение. Проверьте формат файла.")
    return image


def preprocess_image(image: np.ndarray, blur_kernel=(9, 9), threshold_value=None):
    """Преобразует изображение для выделения штрих-кода."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, blur_kernel)

    # Если threshold_value не задан, используем Otsu
    if threshold_value is None:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    return thresh



def find_barcode_contours(thresh, morph_kernel=(21, 7), iterations=4, min_area=5000):
    """Находит контуры возможных штрих-кодов."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=iterations)
    closed = cv2.dilate(closed, None, iterations=iterations)

    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Штрих-код не найден на изображении.")

    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not cnts:
        raise RuntimeError("Подходящих контуров для штрих-кода не найдено.")

    return cnts


def draw_barcodes(image: np.ndarray, cnts, color=(0, 255, 0), thickness=3):
    """Рисует найденные штрих-коды и возвращает список координат прямоугольников."""
    output = image.copy()
    boxes = []

    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        boxes.append(box)
        cv2.drawContours(output, [box], -1, color, thickness)

    return output, boxes


def detect_barcode(
    image_path: str,
    save_path: str = None,
    blur_kernel=(9, 9),
    threshold_value=225,
    morph_kernel=(21, 7),
    iterations=4,
    min_area=5000,
    show=True,
    return_all=True
):
    """
    Основная функция: поиск штрих-кодов.
    Возвращает список координат найденных прямоугольников.
    """
    image = load_image(image_path)
    thresh = preprocess_image(image, blur_kernel, threshold_value)
    cnts = find_barcode_contours(thresh, morph_kernel, iterations, min_area)

    if not return_all:
        cnts = [max(cnts, key=cv2.contourArea)]

    output, boxes = draw_barcodes(image, cnts)

    if save_path:
        cv2.imwrite(save_path, output)

    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Оригинал")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Найденные штрих-коды" if return_all else "Самый большой штрих-код")
        plt.axis("off")
        plt.show()

    return boxes


if __name__ == "__main__":
    try:
        path = input("Введите путь к изображению: ").strip()
        coords = detect_barcode(
            image_path=path,
            save_path="barcode_detected.png",
            threshold_value=200,
            min_area=3000,
            return_all=True  # <<< все найденные штрих-коды
        )
        print("Найденные координаты:", coords)

    except Exception as e:
        print("Ошибка:", e)
