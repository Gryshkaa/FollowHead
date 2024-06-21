import cv2
import numpy as np
import pyautogui

# Настройка цветового диапазона для обнаружения руки
lower_hsv = np.array([0, 20, 70])
upper_hsv = np.array([20, 255, 255])

# Получение разрешения экрана
screen_width, screen_height = pyautogui.size()

# Инициализация видеопотока с веб-камеры
cap = cv2.VideoCapture(0)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success, frame = cap.read()
    if not success:
        break

    # Поворот изображения (OpenCV считывает изображение зеркально)
    frame = cv2.flip(frame, 1)

    # Преобразование изображения в формат HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Создание маски для цвета кожи
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Нахождение контуров в маске
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Выбор контура с наибольшей площадью (вероятно, это рука)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 1000:
            # Нахождение центра контура
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Преобразование координат в экранные координаты
                screen_x = int(cX * screen_width / cap_width)
                screen_y = int(cY * screen_height / cap_height)

                # Перемещение курсора мыши
                pyautogui.moveTo(screen_x, screen_y)

                # Отображение центра контура на изображении
                cv2.circle(frame, (cX, cY), 7, (255, 0, 0), -1)
    
    # Отображение изображения в отдельном окне
    cv2.imshow('Hand Tracking', frame)
    
    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
