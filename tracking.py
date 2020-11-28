from collections import deque
from imutils.video import VideoStream
import time
import numpy as np
import cv2 as cv
import imutils

listlen = 32
# Последовательность в виде списка, оптимизированная для доступа к данным рядом с конечными точками
points = deque(maxlen=listlen)

cap = cv.VideoCapture(0)

time.sleep(2.0)

if not cap.isOpened():
    print('Не получается открыть камеру')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не получается получить видеопоток, что ж выйдем тогда")
        break

    # размер окошка
    frame = imutils.resize(frame, width=300)

    # блюрим окошко
    frame_g_blurred = cv.GaussianBlur(frame, (7, 7), 0)

    # переводим в цветовую модель хсв
    hsv = cv.cvtColor(frame_g_blurred, cv.COLOR_BGR2HSV)
    # hsv = cv.erode(hsv, None, iterations=2)
    # hsv = cv.dilate(hsv, None, iterations=2)

    # нужный нам диапозон зеленого цвета
    lower_color = np.array([58, 45, 25])
    upper_color = np.array([88, 255, 255])

    # получаем диапозон зеленого цвета в кадре
    color_range = cv.inRange(hsv, lower_color, upper_color)
    color_range = cv.erode(color_range, None, iterations=2)
    color_range = cv.dilate(color_range, None, iterations=2)

    # маску побитовую для конечного цвета применим
    # Чтобы наложить маску поверх оригинального изображения, используется cv.bitwise_and().
    # Она сохраняет каждый пиксель изображения, если соответствующее значение маски равно 1
    mask = cv.bitwise_and(frame_g_blurred, frame_g_blurred, mask=color_range)

    # Преобразования цветового пространства
    color_s_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    edge = cv.Canny(color_s_gray, 40, 300)

    #Finds circles in a grayscale image using the Hough transform
    circles = cv.HoughCircles(edge, cv.HOUGH_GRADIENT, dp=1,
                              minDist=500, param1=150, param2=0.9,
                              minRadius=10, maxRadius=200)
    # circles = cv.HoughCircles(edge, cv.HOUGH_GRADIENT_ALT, dp=1,
    #                           minDist=50, param1=300, param2=0.7,
    #                           minRadius=30, maxRadius=300)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # рисование по обнаруженному кругу и его центру
            center = (i[0], i[1])
            # внешнее кольцо
            cv.circle(frame, center, i[2], (0, 255, 0), 2)
            # сам центр
            cv.circle(frame, center, 2, (0, 0, 255), 3)
            points.appendleft(center)
    else:
        if len(points) > 0:
            points.clear()

    # переберем набор отслеживаемых точек
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue

        # вычисляем толщину линии
        thickness = int(np.sqrt(listlen / float(i + 1)) * 3)

        # и соединяем линии
        cv.line(frame, points[i - 1], points[i], (0, 0, 0), thickness)


    # показываем очертания круга на экране серым цветом
    cv.imshow('circle', frame)
    # cv.imshow('gray', color_s_gray)
    # cv.imshow('color_range', color_range)
    cv.imshow('canny', edge)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()

cv.destroyAllWindows()
