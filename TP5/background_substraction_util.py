import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import naive_background_substraction as naive

BS_NAIVE = 0
BS_MOG2 = 1
BS_KNN = 2
BS_KNN_SH = 3

def main_track(filename, metodo = BS_NAIVE, N=10, interval=2):
    """
    Genera las foreground masks para el video (filename) usando el método indicado.
    N: número de frames para el background (solo si el método es Naive)
    interval: Intervalo de tiempo para actualizar el background (solo si el método es Naive)
    """

    capture = cv.VideoCapture(filename)
    if int(capture.get(5)) == 0:
        print('Falla al abrir el archivo: ' + filename)
        exit(0)

    if metodo == BS_NAIVE:
        backSub = naive.naiveBackgroundSubstraction(capture, N, interval)
    elif metodo == BS_KNN:
        backSub = cv.createBackgroundSubtractorKNN()
    elif metodo == BS_KNN_SH:
        backSub = cv.createBackgroundSubtractorKNN(detectShadows = True)
    else:
        backSub = cv.createBackgroundSubtractorMOG2()

    cv.namedWindow("Frame", cv.WINDOW_NORMAL); cv.resizeWindow('Frame', 640, 480)
    cv.namedWindow("FG Mask", cv.WINDOW_NORMAL); cv.resizeWindow('FG Mask', 640, 480)
    cv.moveWindow('Frame', 0, 0)
    cv.moveWindow('FG Mask', 720, 0)


    # Corremos la sustraccion
    while True:
        # Lee un frame
        ret, frame = capture.read()

        if frame is None:
            break
        
        # Aplicam la sustracción al frame leído
        # Cada frame se utiliza tanto para calcular la máscara de primer plano como para actualizar el fondo.
        fgMask = backSub.apply(frame)
        
        # Escribe sobre la imagen el número de frame procesado
        cv.rectangle(frame, (10, 2), (100, 50), (0, 0, 0), -1)
        cv.putText(frame, f'{int(capture.get(cv.CAP_PROP_POS_FRAMES)):4}', (30, 35)
                   , cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), thickness=1, lineType=cv.LINE_AA)

        # Muestra frame original e imagen binaria background/foreground
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)
        
        # Ejecuta hasta que termine o se aprieta escape
        keyboard = cv.waitKey(10)
        if keyboard == 'q' or keyboard == 27:
            break

    cv.destroyAllWindows()
    capture.release()

def compare_track(filename, N=10, interval=2):
    """
    Dado un video, compara la salida de los algoritmos KNN, MOG2 y Naive
    N: número de frames para el background (para el método Naive)
    interval: Intervalo de tiempo para actualizar el background (para el método Naive)
    """

    capture = cv.VideoCapture(filename)
    if int(capture.get(5)) == 0:
        print('Falla al abrir el archivo: ' + filename)
        exit(0)

    backSubNaive = naive.naiveBackgroundSubstraction(capture, N, interval)
    backSubKNN = cv.createBackgroundSubtractorKNN()
    backSubMOG2 = cv.createBackgroundSubtractorMOG2()

    cv.namedWindow("Original", cv.WINDOW_NORMAL); cv.resizeWindow('Original', 480, 360)
    cv.namedWindow("Naive", cv.WINDOW_NORMAL); cv.resizeWindow('Naive', 480, 360)
    cv.namedWindow("KNN", cv.WINDOW_NORMAL); cv.resizeWindow('KNN', 480, 360)
    cv.namedWindow("MOG2", cv.WINDOW_NORMAL); cv.resizeWindow('MOG2', 480, 360)

    cv.moveWindow('Original', 0, 0)
    cv.moveWindow('Naive', 500, 0)
    cv.moveWindow('KNN', 0, 400)
    cv.moveWindow('MOG2', 500, 400)

    # Corre la sustracción
    while True:
        # Lee un frame
        ret, frame = capture.read()

        if frame is None:
            break
        
        # Aplica la sustracción al frame leído
        # Cada frame se utiliza tanto para calcular la máscara de primer plano como para actualizar el fondo.

        fgMask_Naive = backSubNaive.apply(frame)
        fgMask_KNN = backSubKNN.apply(frame)
        fgMask_MOG2 = backSubMOG2.apply(frame)
        
        # Escribe sobre la imagen el número de frame procesado
        cv.rectangle(frame, (10, 2), (100, 50), (0, 0, 0), -1)
        cv.putText(frame, f'{int(capture.get(cv.CAP_PROP_POS_FRAMES)):4}', (30, 35)
                   , cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), thickness=1, lineType=cv.LINE_AA)

        # Muestra frame original e imagen binaria background/foreground
        cv.imshow('Original', frame)
        cv.imshow('Naive', fgMask_Naive)
        cv.imshow('KNN', fgMask_KNN)
        cv.imshow('MOG2', fgMask_MOG2)
        
        # Ejecuta hasta que termine o se aprieta escape
        keyboard = cv.waitKey(10)
        if keyboard == 'q' or keyboard == 27:
            break

    cv.destroyAllWindows()
    capture.release()