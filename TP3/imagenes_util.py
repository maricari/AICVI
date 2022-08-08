
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import match_template as mt


def piramide_gaussiana(imagen):
    """
    Construye una pirámide Gaussiana de 7 imágenes a partir de la imagen de entrada
    """

    # pyrDown
    img0 = imagen.copy()
    imgdn1 = cv.pyrDown(img0)
    imgdn2 = cv.pyrDown(imgdn1)
    imgdn3 = cv.pyrDown(imgdn2)

    # PyrUp
    imgup1 = cv.pyrUp(img0)
    imgup2 = cv.pyrUp(imgup1)
    imgup3 = cv.pyrUp(imgup2)

    imagenes = {-3: imgdn3, -2: imgdn2, -1: imgdn1, 
                0: img0,
                1: imgup1, 2: imgup2, 3: imgup3
               }

    return imagenes

class image_generator:
    """
    Genera imagenes a distintas escalas a partir de una imagen dada.
    """
    def __init__(self, imagen, start_percent = 100, scale_percent=2):
        self.imagen = imagen
        self.step = scale_percent/100
        self.pct = start_percent/100
        self.contador = 0

    def next_img(self):
        """
        Cada vez que se invoca retorna una imagen nueva.
        La primera vez, la escala de la imagen será la dada por start_percent. 
            Por ej: si start_percent = 50 la primera imagen estará a una escala de 50% con respecto a la original
        A partir de ahí, cada llamada sucesiva, aumentará la escala en un scale_percent.
        Por ejemplo, para los valores default, las escalas de las imágenes serán:
        100%, 102%, 104%, etc. con respecto a la original.
        """
        img = cv.resize(self.imagen, None, fx= self.pct, fy= self.pct, interpolation= cv.INTER_LINEAR)
        self.pct = self.pct + self.step
        self.contador = self.contador + 1

        return img, self.contador, self.pct


def plot_set_imagenes(imagenes, titulo = "", columnas=4, print_size=True):

    """
    Plotea un set de imágenes
    """
    filas = int(len(imagenes.items())/columnas)+1

    plt.figure(figsize=[columnas * 4, filas * 4])
    plt.suptitle(f'{titulo}', fontsize=16)

    i=0
    for key, img in imagenes.items():
        if print_size:
            subtitulo = f'{key} ({img.shape[0]} x {img.shape[1]})'
        else:
            subtitulo = f'{key}'
        i=i+1
        plt.subplot(filas,columnas, i); plt.imshow(img, cmap='gray'); plt.title(subtitulo)

    plt.tight_layout()
    plt.show()


def plot_template_imagen(imagen):
    
    """
    Plotea el template y la imagen de entrada, side by side, a la misma escala
    """

    fig,axs = plt.subplots(1,2, figsize = (8,6), sharex=True, sharey=True)
    plt.xticks([]), plt.yticks([])
    for idx in [0,1]:
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        axs[idx].spines['bottom'].set_visible(False)
        axs[idx].spines['left'].set_visible(False)

    plt.subplot(1,2,1)
    plt.imshow(mt.template, cmap='gray'); plt.title("Template")
    plt.subplot(1,2,2)
    plt.imshow(cv.cvtColor(imagen, cv.COLOR_BGR2GRAY), cmap='gray'); plt.title("Imagen B&N")

    plt.tight_layout()
    plt.show()


def save_image(nombre_archivo, imagen, dimensiones):

    """
    Hace un resize de una imagen y la guarda en disco
    Parámetros:
    nombre_archivo: destino de la imagen
    imagen: la imagen a guardar
    dimensiones: el size al cual cambiar el tamaño antes de guardar
    """

    dim = (dimensiones[1], dimensiones[0])
    imagen_final = cv.cvtColor(cv.resize(imagen, dim, interpolation = cv.INTER_AREA), cv.COLOR_RGB2BGR)
    cv.imwrite(nombre_archivo, imagen_final)
