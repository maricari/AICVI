import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class naiveBackgroundSubstraction:
    def __init__(self, video_capture, N=10, interval=1):
        """
        Parámetros
        video_capture: el handler del video
        N: cantidad de frames utilizados para la estimación
        interval: el intervalo de tiempo para recalcular el fondo (en segundos)
        """
        self.video_capture = video_capture
        self.frame_count = self.video_capture.get(7)
        self.frame_width = int(self.video_capture.get(3))
        self.frame_height = int(self.video_capture.get(4))
        self.fps = int(self.video_capture.get(5))
        self.bg = None
        self.interval_frames = int(interval * self.fps)
        self.counter = 0
        self.index = 0

        self.kernel = np.ones((3,3),np.uint8)

        self.init_background(N)


    def init_background(self, n_frames):
            """
            Crea el background inicial a partir de n_frames random
            """
            bg = np.zeros([self.frame_width, self.frame_height])
            frames = np.random.randint(0,self.frame_count, n_frames)
            frames = [*set(frames)]
            frames.sort()

            base = np.empty([n_frames, self.frame_width * self.frame_height * 3], dtype=np.uint8)
            i = 0
            for f in frames:
                i = i+1
                self.video_capture.set(cv.CAP_PROP_POS_FRAMES, f-1)
                ret, frame = self.video_capture.read()
                f_flatt = frame.flatten()
                base[:i,:] = f_flatt 
            self.video_capture.set(cv.CAP_PROP_POS_FRAMES, -1)

            flat_bg = np.median(base, axis=0)
            background = flat_bg.reshape(self.frame_height, self.frame_width, 3)
            background = background.astype(np.uint8)
            self.bg = background
            self.base = base

    def update_background(self, frame):
        """
        Actualiza el background.
        (reemplaza el frame más antiguo de los usados en el cálculo anterior. Luego recalcula)
        """

        # reemplaza el frame más antiguo usado para el bg
        f_flatt = frame.flatten()
        self.index = (self.index + 1) % self.base.shape[0]
        # print(f' Index {self.index}')
        self.base[self.index: self.index+1, :] = f_flatt 
        
        # recalcula la mediana
        flat_bg = np.median(self.base, axis=0)

        # actualiza el background
        background = flat_bg.reshape(self.frame_height, self.frame_width, 3)
        background = background.astype(np.uint8)
        self.bg = background


    def background(self):
        return self.bg


    def apply(self, frame):

        # Actualiza el background (si corresponde)
        self.counter = self.counter + 1
        if self.counter == self.interval_frames:
            self.update_background(frame)
            self.counter = 0

        # Calcula la foreground_mask
        a = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        b = cv.cvtColor(self.bg, cv.COLOR_RGB2GRAY)
        fg_mask = abs(a.astype(np.int8) - b.astype(np.int8)).astype(np.uint8)
        ret, fg_mask_bin = cv.threshold(fg_mask,thresh=0.7*np.max(fg_mask),maxval=255,type=cv.THRESH_BINARY)
        fg_mask_bin = cv.dilate(fg_mask_bin, self.kernel, iterations=1)
        
        return fg_mask_bin
