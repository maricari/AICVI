import cv2 as cv
import numpy as np
from imagenes_util import set_imagenes

# templates
path_template = 'template/pattern.png'

template = cv.imread(path_template,0)                # original
template_inv = (template * (-1)).astype('uint8')     # invertido

TEMPLATE_ORIGINAL = 1
TEMPLATE_INVERTIDO = 2

IMG_GRAY = 1
IMG_EDGES = 2

METODOS = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


def matchTemplate(img_rgb,
                  metodo=cv.TM_CCORR_NORMED, 
                  cual_imagen=IMG_EDGES,
                  cual_template=TEMPLATE_ORIGINAL):

    """
    Aplica el match template a la imagen de entrada contra el template de Coca Cola.
    Parámetros:
    img_rgb: la imagen contra la cual se aplica el procedimiento
    metodo: método a utilizar
    cual_imagen: IMG_EDGES (default) usa la imagen de bordes (Canny)
                 IMG_GRAY usa la imagen en escala de grises
    cual_template: TEMPLATE_ORIGINAL (default) usa el template de Coca Cola original
                   TEMPLATE_INVERTIDO usa el template con los colores invertidos

    Retorna:
        el resultado de aplicar el método
        una imagen de salida con el boundind box (si se detectó un match)
        los valores de intensidad máximo y mínimo detectados
    """

    if cual_imagen == IMG_EDGES:
        edges = cv.Canny(img_rgb,40,105,L2gradient=True)
        img_match = edges
    else:
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        img_match = img_gray
    
    if cual_template == TEMPLATE_INVERTIDO:
        pattern = template_inv
    else:
        pattern = template

    # Si el template es mas grande, lo achico
    scale_percent_w = img_rgb.shape[1]*100/pattern.shape[1]
    scale_percent_h = img_rgb.shape[0]*100/pattern.shape[0]
    if (scale_percent_h < 100 or scale_percent_w < 100):
        scale_percent = (scale_percent_h - 10) if scale_percent_h < scale_percent_w else (scale_percent_w - 10)
        width = int(pattern.shape[1] * scale_percent / 100)
        height = int(pattern.shape[0] * scale_percent / 100)
        dim = (width, height)
        pattern = cv.resize(pattern, dim, interpolation = cv.INTER_AREA) 

    img_salida = img_rgb.copy() # la imagen para visualizar
    
    # Aplicamos la coincidencia de patrones
    res = cv.matchTemplate(img_match, pattern, metodo)

    # Encontramos los valores máximos y mínimos
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    # Si el método es TM_SQDIFF o TM_SQDIFF_NORMED, tomamos el mínimo
    if metodo in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    # Marcamos el lugar donde lo haya encontrado
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    pos_text = (top_left[0] + 30, top_left[1] - 15)
    cv.rectangle(img_salida,top_left, bottom_right, (0,255,0), thickness = 5, lineType = cv.FILLED)
    img_salida = cv.putText(img_salida, f'NC: {max_val:.4}', org=pos_text, fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(0,255,0), thickness=4, lineType=cv.LINE_AA)    
                
    return res, img_salida, (min_val, max_val)



def non_max_suppression(boxes, overlapThresh):

	# Algoritmo NON MAX SUPPRESSION
    # Fuente: https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]


def matchTemplateMultiParam(img_rgb,
                  metodo=cv.TM_CCORR_NORMED, 
                  cual_imagen=IMG_EDGES,
                  cual_template=TEMPLATE_ORIGINAL,
                  umbral_coincidencia = 0.8,
                  umbral_superposicion = 0.4):

    """
    Aplica el match template para detecciones múltiples
    Parámetros:
    img_rgb: la imagen contra la cual se aplica el procedimiento
    metodo: método a utilizar
    cual_imagen: IMG_EDGES (default) usa la imagen de bordes (Canny)
                 IMG_GRAY usa la imagen en escala de grises
    cual_template: TEMPLATE_ORIGINAL (default) usa el template de Coca Cola original
                   TEMPLATE_INVERTIDO usa el template con los colores invertidos
    umbral_coincidencia: umbral para decidir si es un match a considerar o no
    umbral_superposicion: umbral para decidir si dos boxes superpuestos deben conservarse
                         o uno debe ser eliminado

    Retorna:
        el resultado de aplicar el método
        una imagen de salida con las boundind boxes (si se detectaron matches)
    """

    if cual_imagen == IMG_EDGES:
        edges = cv.Canny(img_rgb,40,105,L2gradient=True)
        img_match = edges
    else:
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        img_match = img_gray
    
    if cual_template == TEMPLATE_INVERTIDO:
        pattern = template_inv
    else:
        pattern = template

    # Si el template es mas grande, lo achico
    scale_percent_w = img_rgb.shape[1]*100/pattern.shape[1]
    scale_percent_h = img_rgb.shape[0]*100/pattern.shape[0]
    if (scale_percent_h < 100 or scale_percent_w < 100):
        scale_percent = (scale_percent_h - 10) if scale_percent_h < scale_percent_w else (scale_percent_w - 10)
        width = int(pattern.shape[1] * scale_percent / 100)
        height = int(pattern.shape[0] * scale_percent / 100)
        dim = (width, height)
        pattern = cv.resize(pattern, dim, interpolation = cv.INTER_AREA) 

    img_salida = img_rgb.copy() # la imagen para visualizar

    # Aplicamos la coincidencia de patrones
    res = cv.matchTemplate(img_match, pattern, metodo)

    # Construyo un array de bounding boxes por arriba del umbral de coincidencia
    loc = np.where( res >= umbral_coincidencia)
    h, w = template.shape   
    boundingBoxes = []
    for pt in zip(*loc[::-1]):
        boundingBoxes = boundingBoxes + [(pt[0], pt[1], pt[0] + w, pt[1] + h)]
    boundingBoxes = np.array(boundingBoxes)

    # Realizo el non-max suppression
    pick = non_max_suppression(boundingBoxes, umbral_superposicion)

    # print(f'Bounding boxes antes: {len(boundingBoxes)} - ahora: {len(pick)}')

    # Itero sobre los bounding boxes seleccionados y los dibujo en la imagen de salida
    for (startX, startY, endX, endY) in pick:
        cv.rectangle(img_salida, (startX, startY), (endX, endY), (0, 255, 0), thickness = 5, lineType = cv.FILLED)
        pos_text = (startX + 30, startY - 15)
        img_salida = cv.putText(img_salida, f'NC: {res[startY, startX]:.4}', org=pos_text, fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(0,255,0), thickness=4, lineType=cv.LINE_AA)    
         
    return res, img_salida, len(boundingBoxes)

"""
matchTemplateAuto construye un set de imagenes a distintas escalas a partir de una imagen de entrada
y busca detectar el logo de Coca Cola usando diferentes métodos.

"""
escalas = np.append(np.arange(-100, 0, 3), np.arange(400, 0, -10))

opciones = [
     (cv.TM_CCORR_NORMED,  IMG_EDGES, TEMPLATE_ORIGINAL)
    ,(cv.TM_CCORR_NORMED,  IMG_EDGES, TEMPLATE_INVERTIDO)
    ,(cv.TM_CCOEFF_NORMED, IMG_EDGES, TEMPLATE_ORIGINAL)
    ,(cv.TM_CCOEFF_NORMED, IMG_EDGES, TEMPLATE_INVERTIDO)
    ]

def matchTemplateAutoParam(imagenes, mm, ii, tt):

    best_ic = -1
    key_selected = -1
    img_selected = None
    out_selected = None
    for key, img in imagenes.items():
        resultado, salida, valores = matchTemplate(img
                                                  , metodo = mm
                                                  , cual_imagen = ii
                                                  , cual_template = tt)

        if (valores[1] > best_ic and
            (    (valores[1]>0.2 and valores[1] < 0.28 and ii == IMG_EDGES and mm == cv.TM_CCORR_NORMED)
              or (valores[1]>0.5 and valores[1] < 0.8 and ii == IMG_GRAY and mm == cv.TM_CCORR_NORMED)
              or (valores[1]>0.2 and mm != cv.TM_CCORR_NORMED)
                 )
            ):
            best_ic = valores[1]
            key_selected = key             # escala
            img_selected = img.copy()      # imagen
            out_selected = salida.copy()   # imagen con la detección
            params = (mm, ii, tt, img_selected.shape)
        elif (valores[1] < best_ic):
            break

    if key_selected > -1:
        return key_selected, img_selected, out_selected, params, best_ic
    return -1, None, None, None, None


def matchTemplateAuto(img_rgb):
    imagenes = set_imagenes(img_rgb, escalas)
    for mm, ii, tt in opciones:
        key_selected, img_selected, out_selected, parametros, ic = matchTemplateAutoParam(imagenes, mm, ii, tt)
        if key_selected > -1:
            return key_selected, img_selected, out_selected, parametros, ic

    return -1, None, None, None, None


def matchTemplateMulti(img_rgb):
    imagenes = set_imagenes(img_rgb, escalas)
    for key, img in imagenes.items():
        # print(f'key={key}  size={img.shape}')
        resultado, salida, boxes = matchTemplateMultiParam(img, metodo=cv.TM_CCORR_NORMED
                                               , umbral_coincidencia = 0.65
                                               , umbral_superposicion = .25
                                               , cual_imagen=IMG_GRAY
                                               , cual_template=TEMPLATE_INVERTIDO
                                                      )
        if boxes > 0:
            return resultado, salida

    return None, None
