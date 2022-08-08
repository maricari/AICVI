import cv2 as cv
import numpy as np
import imagenes_util as img_util

# templates
path_template = 'template/pattern.png'

template = cv.imread(path_template,0)                      # original
template_edges = cv.Canny(template,40,105,L2gradient=True) # edges


def matchTemplateParam(img_rgb, img_match, metodo=cv.TM_CCORR_NORMED):

    """
    Aplica el match template a la imagen de entrada contra el template de Coca Cola.
    Parámetros:
    img_rgb: la imagen original para poder armar la imagen de salida
    img_match: la imagen contra la cual se aplica el procedimiento (por ej con canny aplicado)
    metodo: método a utilizar

    Retorna:
        el resultado de aplicar el método
        una imagen de salida con el boundind box (si se detectó un match)
        los valores de intensidad máximo y mínimo detectados
    """

    pattern = template_edges

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
    top_left = max_loc

    # Marcamos el lugar donde lo haya encontrado
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    pos_text = (top_left[0] + 30, top_left[1] - 15)
    cv.rectangle(img_salida,top_left, bottom_right, (0,255,0), thickness = 5, lineType = cv.FILLED)
    img_salida = cv.putText(img_salida, f'NC: {max_val:.4}', org=pos_text, fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(0,255,0), thickness=4, lineType=cv.LINE_AA)    
                
    return res, img_salida, (min_val, max_val)


def matchTemplate(img_rgb):

    imagenes = img_util.image_generator(img_rgb, 25, 2)

    metodo = cv.TM_CCORR_NORMED
    umbral_sup = 0.161

    best_ic = -1
    img_selected = None
    out_selected = None
    while True:
        img, cc, pct = imagenes.next_img()
        if img is None:
            continue
        if pct > 4:
            break

        img_edges = cv.Canny(img,40,105, L2gradient=True)

        resultado, salida, valores = matchTemplateParam(img, img_edges, metodo)

        if (valores[1] > best_ic
            and (valores[1]>0.1 and valores[1] < umbral_sup)
            ):
            best_ic = valores[1]
            img_selected = img.copy()       # imagen
            out_selected = salida.copy()    # imagen con la detección
        elif (valores[1] < 0.85 * best_ic): # considero que ya es una buena métrica
            break


    if best_ic > -1:
        # print(f'shape {img_selected.shape} best = {best_ic}')
        return img_selected, out_selected, best_ic
    return -1, None, None

"""
DETECCION MULTIPLE
"""

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


def matchTemplateMultiParam(img_rgb
        , img_match
        , metodo=cv.TM_CCORR_NORMED
        , umbral_coincidencia = 0.2
        , umbral_superposicion = 0.4):

    """
    Aplica el match template para detecciones múltiples
    Parámetros:
    img_rgb: la imagen original para poder armar la imagen de salida
    img_match: la imagen contra la cual se aplica el procedimiento (por ej con canny aplicado)
    metodo: método a utilizar
    umbral_coincidencia: umbral para decidir si es un match a considerar o no
    umbral_superposicion: umbral para decidir si dos boxes superpuestos deben conservarse
                         o uno debe ser eliminado

    Retorna:
        el resultado de aplicar el método
        una imagen de salida con las boundind boxes (si se detectaron matches)
    """

    pattern = template_edges

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
    loc = np.where(res >= umbral_coincidencia)

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
         
    return res, img_salida, len(pick)

"""
matchTemplateMulti construye un set de imagenes a distintas escalas a partir de una imagen de entrada
y busca detectar el logo de Coca Cola usando diferentes métodos.
"""

def matchTemplateMulti(img_rgb, start = 25):
    imagenes = img_util.image_generator(img_rgb, start, 2)

    max_boxes = 0
    while True:
        img, cc, pct = imagenes.next_img()
        if img is None:
            continue
        if pct > 4.5:
            break

        img_edges = cv.Canny(img,40,105, L2gradient=True)
        resultado, salida, boxes = matchTemplateMultiParam(img, img_edges
                                        , umbral_coincidencia = 0.15
                                        , umbral_superposicion = .2
                                        )

        if (boxes > max_boxes):
            max_boxes = boxes
            res_selected = resultado
            img_selected = img.copy()       # imagen
            out_selected = salida.copy()    # imagen con la detección
            # print(f'shape {img.shape} boxes = {boxes}')

    if max_boxes > 0:
        return res_selected, out_selected, max_boxes

    return None, None, 0
