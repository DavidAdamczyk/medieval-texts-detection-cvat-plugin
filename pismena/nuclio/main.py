import base64
from model_loader import ModelLoader
from fastai.vision import *

from helper.object_detection_helper import *
import locale
locale.setlocale(locale.LC_ALL, 'C')
import json
from PIL import Image, ImageFilter
import numpy as np

import torchvision
import torch
import itertools
import math
from torchvision.transforms import ToTensor


def init_context(context):
    context.logger.info("Init context...  0%")
    context.logger.info(f"CUDA: {torch.cuda.is_available()}")
    functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # Read the DL model
    model = ModelLoader()
    setattr(context.user_data, 'model', model)

    setattr(context.user_data, 'labels', labels)
    context.logger.info("Init context...100%")




def get_neighbour_ixs(row, col, store_matrix):
    num_rows = len(store_matrix) - 1
    num_cols = len(store_matrix[0]) - 1

    ixs_row = [row]
    ixs_row_bellow = []
    ixs_col = [col]
    ixs_cols_bellow = [col]
    if row < num_rows:
        ixs_row_bellow.append(row + 1)
    if col > 0:
        ixs_cols_bellow.append(col - 1)
    if col < num_cols:
        ixs_col.append(col + 1)
        ixs_cols_bellow.append(col + 1)
    return list(itertools.product(ixs_row, ixs_col)) + list(itertools.product(ixs_row_bellow, ixs_cols_bellow))


def compare_with_neigh(current, neigh):
    # return current,neigh

    new_neigh, new_current = [], []
    for box2 in neigh:
        do_skip = False
        new_current = []
        for i, box in enumerate(current):
            do_overlap = check_overlap(box, box2, 0.3)
            if do_overlap:
                current[i] = get_merged_box(box, box2)
            do_skip = do_skip or do_overlap
        if not do_skip:
            new_neigh.append(box2)

    return current, new_neigh


def check_overlap(box, box2, thresh):
    def check_if_1_inside_2(first, second):
        y0 = max(first[0], second[0])
        x0 = max(first[1], second[1])
        y1 = min(first[0] + first[2], second[0] + second[2])
        x1 = min(first[1] + first[3], second[1] + second[3])
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        area_of_first = first[2] * first[3]
        proportion = intersection / area_of_first
        if proportion > thresh:
            return True
        else:
            return False

    return check_if_1_inside_2(box, box2) or check_if_1_inside_2(box2, box)


def get_merged_box(box, box2):
    y0 = min(box[0], box2[0])
    x0 = min(box[1], box2[1])
    y1 = max(box[0] + box[2], box2[0] + box2[2])
    x1 = max(box[1] + box[3], box2[1] + box2[3])
    h = y1 - y0
    w = x1 - x0
    box[0], box[1], box[2], box[3] = y0, x0, h, w
    return box


def compare_with_self(current):
    new_current = []
    removed_box_ixs = []
    for i, box in enumerate(current):
        if i in removed_box_ixs:
            continue
        for j, box2 in enumerate(current[i + 1:]):
            do_overlap = check_overlap(box, box2, 0.6)
            if do_overlap:
                removed_box_ixs.append(i + 1 + j)
                box = get_merged_box(box, box2)
        new_current.append(box)
    return new_current


def filter_page_boxes(all_boxes, img):
    _, h, w = img.size()
    cell_size = 100
    store_cells_per_height = math.ceil(h / cell_size) + 1
    store_cells_per_width = math.ceil(w / cell_size) + 1
    # create matrix store so boxes are only compared with neighbour cells
    store_matrix = [[[] for j in range(store_cells_per_width)] for i in range(store_cells_per_height)]
    for box in all_boxes:
        y_index_in_matrix = int(box[0] // cell_size)
        x_index_in_matrix = int(box[1] // cell_size)
        store_matrix[y_index_in_matrix][x_index_in_matrix].append(box)
    for i, row in enumerate(store_matrix):
        for j, current in enumerate(row):
            neighbourgh_ixs = get_neighbour_ixs(i, j, store_matrix)
            for neigh_ix in neighbourgh_ixs:
                neigh = store_matrix[neigh_ix[0]][neigh_ix[1]]
                if (i, j) == neigh_ix:
                    new_current = compare_with_self(current)
                    store_matrix[i][j] = new_current
                    current = new_current
                else:
                    new_current, new_neigh = compare_with_neigh(current, neigh)
                    store_matrix[i][j] = new_current
                    store_matrix[neigh_ix[0]][neigh_ix[1]] = new_neigh

    rows = []
    for sublist in store_matrix:
        rows += sublist

    flat = []
    for r in rows:
        flat += r
    return flat


def handler(context, event):
    #context.logger.info("Poustim labelovani pismen")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    threshold = float(data.get("threshold", 0.5))
    l_img = Image.open(buf)
    width, height = l_img.size
    #context.logger.info(f"Obrazek ma rozmery: {width} {height}")

    l_img = ToTensor()(l_img)
    l_img = l_img[0:3, :, :]
    #context.logger.info(f"Start crop")
    crops, croppos = crop_image(l_img)
    batch = torch.stack(crops)

    #(boxes, scores, classes, num_detections) = context.user_data.model_handler.infer(image)
    #context.logger.info(f"Infer starting")
    bboxes = context.user_data.model.infer(image=batch, context=context)
    #context.logger.info(f"Infer finished")
    #context.logger.info(f"bboxes: {len(bboxes)}")
    all_boxes = []
    for ix, j in enumerate(bboxes):
        all_boxes = all_boxes + moveBoxes(BB=j, ix=ix, croppos=croppos)
    all_boxes = filter_page_boxes(all_boxes,l_img)
    results = []
    context.logger.info(f"Full all_boxes: {len(all_boxes)}")
    #context.logger.info("Compose results")
    q = np.asarray(all_boxes)
    q = np.unique(q, axis=0)
    #context.logger.info(f"q: {len(q)}")
    for i in all_boxes:
        w = i[0]
        h = i[1]
        xtl = h
        ytl = w
        xbr = h + i[3]
        ybr = w + i[2]

        results.append({
            'confidence': str(1),
            'label': 'character',
            'points': [str(int(xtl)), str(int(ytl)), str(int(xbr)), str(int(ybr))],
            'type': 'rectangle'
        })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)


def crop_image(l_img, box_size=256, overlap_x=60, overlap_y=90):
    _, height, width = l_img.shape

    croppos = {}  #
    crop_ix = 0  #

    crops = []
    i = 0
    start_y = 0
    while True:
        start_x = 0
        y0, y1 = start_y, start_y + box_size
        if y1 >= height:
            y0 = height - box_size
            y1 = height
        j = 0
        while True:
            x0, x1 = start_x, start_x + box_size
            if x1 >= width:
                x0 = width - box_size
                x1 = width
            roi = l_img[:, y0:y1, x0:x1]
            crops.append(roi)

            croppos[crop_ix] = [x0, y0]  # [i,j]
            crop_ix += 1  #

            if start_x + box_size >= width:
                break
            start_x = x1 - overlap_x
            j += 1

        if start_y + box_size >= height:
            break
        start_y = y1 - overlap_y
        i += 1
    return crops, croppos


def moveBoxes(BB, ix, croppos):
    w, h = croppos[ix][0], croppos[ix][1]
    boxes = BB
    ret = []
    if type(boxes) != np.ndarray:
        return ret
    for i, b in enumerate(boxes):
        boxes[i][0] += h
        boxes[i][1] += w
        ret.append(boxes[i])
    return ret