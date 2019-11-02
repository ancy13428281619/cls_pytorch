from __future__ import print_function
import numpy as np
import torch
import yaml
import errno
import os


def overlap_crop(image, shape=(256, 256), overlap_row=0, overlap_col=0, divide_255=False):
    assert overlap_row < shape[0]
    assert overlap_col < shape[1]

    row, col = image.shape[:2]
    map = np.zeros(image.shape[:2], dtype=np.float32)

    if row <= shape[0] and col <= shape[1]:
        if divide_255:
            image = np.array(image, np.float32) / 255.0
        return [image], [[0, 0]], np.ones(image.shape[:2], dtype=np.float32)

    piece_list = []
    pts_list = []
    stride_row = shape[0] - overlap_row
    stride_col = shape[1] - overlap_col

    for row_n in range(0, row - shape[0] + stride_row, stride_row):
        for col_n in range(0, col - shape[1] + stride_col, stride_col):

            row_start = row_n
            row_end = row_n + shape[0]
            col_start = col_n
            col_end = col_n + shape[1]
            if row_n + shape[0] > row:
                row_start = row - shape[0]
                row_end = row
            if col_n + shape[1] > col:
                col_start = col - shape[1]
                col_end = col

            piece = image[row_start:row_end, col_start:col_end]
            map[row_start:row_end, col_start:col_end] += 1
            pts = [row_start, col_start]
            if divide_255:
                piece = np.array(piece, np.float32) / 255.0
            piece_list.append(piece)
            pts_list.append(pts)

    return piece_list, pts_list, map


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def loadYaml(fileName, method='r'):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    """
    with open(fileName, method) as f:
        return yaml.load(stream=f, Loader=yaml.FullLoader)


def set_gpu(config):
    # gpu使用
    assert isinstance(config['Misc']['GpuId'], int), '请输入正确的gpu id'
    if config['Misc']['GpuId'] in [i for i in range(torch.cuda.device_count())]:
        device = torch.device("cuda:{}".format(config['Misc']['GpuId']))
        torch.backends.cudnn.benchmark = True
        print('use gpu:{}'.format(config['Misc']['GpuId']))
    else:
        device = torch.device("cpu")
        print('use cpu')
    return device
