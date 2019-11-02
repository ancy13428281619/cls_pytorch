from __future__ import print_function

import os
import torch
import cv2
import numpy as np
from torchvision.models import resnet18
from .lib import utils
from toolbox import imgproctool as IPT
from .lib.config import DF1BConfig
from .lib.logger import get_logger, setup_logger
import time

__all__ = [
    'DF1BDetector'
]


class DF1BDetector:
    def __init__(self):

        cur_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
        self._initLogger(cur_dir)
        self._logger.debug('DF1BDetector 开始初始化...')
        self.img_shape = DF1BConfig.img_shape
        self.overlap_piexl = DF1BConfig.overlap_piexl
        self.imgs_index_dict = DF1BConfig.imgs_index_dict
        self.imgs_index_map_dict = DF1BConfig.imgs_index_map_dict

        self.model = resnet18(num_classes=2).cuda() if torch.cuda.is_available() else resnet18(num_classes=2)

        checkpoint = torch.load(os.path.join(cur_dir, DF1BConfig.model_path), map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self._debug = False
        self._logger.debug('DF1BDetector 初始化结束...')

    def _initLogger(self, logger_path):
        setup_logger(logger_path)
        self._logger = get_logger()

    def preprocessImg(self, img):
        img_float = np.float32(img) / 255.
        img_tensor = torch.from_numpy(img_float)
        img_batch = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        img_batch = img_batch.cuda() if torch.cuda.is_available() else img_batch
        return img_batch

    def _mkDF1BDir(self, images_folder):
        df1b_folder_path = os.path.join(images_folder, 'DF1B')
        utils.mkdir(df1b_folder_path)
        return df1b_folder_path

    def isNeededImg(self, img_name):
        if not img_name.endswith('_1.png'):
            return False
        view_point = img_name.split('_')[2]
        if int(view_point) not in self.imgs_index_map_dict.keys():
            return False
        return True

    def run(self, root_path):
        self._logger.debug('DF1B检测开始...')
        start_time_s = time.time()
        df1b_folder_path = self._mkDF1BDir(root_path)
        for img_name in os.listdir(root_path):
            if self.isNeededImg(img_name):
                box_type, _, view_point = img_name.split('_')[:3]
                img = cv2.imread(os.path.join(root_path, img_name))
                roi_imgs_list, rois_start_xy_list, _ = utils.overlap_crop(img, shape=self.img_shape,
                                                                          overlap_row=self.overlap_piexl,
                                                                          overlap_col=self.overlap_piexl)
                detect_time_s = time.time()
                for index, roi_img in enumerate(roi_imgs_list):
                    view_point_map = [str(self.imgs_index_map_dict[key]) for key in self.imgs_index_map_dict.keys() if
                                      view_point == str(key)][0]
                    if index in self.imgs_index_dict[box_type + view_point_map]:

                        with torch.no_grad():
                            input_img = roi_imgs_list[index]
                            image = self.preprocessImg(input_img)
                            output = self.model(image)
                            _, pred = output.topk(1, 1, largest=True, sorted=True)
                            if pred.cpu().numpy()[0][0] == DF1BConfig.ng_label:
                                rect_xywh = [rois_start_xy_list[index][1], rois_start_xy_list[index][0], 256, 256]
                                IPT.drawRoi(img, rect_xywh, IPT.ROI_TYPE_XYWH, (255, 255, 255), 2)
                                cv2.putText(input_img, 'ng', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                self._logger.debug('检测{}的时间为{},总共：{}张'.format(img_name, time.time() - detect_time_s, index))
                cv2.imwrite(os.path.join(df1b_folder_path, img_name), img)
                if self._debug:
                    cv2.namedWindow('img', 0)
                    cv2.imshow('img', img)
                    cv2.waitKey()
        end_time_s = time.time()
        self._logger.debug('DF1B({})检测总时间为：{}'.format(root_path, end_time_s - start_time_s))
