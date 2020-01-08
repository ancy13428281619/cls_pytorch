# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
from toolbox import fileinterface as FIL
from toolbox import imgproctool as IPT


class CropBgNgSmallPicByROIMask:
    def __init__(self):

        # 需要设置的参数
        self._roi_yaml_path = '/home/pi/PycharmProjects/tst/roi.yaml'
        self._imgs_path = '/home/pi/PycharmProjects/tst/625'
        self._crop_roi_size = (256, 256)
        self._overlap_row = 250
        self._overlap_col = 250
        self._save_bg_img_dir = '/home/pi/Desktop/data/train/bg'
        self._save_ng_img_dir = '/home/pi/Desktop/data/train/ng'
        self._area_pixel = 200

        os.makedirs(self._save_bg_img_dir, exist_ok=True)
        os.makedirs(self._save_ng_img_dir, exist_ok=True)

        # 每个视角下图片的roi信息
        # {view_point: list}
        # {'25': [46, 222, 2442, 1822]}
        self._rois_dict_xyxy = FIL.loadYaml(self._roi_yaml_path)

    @staticmethod
    def overlap_crop(image, roi_size=(256, 256), overlap_row=30, overlap_col=30):
        """划窗裁剪"""
        assert overlap_row < roi_size[0] and overlap_col < roi_size[1]
        assert image.shape[0] >= roi_size[0] and image.shape[1] >= roi_size[1]

        rois_xywh_list = []
        rois_imgs_list = []
        stride_row = roi_size[0] - overlap_row
        stride_col = roi_size[1] - overlap_col

        for row in range(0, image.shape[0] + 1, stride_row):
            for col in range(0, image.shape[1] + 1, stride_col):
                row_start = row if row + roi_size[0] < image.shape[0] else image.shape[0] - roi_size[0]
                col_start = col if col + roi_size[1] < image.shape[1] else image.shape[1] - roi_size[1]
                rois_xywh_list.append([row_start, col_start, roi_size[0], roi_size[1]])
                rois_imgs_list.append(image[row_start:row_start + roi_size[0], col_start:col_start + roi_size[1]])

        return rois_imgs_list, rois_xywh_list

    @staticmethod
    def _is_needed_img(img_name):
        """根据图片名字判断是否为需要的图片"""
        if img_name.endswith('.png') and not img_name.endswith('_mask.png'):
            return True
        return False

    def get_img_mask_path(self):
        """获得原图与mask的路径
            如没有mask，会自动生成mask图片
            imgs_info_list: 'UL_0'
        """
        all_imgs_name = os.listdir(self._imgs_path)

        imgs_info_list, imgs_path_list, masks_path_list = [], [], []

        for one_img_name in all_imgs_name:
            one_img_path = os.path.join(self._imgs_path, one_img_name)
            if self._is_needed_img(one_img_name):
                transformer_type, _, view_point, *_ = one_img_name.split('_')
                imgs_info_list.append(transformer_type + '_' + view_point)
                imgs_path_list.append(one_img_path)
                one_mask_path = one_img_path.replace('.png', '_mask.png')
                if not os.path.isfile(one_mask_path):
                    mask = np.zeros_like(cv2.imread(one_img_path))
                    cv2.imwrite(one_mask_path, mask)
                masks_path_list.append(one_img_path.replace('.png', '_mask.png'))

        return imgs_info_list, imgs_path_list, masks_path_list

    def is_contour_in_image_center(self, mask, offset_range=0.25):
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moment = cv2.moments(contour)
            center_x = int(moment["m10"] / moment["m00"])
            center_y = int(moment["m01"] / moment["m00"])
            if mask.shape[0] * offset_range < center_x < mask.shape[0] * offset_range * 3 and mask.shape[
                1] * offset_range < center_y < mask.shape[1] * offset_range * 3:
                return True
        return False

    def run(self):
        imgs_info_list, imgs_path_list, masks_path_list = self.get_img_mask_path()
        for index in range(len(imgs_path_list)):
            # 获取每张原图需要检测的roi
            roi_xyxy = self._rois_dict_xyxy[imgs_info_list[index]]
            _, roi_img = IPT.getRoiImg(cv2.imread(imgs_path_list[index]), roi_xyxy, IPT.ROI_TYPE_XYXY)
            _, roi_mask = IPT.getRoiImg(cv2.imread(masks_path_list[index]), roi_xyxy, IPT.ROI_TYPE_XYXY)
            # 对roi进行划窗裁剪
            rois_imgs_list, rois_xywh_list = self.overlap_crop(roi_img, self._crop_roi_size, self._overlap_row,
                                                               self._overlap_col)
            rois_mask_list, _ = self.overlap_crop(roi_mask, self._crop_roi_size, self._overlap_row,
                                                  self._overlap_col)

            # 对划窗获得的小roi图片进行判断
            # 若存在缺陷，则mask不为0
            # 当mask> self._area_pixel,保存为ng
            # 当mask= 0,保存为bg
            # 当0< mask < self._area_pixel,不保存
            for roi_idx in range(len(rois_imgs_list)):
                roi_img_name = imgs_path_list[index].split('/')[-1].replace('.png',
                                                                            '_' + str(rois_xywh_list[roi_idx]) + '.png')

                gray = cv2.cvtColor(rois_mask_list[roi_idx], cv2.COLOR_BGR2GRAY)
                if cv2.countNonZero(gray) > self._area_pixel:
                    # cv2.imshow('gray', gray)
                    # cv2.waitKey()
                    # 缺陷需要
                    if self.is_contour_in_image_center(gray):
                        cv2.imwrite(os.path.join(self._save_ng_img_dir, roi_img_name),
                                    rois_imgs_list[roi_idx])

                elif cv2.countNonZero(gray) == 0:
                    # cv2.imwrite(os.path.join(self._save_bg_img_dir, roi_img_name),
                    #             rois_imgs_list[roi_idx])
                    pass

if __name__ == '__main__':
    prj = CropBgNgSmallPicByROIMask()
    prj.run()
