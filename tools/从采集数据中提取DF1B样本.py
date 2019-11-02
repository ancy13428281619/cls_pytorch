#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import os
import shutil

root_path = '/media/pi/Elements/PIE电源盒子测试数据/20191101'
output_path = '/home/pi/Desktop/df1b_dataset/20191101'

view_point_list = [i for i in range(12, 20)]
view_point_list.extend([i for i in range(50, 58)])

# 所有样本目录
samplers_dir = os.listdir(root_path)
for sampler_dir in samplers_dir:
    sampler_path = os.path.join(root_path, sampler_dir)

    if os.path.isdir(sampler_path):

        imgs_name = os.listdir(sampler_path)
        for img_name in imgs_name:
            if img_name.endswith('_1.png') and int(img_name.split('_')[2]) in view_point_list:
                shutil.copyfile(os.path.join(sampler_path, img_name), os.path.join(output_path, img_name))
