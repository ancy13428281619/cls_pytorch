"""本脚本支持torchvision里所有分类模型的特征可视化"""
import cv2
import torch
import argparse
import numpy as np
from torchvision.models import *
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

try:
    from captum.attr import GradientShap, visualization as viz
except ImportError:
    import os

    os.system('pip install captum')
    from captum.attr import GradientShap, visualization as viz


class VisModel:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(0)
        np.random.seed(0)

    def load_model(self):
        model = eval(self.args.model_name)(num_classes=self.args.num_classes, pretrained=False)
        checkpoint = torch.load(self.args.weight_path, map_location='cpu')
        # 万能框架权重转torchvision权重
        if self.args.is_universal_framework:
            checkpoint = {key[21:]: value for key, value in checkpoint['model'].items()}
        model.load_state_dict(checkpoint)
        model = model.eval()
        return model.cuda()

    def load_image(self):
        image = cv2.imread(self.args.image_path)
        return image

    def preprocess_image(self, image):
        img_float = np.float32(image) / 255.
        img_tensor = torch.from_numpy(img_float)
        img_batch = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        img_batch = img_batch
        return img_batch.cuda()

    def postprocess_image(self, output_tensor):
        output = F.softmax(output_tensor, dim=1)
        print(output)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        # print('Predicted label:', pred_label_idx.numpy(), 'score: ', '(', prediction_score.squeeze().item(), ')')
        return pred_label_idx

    def visualizes_attributions(self, input_image, pred_label_idx, model):
        colors = [(1, 1, 1), (0, 0, 1), (0, 0.8, 1), (1, 1, 0.2), (1, 0, 0)]
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         colors, N=256)
        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([input_image * 0, input_image * 1])

        attributions_gs = gradient_shap.attribute(input_image,
                                                  n_samples=1,
                                                  stdevs=0.01,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(input_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=default_cmap,
                                              show_colorbar=True)

    def run(self):
        image = self.load_image()
        model = self.load_model()
        input_image = self.preprocess_image(image)
        output_tensor = model(input_image)
        pred_label_idx = self.postprocess_image(output_tensor)
        self.visualizes_attributions(input_image, pred_label_idx, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizes model attributions')
    parser.add_argument(
        "--image_path",
        default='/home/pi/Desktop/test_vision/EU_6006_4_0_28_16_3_64091.png',
        help="path to config file",
        type=str,
    )
    parser.add_argument("--weight_path", type=str,
                        default='/home/pi/Desktop/company_project/PIE电源盒子缺陷检测/AUA01R003_V2/src/modules/ai_deployer/DF30/inputs/configs/weights/model_cls_PIE_DF30_1226.pth')
    parser.add_argument("--model_name", type=str, default='resnet18')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--is_universal_framework", type=bool, default=True, help='是否为公司的万能框架的权重')
    vm = VisModel(args=parser.parse_args())
    vm.run()
