# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 8:36
# @Author  : yblir
# @File    : glip_predict.py
# explain  : 
# =======================================================
import warnings

warnings.filterwarnings("ignore")
from transformers import logging

logging.set_verbosity_error()
# pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.

        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def draw_images(image, boxes, classes, scores, colors, xyxy=True):
    """
    对单张图片进行多个框的绘制
    Args:
        image: pillow与numpy格式都行, 反正都会转成pillow格式,h,w,c
        boxes: tensor与numpy格式都行, 最后都会转成numpy格式
        xyxy: 默认是xyxy格式, 如果为False,就是xywh格式,需要进行一次格式转换
        classes: 每个框的类别,list, 与boxes框对应
        scores: 每个预测类别得分,list, 与boxes框对应
        colors: 每个类别框颜色
    Returns:
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    # 设置字体,pillow 绘图环节
    font = ImageFont.truetype(font='configs/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    # 多次画框的次数,根据图片尺寸不同,把框画粗
    thickness = max((image.size[0] + image.size[1]) // 300, 1)
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i]

        label = '{}:{:.2f}'.format(classes[i], scores[i])
        tx1, ty1, tx2, ty2 = font.getbbox(label)
        tw, th = tx2 - tx1, ty2 - tx1

        text_origin = np.array([x1, y1 - th]) if y1 - th >= 0 else np.array([x1, y1 + 1])

        # 在目标框周围偏移几个像素多画几次, 让边框变粗
        for j in range(thickness):
            draw.rectangle((x1 + j, y1 + j, x2 - j, y2 - j), outline=color)

        # 画标签
        draw.rectangle((text_origin[0], text_origin[1], text_origin[0] + tw, text_origin[1] + th), fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    return image


config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = r'E:\PyCharm\PreTrainModel\glip_tiny_model_o365_goldg_cc_sbu.pth'

# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)


def glip_inference(image_, caption_):
    # 为不同类别设置颜色, 从caption提取的类别不同
    colors_ = Colors()

    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.5)

    # 从预测结果中提取预测类别,得分和检测框
    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()

    # 为每个预测类别设置框颜色
    colors = [colors_(idx) for idx in labels]
    # 获得标签数字对应的类别名
    labels_names = glip_demo.get_label_names(labels)

    return boxes, scores, labels_names, colors


if __name__ == '__main__':
    # caption = 'bobble heads on top of the shelf'
    # caption = "Striped bed, white sofa, TV, carpet, person"
    # caption = "table on carpet"
    # caption = "Table, TV"
    caption = 'person'
    image = cv2.imread('docs/bus.jpg')

    boxes, scores, labels_names, colors = glip_inference(image, caption)

    print(labels_names, scores)
    print(boxes)

    image = draw_images(image=image, boxes=boxes, classes=labels_names, scores=scores, colors=colors)

    image.show()
    # image.save('bb.jpg')
