import cv2
import numpy as np


def AddBlankRegion(image, min_length, max_length):
    blank_length = np.random.randint(min_length, max_length + 1)
    h, w, _ = image.shape
    # 矩形の始点を決める
    sx = np.random.randint(0, w - blank_length)
    sy = np.random.randint(0, h - blank_length)

    cv2.rectangle(
        image, (sx, sy), (sx + blank_length, sy + blank_length), (0, 0, 0), thickness=-1
    )

    return image
