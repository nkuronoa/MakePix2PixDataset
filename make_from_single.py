import argparse
import os
import shutil
import time

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import ImageProcessing as ip


def make_merge_image(imageA_path, imageB_path, resize_flag=False, blank_flag=False):
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)

    if resize_flag:
        imageA = cv2.resize(imageA, (256, 256))
        imageB = cv2.resize(imageB, (256, 256))

    if blank_flag:
        # make blank image for only input image
        imageA = ip.AddBlankRegion(imageA, 16, 32)

    merge_image = np.concatenate([imageA, imageB], axis=1)

    return merge_image


def main():
    parser = argparse.ArgumentParser("Make Dataset for Pix2Pix from single image")
    parser.add_argument("-input", default="image", type=str)
    parser.add_argument("-output", default="single_merge", type=str)
    parser.add_argument("-test_ratio", default=0.0, type=float)
    parser.add_argument("--blank", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    image_list = os.listdir(args.input)

    start = time.time()
    count = 1
    for i in range(len(image_list)):
        merge_image = make_merge_image(
            os.path.join(args.input, image_list[i]),
            os.path.join(args.input, image_list[i]),
            blank_flag=args.blank,
        )
        name, ext = os.path.splitext(image_list[i])
        cv2.imwrite(
            os.path.join(args.output, name + ".jpg"), merge_image
        )
        count += 1

    end = time.time()
    print("elapsed time:{}".format(end - start))
    # test_ratioが0より高い場合はtrainとtestに振り分ける
    if (args.test_ratio > 0) and (args.test_ratio < 1):
        train_list, test_list = train_test_split(image_list, test_size=args.test_ratio)
        os.makedirs(os.path.join(args.output, "train"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "test"), exist_ok=True)
        for i in range(len(train_list)):
            name, ext = os.path.splitext(train_list[i])
            shutil.move(
                os.path.join(args.output, name + ".jpg"),
                os.path.join(args.output, "train"),
            )

        for i in range(len(test_list)):
            name, ext = os.path.splitext(test_list[i])
            shutil.move(
                os.path.join(args.output, name + ".jpg"),
                os.path.join(args.output, "test"),
            )


if __name__ == "__main__":
    main()
