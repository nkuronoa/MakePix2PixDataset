import argparse
import os
import shutil
import time

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def get_kp_list(dirpath):
    akaze = cv2.AKAZE_create()

    image_list = os.listdir(dirpath)
    kp_list = []
    des_list = []
    for filename in image_list:
        image = cv2.imread(os.path.join(dirpath, filename))
        keypoints, descriptors = akaze.detectAndCompute(image, None)
        kp_list.append(keypoints)
        des_list.append(descriptors)

    return image_list, kp_list, des_list


def make_merge_image(imageA_path, imageB_path):
    imageA = cv2.imread(imageA_path)
    imageB = cv2.imread(imageB_path)

    resize_imageA = cv2.resize(imageA, (256, 256))
    resize_imageB = cv2.resize(imageB, (256, 256))

    merge_image = np.concatenate([resize_imageA, resize_imageB], axis=1)

    return merge_image


def main():
    parser = argparse.ArgumentParser("Make Dataset for Pix2Pix")
    parser.add_argument("-inputA", default="imageA", type=str)
    parser.add_argument("-inputB", default="imageB", type=str)
    parser.add_argument("-output", default="merge", type=str)
    parser.add_argument("-num", default=50, type=int)
    parser.add_argument("-ratio", default=0.6, type=float)
    parser.add_argument("-test_ratio", default=0.0, type=float)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 特徴点準備
    imageA_list, kpA_list, desA_list = get_kp_list(args.inputA)
    imageB_list, kpB_list, desB_list = get_kp_list(args.inputB)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    image_list = []
    # AをベースにBの中から最も対応点数が多い画像とペアを作る
    start = time.time()
    count = 1
    for i in range(len(imageA_list)):
        max_good_num = args.num
        current_imageB = -1
        for j in range(len(imageB_list)):
            matches = matcher.knnMatch(desA_list[i], desB_list[j], k=2)
            # Apply ratio test
            good_pairs = []
            for m, n in matches:
                if m.distance < args.ratio * n.distance:
                    good_pairs.append(m)

            if len(good_pairs) > max_good_num:
                current_imageB = j

        if current_imageB >= 0:
            merge_image = make_merge_image(
                os.path.join(args.inputA, imageA_list[i]),
                os.path.join(args.inputB, imageB_list[current_imageB]),
            )
            cv2.imwrite(
                os.path.join(args.output, "merge" + str(count) + ".jpg"), merge_image
            )
            image_list.append("merge" + str(count) + ".jpg")
            count += 1

    end = time.time()
    print("elapsed time:{}".format(end - start))

    # test_ratioが0より高い場合はtrainとtestに振り分ける
    if (args.test_ratio > 0) and (args.test_ratio < 1):
        train_list, test_list = train_test_split(image_list, test_size=args.test_ratio)
        os.makedirs(os.path.join(args.output, "train"), exist_ok=True)
        os.makedirs(os.path.join(args.output, "test"), exist_ok=True)
        for i in range(len(train_list)):
            shutil.move(
                os.path.join(args.output, train_list[i]),
                os.path.join(args.output, "train"),
            )

        for i in range(len(test_list)):
            shutil.move(
                os.path.join(args.output, test_list[i]),
                os.path.join(args.output, "test"),
            )


if __name__ == "__main__":
    main()
