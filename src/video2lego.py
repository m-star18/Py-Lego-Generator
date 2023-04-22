import argparse
import cv2
import numpy as np
from itertools import product
from math import ceil


def get_args():
    parser = argparse.ArgumentParser("Lego-generator")
    parser.add_argument("--input", type=str, default="data/input.mp4", help="Path to input image")
    parser.add_argument("--output", type=str, default="data/output.mp4", help="Path to output image")
    parser.add_argument("--stride", type=int, default=10, help="Size of each lego brick")
    parser.add_argument("--fps", type=int, default=0, help="Frames per second")
    parser.add_argument("--overlay_ratio", type=float, default=0.2, help="Overlay width ratio")
    args = parser.parse_args()
    return args


def preprocess_lego_brick(opt):
    lego_brick = cv2.imread("data/1x1.png", cv2.IMREAD_COLOR)
    lego_brick = cv2.resize(lego_brick, (opt.stride, opt.stride)).astype(np.int64)
    lego_brick[lego_brick < 33] = -100
    lego_brick[(33 <= lego_brick) & (lego_brick <= 233)] -= 133
    lego_brick[lego_brick > 233] = 100
    return lego_brick


def create_video_writer(cap, opt):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = opt.fps if opt.fps != 0 else int(cap.get(cv2.CAP_PROP_FPS))
    size = (ceil(width / opt.stride) * opt.stride, ceil(height / opt.stride) * opt.stride)
    return cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"XVID"), fps, size)


def process_frame(frame, lego_brick, opt):
    frame = cv2.resize(frame, (
        ceil(frame.shape[1] / opt.stride) * opt.stride, ceil(frame.shape[0] / opt.stride) * opt.stride))
    height, width, _ = frame.shape
    blank_image = np.zeros((height, width, 3), np.uint8)

    for i, j in product(range(int(width / opt.stride)), range(int(height / opt.stride))):
        partial_frame = frame[j * opt.stride: (j + 1) * opt.stride,
                              i * opt.stride: (i + 1) * opt.stride, :]
        avg_color = np.mean(np.mean(partial_frame, axis=0), axis=0)
        blank_image[j * opt.stride: (j + 1) * opt.stride, i * opt.stride: (i + 1) * opt.stride, :] = np.clip(avg_color + lego_brick, 0, 255)

    if opt.overlay_ratio:
        overlay = cv2.resize(frame, (int(width * opt.overlay_ratio), int(height * opt.overlay_ratio)))
        blank_image[height - int(height * opt.overlay_ratio):, width - int(width * opt.overlay_ratio):, :] = overlay

    return blank_image


def main(opt):
    lego_brick = preprocess_lego_brick(opt)

    cap = cv2.VideoCapture(opt.input)
    out = create_video_writer(cap, opt)

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        processed_frame = process_frame(frame, lego_brick, opt)
        out.write(processed_frame)

    cap.release()
    out.release()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
