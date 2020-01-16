import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def load_image():
    img = cv2.imread('image_set/oldwell_mosaic.bmp', 0)
    coloured_img = cv2.imread('image_set/oldwell.jpg')
    print(img.dtype)
    cv2.imshow("Crayons", img)
    b_channel, g_channel, r_channel = make_channels(img)
    generated_img_b, generated_img_g, generated_img_r = perform_conv(b_channel, g_channel, r_channel)

    generated_img = cv2.merge((generated_img_b, generated_img_g, generated_img_r))
    numpy_concat = np.concatenate((coloured_img, generated_img), axis=1)
    cv2.imshow('Result', numpy_concat)
    cv2.waitKey(10000)

    squared_differences()

    # get_bayer_array(img)
    # cv2.waitKey(150000)
    # b_pixel, g_pixel, r_pixel = img_gray[200, 50]
    # print('B={}, G={}, R={}'.format(b_pixel, g_pixel, r_pixel))
    # b_channel, g_channel, r_channel  = split_channels(img)
    # kernel = fetch_kernel()
    # generated_image = perform_conv(b_channel, g_channel, r_channel, kernel)


def perform_conv(b_channel, g_channel, r_channel):
    generated_img_b = cv2.filter2D(b_channel, -1, kernel=fetch_kernel(0))
    generated_img_g = cv2.filter2D(g_channel, -1, kernel=fetch_kernel(1))
    generated_img_r = cv2.filter2D(r_channel, -1, kernel=fetch_kernel(2))

    return generated_img_b, generated_img_g, generated_img_r


def fetch_kernel(kernel_index):
    if kernel_index == 0:
        kernel = (np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]], np.uint8))/2
    elif kernel_index == 1:
        kernel = (np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]], np.uint8))/2
    elif kernel_index == 2:
        kernel = (np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]], np.uint8))/4

    return kernel


def squared_differences(coloured_img, generated_img):
    squared_differences = np.square(coloured_img) - np.square(generated_img)
    square_root = np.sqrt(squared_differences).astype(np.uint8)

    return square_root


def make_channels(source_image):
    img_shape = source_image.shape

    b_channel = cv2.bitwise_and(source_image, source_image, mask=fetch_channel_mask(0, img_shape))
    g_channel = cv2.bitwise_and(source_image, source_image, mask=fetch_channel_mask(1, img_shape))
    r_channel = cv2.bitwise_and(source_image, source_image, mask=fetch_channel_mask(2, img_shape))

    return b_channel, g_channel, r_channel


def fetch_channel_mask(color_index, img_shape):
    channel_mask = np.zeros(img_shape, dtype=np.uint8)

    # 0 Blue
    # 1 Green
    # 2 Red
    if color_index == 0:
        channel_mask[::2, ::2] = 1
    elif color_index == 1:
        channel_mask[1::2, 1::2] = 1
    elif color_index == 2:
        channel_mask[::2, 1::2] = 1
        channel_mask[1::2, ::2] = 1

    return channel_mask


if __name__ == "__main__":
    load_image();
