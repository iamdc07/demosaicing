import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def load_image():
    # img = cv2.imread('image_set/oldwell_mosaic.bmp', 0)
    # coloured_img = cv2.imread('image_set/oldwell.jpg')
    # img = cv2.imread('image_set/crayons_mosaic.bmp', 0)
    # coloured_img = cv2.imread('image_set/crayons.jpg')
    img = cv2.imread('image_set/pencils_mosaic.bmp', 0)
    coloured_img = cv2.imread('image_set/pencils.jpg')
    b_channel, g_channel, r_channel = make_channels(img)
    generated_img_b, generated_img_g, generated_img_r = perform_conv(b_channel, g_channel, r_channel)

    generated_img = cv2.merge((generated_img_b, generated_img_g, generated_img_r))
    numpy_concat = np.concatenate((coloured_img, generated_img), axis=1)

    difference = squared_differences(coloured_img, generated_img)

    cv2.imshow('Result', numpy_concat)
    cv2.imshow("Difference", difference)
    cv2.waitKey(5000)

    roi = difference[150:200, 100:150]
    cv2.imshow("Close up patch of Part 1", roi)

    freeman_img = freeman(generated_img_b, generated_img_g, generated_img_r)
    numpy_concat = np.concatenate((freeman_img, generated_img), axis=1)
    cv2.imshow('Freeman Method', numpy_concat)

    difference = squared_differences(coloured_img, freeman_img)
    cv2.imshow("Difference new", difference)

    roi = difference[150:200, 100:150]
    cv2.imshow("Close up patch of Part 2", roi)
    cv2.waitKey()


def perform_conv(b_channel, g_channel, r_channel):
    img_shape = b_channel.shape
    generated_img_b = np.zeros((img_shape), np.uint8)
    generated_img_g = np.zeros((img_shape), np.uint8)
    generated_img_r = np.zeros((img_shape), np.uint8)

    cv2.filter2D(g_channel, dst=generated_img_g, ddepth=-1, kernel=fetch_kernel(1))
    cv2.filter2D(r_channel, dst=generated_img_r, ddepth=-1, kernel=fetch_kernel(2))
    cv2.filter2D(b_channel, dst=generated_img_b, ddepth=-1, kernel=fetch_kernel(0))

    return generated_img_b, generated_img_g, generated_img_r


def fetch_kernel(kernel_index):
    if kernel_index == 0:
        # b
        kernel = (np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], np.uint8))
        kernel = np.divide(kernel, 4)
    elif kernel_index == 1:
        # g
        kernel = (np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], np.uint8))
        kernel = np.divide(kernel, 4)
    elif kernel_index == 2:
        # r
        kernel = (np.array([[0, 1, 0],
                           [1, 4, 1],
                           [0, 1, 0]], np.uint8))
        kernel = np.divide(kernel, 4)

    return kernel


def squared_differences(coloured_img, generated_img):
    d_b, d_g, d_r = cv2.split(generated_img)
    b, g, r = cv2.split(coloured_img)

    difference1 = b - d_b
    difference2 = g - d_g
    difference3 = r - d_r

    difference1 = np.square(difference1)
    difference2 = np.square(difference2)
    difference3 = np.square(difference3)

    square_root1 = np.sqrt(difference1).astype(np.uint8)
    square_root2 = np.sqrt(difference2).astype(np.uint8)
    square_root3 = np.sqrt(difference3).astype(np.uint8)

    difference_img = cv2.merge((square_root1, square_root2, square_root3))\

    return difference_img


def make_channels(source_image):
    img_shape = source_image.shape

    b_channel = cv2.bitwise_and(source_image, source_image, mask=fetch_channel_mask(0, img_shape))
    g_channel = cv2.bitwise_and(source_image, source_image, mask=fetch_channel_mask(1, img_shape))
    r_channel = cv2.bitwise_and(source_image, source_image, mask=fetch_channel_mask(2, img_shape))

    return b_channel, g_channel, r_channel


def freeman(b_channel, g_channel, r_channel):
    g_r = np.subtract(g_channel, r_channel)
    b_r = np.subtract(b_channel, r_channel)

    g_r = cv2.medianBlur(g_r, 1)
    b_r = cv2.medianBlur(b_r, 1)

    g_r = np.add(g_r, r_channel)
    b_r = np.add(b_r, r_channel)

    freeman_img = cv2.merge((b_r, g_r, r_channel))
    return freeman_img


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
