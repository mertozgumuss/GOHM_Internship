import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def def_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def list_image_gray(image):
    for i in range(0, 2447):
        image[i] = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
        return image


def show_image(image, title="default"):
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(image)
    plt.show()


def show_image_matrix(size, image_list):
    for index, image in enumerate(image_list):
        plt.subplot(size, size, index + 1)
        plt.imshow(image)
    plt.show()


def image_features(image):
    dimensions = image.shape
    print('Image Dimension    : ', dimensions)
    image_red = image[:, :, 0]
    image_green = image[:, :, 1]
    image_blue = image[:, :, 2]
    print("Mean value of red", np.mean(image_red))
    print("Mean value of green", np.mean(image_green))
    print("Mean value of blue", np.mean(image_blue))
    return image_green, image_red


def print_image_color(apple, image_green, image_red):
    if np.mean(image_green) > np.mean(image_red):
        show_image(apple, title="Green")
    elif np.mean(image_green) < np.mean(image_red):
        show_image(apple, title="Red")


def image_filter_green(change_HSV, change_RGB):
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    mask_green = cv2.inRange(change_HSV, green_lower, green_upper)
    res = cv2.bitwise_and(change_RGB, change_RGB, mask=mask_green)
    show_image(res, "green")


def image_filter_red(change_HSV, change_RGB):
    red_lower = np.array([100, 100, 100], np.uint8)
    red_upper = np.array([179, 255, 255], np.uint8)
    mask_red = cv2.inRange(change_HSV, red_lower, red_upper)
    res = cv2.bitwise_and(change_RGB, change_RGB, mask=mask_red)
    show_image(res, "red")


def image_filter_yellow(change_HSV, change_RGB):
    yellow_lower = np.array([6, 100, 100])
    yellow_upper = np.array([28, 255, 255])
    mask_yellow = cv2.inRange(change_HSV, yellow_lower, yellow_upper)
    res = cv2.bitwise_and(change_RGB, change_RGB, mask=mask_yellow)
    show_image(res, "Filtering yellow")


def canny_edge_threshold(image, upper_threshold, lower_threshold):
    canny_image = cv2.Canny(image, threshold1=upper_threshold, threshold2=lower_threshold)
    return canny_image


def merge_images(image1, image2):
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    if height1 > height2:
        scale = (height1 / height2)
        height2 = height1
        width2 = int(width2 * scale)
        image2 = resize_image(image2, (width2, height2))
    else:
        scale = (height2 / height1)
        height1 = height2
        width1 = int(width1 * scale)
        image1 = resize_image(image1, (width1, height1))

    width_total = width1 + width2
    height_total = height1
    blank_image = np.zeros((height_total, width_total, 3), np.uint8)
    blank_image[0:height1, 0:width1, :] = image1
    blank_image[0:height2, width1:, :] = image2
    return blank_image


def resize_image(image, dim):
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
