import cv2
import matplotlib.pyplot as plt
import numpy as np

def def_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(image, title ="default"):
    plt.imshow(image)
    plt.title(title)
    plt.show()

def image_features(image):
    dimensions = image.shape
    print('Image Dimension    : ', dimensions)
    image_red = image[:, :, 0]
    image_green = image[:, :, 1]
    image_blue = image[:, :, 2]
    print(np.mean(image_red))
    print(np.mean(image_green))
    print(np.mean(image_blue))
    return image_green, image_red

def print_image_color(apple, image_green, image_red):
    if np.mean(image_green) > np.mean(image_red):
        show_image(apple, title="green")
    elif np.mean(image_green) < np.mean(image_red):
        show_image(apple, title="red")


def image_filter_green(change_HSV, change_RGB):
    green_lower = np.array([25, 52, 72],np.uint8)
    green_upper = np.array([102, 255, 255],np.uint8)
    mask_green = cv2.inRange(change_HSV, green_lower, green_upper)
    res = cv2.bitwise_and(change_RGB, change_RGB, mask=mask_green)
    show_image(res, "green")


def image_filter_red(change_HSV, change_RGB):
    red_lower = np.array([100, 100, 100], np.uint8)
    red_upper = np.array([179, 255, 255], np.uint8)
    mask_red = cv2.inRange(change_HSV, red_lower, red_upper)
    res = cv2.bitwise_and(change_RGB, change_RGB, mask=mask_red)
    show_image(res, "red")
