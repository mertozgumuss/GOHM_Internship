import cv2
import matplotlib.pyplot as plt
import numpy as np
from util.image_utils import show_image, merge_images
from util.image_utils import def_image
from util.image_utils import canny_edge_threshold
from util.image_utils import show_image_matrix


def main():

    cherry = def_image('../data/cherry.jpg')
    apple = def_image('../data/red_apple.jpg')
    image_gray_cherry = cv2.cvtColor(cherry, cv2.COLOR_BGR2GRAY)
    single_cherry_edges = canny_edge_threshold(image_gray_cherry, 100, 30)
    image_gray_apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
    single_apple_edges = canny_edge_threshold(image_gray_apple, 100, 50)
    images = [cherry, single_cherry_edges, apple, single_apple_edges]
    show_image_matrix(2, images)
    new_image = merge_images(cherry, apple)
    hole_image_edges = canny_edge_threshold(new_image, 100, 50)
    apple_part = new_image[0:800, 1300:2200]
    cherry_part = new_image[0:800, 0:1050]
    apple_gray = cv2.cvtColor(apple_part, cv2.COLOR_BGR2GRAY)
    apple_edges = canny_edge_threshold(apple_gray, 100, 50)
    apple_edge_number = np.count_nonzero(apple_edges)
    print("Number of edges in apple = ", apple_edge_number)
    cherry_gray = cv2.cvtColor(cherry_part, cv2.COLOR_BGR2GRAY)
    cherry_edges = canny_edge_threshold(cherry_gray, 100, 50)
    cherry_edge_number = np.count_nonzero(cherry_edges)
    print("Number of edges in cherry = ", cherry_edge_number)
    images = (new_image, hole_image_edges, apple_edges, cherry_edges)
    show_image_matrix(2,images)
    print("Upper threshold value = 100")
    print("Lower threshold value = 50")

    if cherry_edge_number > apple_edge_number:
         show_image(cherry_part, "The fruit is cherry")
         show_image(apple_part, "The fruit is apple")

if __name__ == "__main__":
    main()
