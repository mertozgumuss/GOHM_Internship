import cv2
import numpy as np
import matplotlib.pyplot as plt
from util.image_utils import def_image
from util.image_utils import show_image
from util.image_utils import show_image_matrix
from util.image_utils import merge_images
from util.image_utils import image_filter_yellow
from util.image_utils import canny_edge_threshold
def main():
    peach = def_image("../data/peach.png")
    cherry = def_image("../data/cherry.jpg")
    apple = def_image("../data/red_apple.jpg")
    new_image = merge_images(cherry,apple)
    last_image = merge_images(new_image,peach)
    show_image(last_image,"Initial")
    change_HSV = cv2.cvtColor(last_image, cv2.COLOR_RGB2HSV)
    image_filter_yellow(change_HSV,last_image)
    yellow_part = last_image[0:850, 3000:4000]
    change_HSV = cv2.cvtColor(yellow_part,cv2.COLOR_RGB2HSV)
    image_filter_yellow(change_HSV,yellow_part)
    last_image_edges = canny_edge_threshold(last_image, 100, 50)
    apple_part = new_image[0:800, 1300:2200]
    cherry_part = new_image[0:800, 0:1300]
    apple_gray = cv2.cvtColor(apple_part, cv2.COLOR_BGR2GRAY)
    apple_edges = canny_edge_threshold(apple_gray, 100, 50)
    apple_edge_number = np.count_nonzero(apple_edges)
    print("Number of edges in apple = ", apple_edge_number)
    cherry_gray = cv2.cvtColor(cherry_part, cv2.COLOR_BGR2GRAY)
    cherry_edges = canny_edge_threshold(cherry_gray, 100, 50)
    cherry_edge_number = np.count_nonzero(cherry_edges)
    print("Number of edges in cherry = ", cherry_edge_number)
    images = (last_image, last_image_edges, apple_edges, cherry_edges)
    show_image_matrix(2,images)
    print("Upper threshold value = 100")
    print("Lower threshold value = 50")

    if cherry_edge_number > apple_edge_number:
        show_image(cherry_part, "The fruit is cherry")
        show_image(apple_part, "The fruit is apple")
        show_image(yellow_part,"The fruit is peach")



if __name__ == "__main__":
    main()
