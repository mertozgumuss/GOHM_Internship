
import cv2

from util.image_utils import show_image
from util.image_utils import image_filter_red
from util.image_utils import image_filter_green

def main():
    image = cv2.imread('../data/apples.jpg')
    change_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image(change_RGB, "initial")
    change_HSV = cv2.cvtColor(change_RGB, cv2.COLOR_BGR2HSV)
    image_filter_green(change_HSV, change_RGB)
    image_filter_red(change_HSV, change_RGB)

if __name__ == "__main__":
    main()

