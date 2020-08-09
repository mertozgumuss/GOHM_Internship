
from util.image_utils import image_features
from util.image_utils import def_image
from util.image_utils import print_image_color

def main():

    green_apple = def_image("../data/apple_green.jpg")
    image_green, image_red = image_features(green_apple)
    print_image_color(green_apple, image_green, image_red)

    red_apple = def_image("../data/red_apple.jpg")
    image_green, image_red = image_features(red_apple)
    print_image_color(red_apple, image_green, image_red)


if __name__ == "__main__":
    main()