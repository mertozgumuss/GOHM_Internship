import os
import cv2
from contextlib import contextmanager

def listdir_nohidden(path):
    return sorted([f for f in os.listdir(path) if not f.startswith('.')], key=str.lower)


@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)

class Dataloader:
    def __init__(self,data_dir_path):
        self.dir = data_dir_path
        self.image_path_list= []
        self.image_list=[]

    def get_image_paths(self):
        with working_directory(self.dir):
            self.image_path_list = os.listdir()

    def load_images(self):
        for image in self.image_path_list:
            image_path_actual = self.dir + image
            self.image_list.append(cv2.imread(image_path_actual))

    def __call__(self):
        self.get_image_paths()

        self.load_images()


def assign_label(img, apple_type):
    return apple_type


def make_train_data(apple_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, apple_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))