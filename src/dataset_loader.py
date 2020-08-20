import cv2
import matplotlib.pyplot as plt
from util.dir_utils import Dataloader

data_dir_test = "../digit_recognizer/sample_submission"
loader = Dataloader(data_dir_test)
loader()
print(loader.image_list)
plt.imshow(loader.image_list[0])
plt.show()
