# import pandas as pd
# import numpy as np
import cv2
import os
# from sklearn.preprocessing import scale
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# import matplotlib.pyplot as plt
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from keras.utils import to_categorical
from util.dir_utils import listdir_nohidden

IMAGE_SIZE = 100


def main():
    data_path = "../fruits/Training1"
    labels = listdir_nohidden(data_path)
    labelled_train_list = []

    for i in labels:
        path = os.path.join(data_path, i)
        class_num = labels.index(i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            labelled_train_list.append([img_array, class_num])



if __name__ == "__main__":
    main()

#
# directory_test = "../fruits/Test1/"
# label_test = ["Apple_Golden", "Avocado_", "Banana_", "Cherry_", "Cocos_", "Kiwi_",
#             "Lemon_", "Mango_", "Orange_", "Apple_Red_Delicious"]
#
# labelled_test_list = []
# img_size = 100
# for i in label_test:
#     path = os.path.join(directory_test, i)
#     class_num2 = label_test[i]
#     for img in os.listdir(path):
#         img_array2 = cv2.imread(os.path.join(path, img))
#         labelled_test_list.append([img_array2, class_num2])
#
# fruits_array_train = []
# for features, label in labelled_train_list:
#     fruits_array_train.append(features)
#
# X_train = []
# Y_train = []
# for features, label in labelled_train_list:
#     X_train.append(features)
#     Y_train.append(label)
# X_train = np.array(X_train)
#
# X_test = []
# Y_test = []
# for features, label in labelled_test_list:
#     X_test.append(features)
#     Y_test.append(label)
# X_test = np.array(X_test)
#
# print("shape of X_train= ", X_train.shape)
# print("shape of X_test=  ", X_test.shape)
# nsamples, nx, ny, nz = X_train.shape
# d2_train_dataset = X_train.reshape((nsamples, nx * ny * nz))
# print("shape of X_train= ", d2_train_dataset.shape)
# x_train, x_test, y_train, y_test = train_test_split(d2_train_dataset, Y_train, test_size=0.3, random_state=40)
# clf = svm.SVC(kernel='linear')
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_test)
# print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
# print(metrics.confusion_matrix(y_true=y_test, y_pred=y_predict))
# print("Predicted values", y_predict[1400:1406])
# print("Test values\n", y_test[1400:1406])
# for i in label_train:
#     plt.imshow(y_predict[i])
#     plt.show()
