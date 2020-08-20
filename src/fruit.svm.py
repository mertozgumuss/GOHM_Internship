# import pandas as pd
import numpy as np
import cv2
import os


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from util.dir_utils import listdir_nohidden

IMAGE_SIZE = 100
CHANNEL_SIZE = 3
SEED = 12345

def shuffle_dataset(dataset):
    np.random.seed(SEED)
    np.random.shuffle(dataset)

def calculate_dataset_size(data_dir):
    labels = listdir_nohidden(data_dir)
    size_list = [len(listdir_nohidden(os.path.join(data_dir,label))) for label in labels]
    return sum(size_list)

def load_features_labels(data_list, label_list, labels, train_dir):
    total_index = 0
    for i in labels:
        path = os.path.join(train_dir, i)
        class_num = labels.index(i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            data_list[total_index, :, :, :] = img_array
            label_list[total_index] = class_num
            total_index += 1

def load_features_labels_mark2(train_data, label_list, labels, train_dir, test_dir):
    total_index = 0
    for i in labels:
        path = os.path.join(train_dir, i)
        class_num = labels.index(i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            train_data[total_index, :, :, :] = img_array
            label_list[total_index] = class_num
            total_index += 1

    for i in labels:
        path = os.path.join(test_dir, i)
        class_num = labels.index(i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            train_data[total_index, :, :, :] = img_array
            label_list[total_index] = class_num
            total_index += 1


def main():
    train_dir = "../fruits/Training1"
    test_dir = "../fruits/Test1"

    train_size = calculate_dataset_size(train_dir)
    test_size = calculate_dataset_size(test_dir)

    labels = listdir_nohidden(train_dir)

    #first method
    # X_train = np.ndarray((train_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE ))
    # Y_train = np.zeros((train_size,1))
    #
    # X_test = np.ndarray((test_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE))
    # Y_test = np.zeros((test_size,1))

    #second method
    dataset = np.ndarray((train_size + test_size, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE))
    labelset = np.ndarray((train_size + test_size,1))

   # load_features_labels(X_train, Y_train, labels, train_dir)
   # load_features_labels(X_test, Y_test, labels, test_dir)

    load_features_labels_mark2(dataset, labelset, labels, train_dir, test_dir)

    # shuffle_dataset(X_train)
    # shuffle_dataset(Y_train)
    #
    # shuffle_dataset(X_test)
    # shuffle_dataset(Y_test)

    shuffle_dataset(dataset)
    shuffle_dataset(labelset)

    # print(Y_test[500:600])
    # print("shape of X_train= {}".format(X_train.shape))
    # print("shape of Y_train= {}".format(Y_train.shape))

    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labelset, test_size=0.25, random_state=40)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

   # d2_train_dataset = X_train.reshape((train_size , IMAGE_SIZE * IMAGE_SIZE * CHANNEL_SIZE))
    d2_train_dataset = X_train.reshape((X_train.shape[0] , IMAGE_SIZE * IMAGE_SIZE * CHANNEL_SIZE))
#    d2_test_dataset = X_test.reshape(test_size, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_SIZE)
    d2_test_dataset = X_test.reshape((X_test.shape[0], IMAGE_SIZE * IMAGE_SIZE * CHANNEL_SIZE))


    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", verbose=True, gamma=0.001))
    clf.fit(d2_train_dataset, Y_train.ravel())
    clf.score(d2_test_dataset, Y_test)
    scores = cross_val_score(clf, d2_train_dataset, Y_test, cv=5)
    print(scores)
    # y_predict = clf.predict(d2_test_dataset)
    # print("Accuracy:", metrics.accuracy_score(Y_test, y_predict))
    # print(metrics.confusion_matrix(y_true=Y_test, y_pred=y_predict))
    # print("Predicted values", y_predict[1100:1110])
    # print("Test values\n", Y_test[1100:1110])


if __name__ == "__main__":
    main()




#

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
