import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def main():

    print(os.listdir("../digit_recognizer"))
    train_data = pd.read_csv("../digit_recognizer/train.csv")
    test_data = pd.read_csv("../digit_recognizer/test.csv")
    round(train_data.drop('label', axis=1).mean(), 2)
    y = train_data['label']
    X = train_data.drop(columns = 'label')
    X = X/255.0
    test_data = test_data/255.0
    print("X:", X.shape)
    print("test_data:", test_data.shape)
    print("train_data", train_data.shape)
    print(y.shape)
    X_scaled = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print("Predicted values",y_pred[100:105])
    print("Test values\n",y_test[100:105])
    for i in (np.random.randint(0,270,6)):
     two_d = (np.reshape(X_test[i], (28, 28)) * 255)
     plt.title('predicted label: {0}'.format(y_pred[i]))
     plt.imshow(two_d, interpolation='nearest', cmap='gray')
     plt.show()
if __name__ == "__main__":
    main()