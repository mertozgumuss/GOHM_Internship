from sklearn.datasets import load_files
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics



train_dir = '../fruits/Training'
test_dir = '../fruits/Test'


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files, targets, target_labels
x_train, y_train,target_labels = load_dataset(train_dir)
x_test, y_test,_ = load_dataset(test_dir)

print('Training set size : ' , x_train.shape[0])
print('Testing set size : ', x_test.shape[0])

no_of_classes = len(np.unique(y_train))
y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
x_test,x_valid = x_test[7000:],x_test[:7000]
y_test,y_vaild = y_test[7000:],y_test[:7000]
print('Vaildation X : ',x_valid.shape)
print('Vaildation y :',y_vaild.shape)
print('Test X : ',x_test.shape)
print('Test y : ',y_test.shape)

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_train = np.array(convert_image_to_array(x_train))
print('Training set shape : ',x_train.shape)
x_valid = np.array(convert_image_to_array(x_valid))
print('Validation set shape : ',x_valid.shape)
x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)
print('1st training image shape ',x_train[0].shape)
x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255
X_train, X_test, y_train, y_test = train_test_split(x_train,y_train, test_size=0.3,random_state=10)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))