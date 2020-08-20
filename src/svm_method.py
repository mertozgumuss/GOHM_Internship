from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    cancer = datasets.load_breast_cancer()
    print("Features: ", cancer.feature_names)
    print("Labels: ", cancer.target_names)
    print(cancer.target.shape)
    print(cancer.data.shape)
    cancer.data.shape
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_predict))

if __name__ == "__main__":
    main()