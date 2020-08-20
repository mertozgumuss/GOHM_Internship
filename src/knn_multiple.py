from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



def main():
    wine = datasets.load_wine()
    print("Wine features= ",wine.feature_names)
    print("Wine types= ",wine.target_names)
    print(wine.target)
    print(wine.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
                                                        test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    knn2 = KNeighborsClassifier(n_neighbors=7)
    knn2.fit(X_train,y_train)
    y_predict2 = knn2.predict(X_test)
    print(y_predict)
    print("Accuracy(n_neighbour = 5):", metrics.accuracy_score(y_test, y_predict))
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_predict))
    print(y_predict2)
    print("Accuracy(n_neighbour = 7)", metrics.accuracy_score(y_test, y_predict2))
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_predict2))







if __name__ == "__main__":
    main()