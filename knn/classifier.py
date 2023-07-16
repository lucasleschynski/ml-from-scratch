import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from utils.utils import euclidean_distance
from utils.pca import PCA

class KNNClassifier():
    def __init__(self) -> None:
        self.data: np.matrix = None
        self.labels: np.ndarray = None

    def load_data(self, data: np.ndarray, labels: np.ndarray):
        if np.shape(data)[0] != np.size(labels):
            raise ValueError("Number of samples not equal to number of labels")
        
        self.data = data
        self.labels = labels

    def classify(self, point: np.ndarray, k):
        if np.size(point) != np.shape(self.data)[1]:
            raise ValueError("Provided point has invalid dimension")
        
        distances = np.array([euclidean_distance(point, p) for p in self.data])
        closest_k = np.argsort(distances)[:k]
        closest_k_classes = [self.labels[i] for i in closest_k]
        class_counts = np.bincount(closest_k_classes)
        estimate = np.argmax(class_counts)
        return estimate
    
    def classify_test_samples(self, test_samples: np.ndarray, test_labels: np.ndarray, k: int):
        predictions = []
        for test_point, truth in list(zip(test_samples, test_labels)):
            # transformed = pca.transform(test_point)
            guess = self.classify(test_point, k)
            predictions.append(1) if guess == truth else predictions.append(0)

        accuracy = sum(predictions)/len(predictions)
        print(f"Test accuracy: {accuracy}")
        return accuracy


if __name__=="__main__":
    clf = KNNClassifier()
    pca = PCA(n_components=2)

    iris = load_iris()
    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2)

    # Obtaining the optimal transform for the training data, then loading it into the classifier
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    clf.load_data(X_train, y_train)

    # Using the training transform on the test data and performing classification 
    X_test = pca.transform(X_test)
    clf.classify_test_samples(X_test, y_test, k=5)
    pca.plot_pca_transform(X_train, y_train)