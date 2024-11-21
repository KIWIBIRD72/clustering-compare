import numpy as np
from constants.general import ROOT_DIR
from clustering.supervised import SupervisedClassifier
from clustering.unsupervised import UnsupervisedClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from numpy.typing import NDArray
from scipy.stats import mode
from rich import inspect


def test_supervised():
    supervised_clustering = SupervisedClassifier()
    model, X_test, y_test, _ = supervised_clustering.cluster()

    y_pred = model.predict(X_test)
    inspect(f"Accuracy: {accuracy_score(y_test, y_pred)}",
            title='Test Supervised')
    print(classification_report(y_test, y_pred))


def test_unsupervised():
    unsupervised_clustering = UnsupervisedClassifier()
    model, X_test, y_test, y_train = unsupervised_clustering.cluster()

    clusters_to_labels = {}
    for cluster_id in range(unsupervised_clustering.CLUSTERS_AMOUNT):
        cluster_indices = np.where(model.labels_ == cluster_id)
        true_labels = y_train[cluster_indices]
        most_common_label = mode(
            true_labels, keepdims=True).mode[0]  # type: ignore
        clusters_to_labels[cluster_id] = most_common_label

    test_clusters: NDArray[np.float64] = model.predict(X_test)
    test_pred_labels = [clusters_to_labels[label]
                        for label in test_clusters]

    accuracy = accuracy_score(y_test, test_pred_labels)
    inspect(f"Accuracy: {accuracy}", title='Test Unsupervised', docs=False)
    print(classification_report(y_test, test_pred_labels, zero_division=1))


def main():
    test_supervised()
    test_unsupervised()


if __name__ == '__main__':
    main()
