import numpy as np
import csv
import sys
from sklearn.model_selection import train_test_split
import calendar


def main():

    # Check command-line arguments
    #if len(sys.argv) != 2:
        #sys.exit("Usage: python shopping.py data")

    # Load data from .csv and split into train and test sets
    evidence, labels = load_data(r"C:\Users\Sumana\Downloads\shopping.csv")
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.25)

    # Train model using Euclidean distance
    # For k=1 
    model_euc = train_nearest_neighbors_classifier(X_train, y_train, k=1, distance_metric='euclidean')
    predictions = model_euc.predict(X_test)
    sensitivity_euc, specificity_euc = evaluate(y_test, predictions)
    f1_euc = 2 * (sensitivity_euc * specificity_euc) / (sensitivity_euc + specificity_euc) if (sensitivity_euc + specificity_euc) > 0 else 0

    # print results for model trained using Euclidean distance
    print("Results for Nearest Neighbor classifier using Euclidean distance:")
    print(f"Number of Correct Predictions: {(np.sum(np.array(y_test) == np.array(predictions))).sum()}")
    print(f"Number of Incorrect Predictions: {(np.sum(np.array(y_test) != np.array(predictions))).sum()}")
    print(f"True Positive Rate: {100 * sensitivity_euc:.2f}%")
    print(f"True Negative Rate: {100 * specificity_euc:.2f}%")
    print(f"F1 score:{100 * f1_euc:.2f}%")

    # Train model using Manhattan distance
    # For k=1
    model_man = train_nearest_neighbors_classifier(X_train, y_train, k=1, distance_metric='manhattan')
    predictions = model_man.predict(X_test)
    sensitivity_man, specificity_man = evaluate(y_test, predictions)
    f1_man = 2 * (sensitivity_man * specificity_man) / (sensitivity_man + specificity_man) if (sensitivity_man + specificity_man) > 0 else 0

    #print results for model trained using manhattan distance
    print("Results for Nearest Neighbor classifier using Manhattan distance:")
    print(f"Number of Correct Predictions: {(np.sum(np.array(y_test) == np.array(predictions))).sum()}")
    print(f"Number of Incorrect Predictions: {(np.sum(np.array(y_test) != np.array(predictions))).sum()}")
    print(f"True Positive Rate: {100 * sensitivity_man:.2f}%")
    print(f"True Negative Rate: {100 * specificity_man:.2f}%")
    print(f"F1 score:{100 * f1_man:.2f}%")

# To load data from a .csv file
def load_data(filename):
  
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)

# To implement the K-nearest neighbor classifier
# For k=1, we get the nearest neighbor classifier
class NearestNeighborsClassifier:
    def __init__(self,k, distance_metric='euclidean_distance'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            # Calculate distances between each test point and all training points
            distances = [self.calculate_distance(x_test, x_train) for x_train in self.X_train]

            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]

            # Get the labels of the k nearest neighbors
            nearest_labels = [self.y_train[i] for i in nearest_indices]

            # Predict the class label as the mode of the k nearest neighbors
            prediction = np.argmax(np.bincount(nearest_labels))
            predictions.append(prediction)
        return predictions

    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(np.array(x1) - np.array(x2)))
        else:
            raise ValueError("Unsupported distance metric")


def train_nearest_neighbors_classifier(X_train, y_train, k=1, distance_metric='euclidean'):
    # Create a NearestNeighborsClassifier instance
    classifier = NearestNeighborsClassifier(k=k, distance_metric=distance_metric)

    # Fit the classifier to the training data
    classifier.fit(X_train, y_train)
     
    return classifier

def evaluate(labels, predictions):

    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)

    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1

        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1

    sensitivity /= total_positive
    specificity /= total_negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()

