# Shopping

The nearest Neighbors Classifier doesn't construct a model during the training phase. It stores the training dataset in memory and uses this stored information to predict outcomes for new data points by comparing them to the training examples based on the distance between them.

#Usage

To use the Nearest Neighbors Classifier with Euclidean and Manhattan distance metrics, follow these steps:

1. **Clone the Repository:**
   - Make sure you have Python installed on your system. Clone this GitHub repository to your local machine or download the code as a ZIP file and extract it.

2. **Install Dependencies:**
   - You may need to install the required libraries, such as NumPy and scikit-learn.

3. Execute the main() function to run the classifier.

4. The results include the number of correct and incorrect predictions, true positive rate (sensitivity), true negative rate (specificity), and F1 score for each distance metric.

While the Euclidean distance-based Nearest Neighbors Classifier showed stronger true positive and F1 scores for positive instances, the Manhattan distance-based classifier performed better in terms of true negatives (signifying its proficiency in accurately identifying negative instances). The choice between these metrics should be tailored to the specific requirements of the classification task.
