# KNN-Classifier

**Overview**

The script utilizes a KNN model for classifying individuals in terms of disease diagnosis. The process encompasses data preprocessing, model training, and evaluation using various performance metrics. Additionally, visualizations such as ROC curves and confusion matrices are generated to interpret the model's performance.

**Dataset**

The dataset Final_Preprocessed_data.csv comprises features significant to disease diagnosis. It is segregated into training and test sets for model validation.

**Features**

The dataset includes a mix of categorical and numerical features. Categorical features undergo one-hot encoding, while numerical features are scaled for optimal model performance.

**Model**

A K-Nearest Neighbors classifier from scikit-learn is employed, known for its efficacy in classification tasks.

**Performance Metrics**

Model performance is evaluated using metrics like:

Accuracy
F1 Score
Precision
Confusion Matrix
ROC (Receiver Operating Characteristic) curve
AUC (Area Under the Curve)

**Visualization**

Two types of visualizations are generated:

ROC Curve: Depicts the trade-off between true positive rate and false positive rate.
Confusion Matrix: Showcases the number of true positives, true negatives, false positives, and false negatives.
