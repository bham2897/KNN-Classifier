import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = '/Users/divya/Desktop/ DAPM Charts /Final_Preprocessed_data.csv'  
data = pd.read_csv(file_path)

# Encoding the target variable
label_encoder = LabelEncoder()
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

# Separating features and target variable
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Splitting into training and test sets (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Identifying categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Creating preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creating the KNN model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Training the model
model.fit(X_train, y_train)

# Function to evaluate the model and plot the confusion matrix
def evaluate_and_plot_confusion_matrix(X, y, dataset_name):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    
    # Print the evaluation metrics
    print(f"{dataset_name} data: Accuracy: {accuracy:.2f}")
    print(f"{dataset_name} data: F1 Score: {f1:.2f}")
    print(f"{dataset_name} data: Precision: {precision:.2f}")
    print(f"{dataset_name} data: Confusion Matrix:\n{conf_matrix}\n")
    
    # Plot Confusion Matrix
    plt.figure()
    ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(values_format='d')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.show()

# Function to plot ROC curves for both training and test data
def plot_combined_roc_curve(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curve for training data
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
    roc_auc_train = roc_auc_score(y_train, y_train_pred_proba)
    
    # Calculate ROC curve for test data
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
    
    # Plot both ROC curves
    plt.plot(fpr_train, tpr_train, label=f'Training ROC curve (area = {roc_auc_train:.2f})', color='blue')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (area = {roc_auc_test:.2f})', color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    
    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    # Show the ROC curve plot
    plt.show()

# Evaluate and plot confusion matrix for training data
evaluate_and_plot_confusion_matrix(X_train, y_train, "Training")

# Evaluate and plot confusion matrix for test data
evaluate_and_plot_confusion_matrix(X_test, y_test, "Test")

# Plot the combined ROC curve
plot_combined_roc_curve(X_train, y_train, X_test, y_test)
