import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy, precision, recall, and F1 score for the given true and predicted labels.
    Returns None if input arrays are empty.
    """
    if not true_labels or not predicted_labels:
        return None, None, None, None
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return accuracy, precision, recall, f1

def clean_text(text):
    """
    Clean the text by removing punctuation and converting to lowercase.
    """
    return re.sub(r'[^\w\s]', '', text.lower())

def determine_label(pred):
    """
    Assign a label based on the presence of '1' or '0' in the prediction.
    Returns:
        - 1 for fully accepted (if '1' is found)
        - 0 for rejected (if '0' is found)
        - None for ambiguous (if neither '1' nor '0' is found)
    """
    # Check if '1' or '0' is in the prediction
    if '1' in pred:
        return 1  # Fully accepted
    elif '0' in pred:
        return 0  # Rejected
    else:
        return None  # Ambiguous (neither '1' nor '0' found)

def class_wise_accuracy(true_labels, predicted_labels):
    """
    Calculate class-wise accuracy based on the confusion matrix.
    """
    confusion_matrix_results = confusion_matrix(true_labels, predicted_labels)
    class_wise_accuracy = []
    for i in range(len(confusion_matrix_results)):
        # Avoid division by zero if no samples exist for a class
        class_accuracy = confusion_matrix_results[i][i] / sum(confusion_matrix_results[i]) if sum(confusion_matrix_results[i]) > 0 else 0
        class_wise_accuracy.append(class_accuracy)
    return class_wise_accuracy

# Read the CSV file
df = pd.read_csv('Pedex_result_phi_q_1.csv')  # Provide the correct path to your CSV file
print(df.columns)

# Extract predicted labels and actual labels
# Assuming the 'Label' column is numerical and the 'llama2_pred' column contains the predictions
df = df.dropna(subset=['Label'])  # Drop any rows where the 'Label' column is NaN

# Convert 'Label' to integers if it's not already
df['Label'] = pd.to_numeric(df['Label'], errors='coerce')

# Filter out rows where conversion failed and the label is NaN
df = df.dropna(subset=['Label'])

pred = df['phi3_pred_exp'].tolist()
actual = df['Label'].astype(int).tolist()  # Ensuring 'Label' column is an integer
#pred = [determine_label(i) for i in pred_list]

# Ensure lengths match
print(len(actual), len(pred))

# Filter out only 'clear acceptance' (1) and 'clear rejection' (0) labels
a1 = []
p1 = []
for i, e in enumerate(pred):
    print(i, e)
    if e == 1 or e == 0:  # Only consider clear acceptance or rejection
        a1.append(actual[i])
        p1.append(e)

# Print the filtered lengths
print(len(a1), len(p1))

# Check if there are valid items to calculate metrics
if len(a1) == 0 or len(p1) == 0:
    print("No valid data to calculate metrics. Please check the filtering conditions.")
else:
    # Calculate the metrics
    accuracy, precision, recall, f1 = calculate_metrics(a1, p1)

    # Output the metrics
    print("Accuracy:", accuracy)
    print("Macro Precision:", precision)
    print("Macro Recall:", recall)
    print("Macro F1-score:", f1)

    # Calculate and print the class-wise accuracy
    class_accuracy = class_wise_accuracy(a1, p1)
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} accuracy: {acc:.4f}")
