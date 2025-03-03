#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from numpy import trapz

### ================== DATA PREPARATION ====================
print("Loading and preparing dataset...")
# Load training data
train_images_df = pd.read_csv('train_images.csv', header=None)
x_train = train_images_df.values
train_labels_df = pd.read_csv('train_labels.csv', header=None)
y_train = train_labels_df.values.ravel()

# Load testing data
test_images_df = pd.read_csv('test_images.csv', header=None)
x_test = test_images_df.values
test_labels_df = pd.read_csv('test_labels.csv', header=None)
y_test = test_labels_df.values.ravel()

# Reshaping images for visualization if needed (28 x 28 pixels)
# x_train_reshaped = x_train.reshape(-1, 28, 28)
# x_test_reshaped = x_test.reshape(-1, 28, 28)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Filtering Data for Trouser(1) and Pullover(2)
train_filter = np.where((y_train == 1) | (y_train == 2))
test_filter = np.where((y_test == 1) | (y_test == 2))

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

# For viz, reshape filtered data
x_train_filtered_reshaped = x_train_filtered.reshape(-1, 28, 28)
x_test_filtered_reshaped = x_test_filtered.reshape(-1, 28, 28)

print(f"Filtered dataset size: {x_train_filtered.shape[0]} training samples, {x_test_filtered.shape[0]} testing samples")

# Sample distribution
train_class_counts = np.bincount(y_train_filtered)[1:3]
# test_class_counts = np.bincount(y_test_filtered)[1:3]
print(f'Training set class distribution: Trouser: {train_class_counts[0]}, Pullover: {train_class_counts[1]}')

# Binarization - select a threshold, 'del', to be 127
def binarize_images(images, threshold=127):
    # convert grayscale images 0-255 into binary, either 1 or 0
    return (images >= threshold).astype(int)

# Default binarization with threshold 127
x_train_bin = binarize_images(x_train_filtered)
x_test_bin = binarize_images(x_test_filtered)

# Convert classes to binary
y_train_binary = (y_train_filtered == 2).astype(int)
y_test_binary = (y_test_filtered == 2).astype(int)

print(f'Binarized feature dimensions: {x_train_bin.shape[1]} features per image')

# Visualizing binarization
idx = 0

plt.subplot(1, 2, 1)
plt.imshow(x_train_filtered[idx].reshape(28, 28), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(x_train_filtered[idx].reshape(28, 28), cmap='gray_r', vmin=0, vmax=1)
plt.title("Binarized Images")

plt.show()

# # Reshape images into feature vectors
# x_train_features = x_train_bin.reshape(x_train_bin.shape[0], -1)
# # print(f'X Train Features: {x_train_features}')
# y_train_binary = (y_train_filtered == 2).astype(int)
# # print(f'Class Labels as binary: {y_train_binary}')



### ============== NAIVE BAYES CLASSIFICATION ===============
print("\nImplementing Naive Bayes from scratch...")
# Apply Naive Bayes Classification to the data
# Using Naive Bayes, we will classify if the image belongs in 'Trouser' or in 'Pullover'
# We must compute P(class/features) through code - no library allowed
# Make Class Predictions based on the higher value of these computed probabilities

def train_naive_bayes(x, y):
    """
    Train Naive Bayes Classifier
    :param x:
    :param y:
    :return:
    - prior: prior probabilities for each class
    - likelihood: likelihood probabilities P(feature/class)
    - classes: unique class labels
    """
    classes = np.unique(y)
    n_samples, n_features = x.shape
    n_classes = len(classes)

    # calculating prior probabilities P(class)
    prior = np.zeros(n_classes)
    for i, c in enumerate(classes):
        prior[i] = np.sum(y == c) / n_samples

    # Calculate likelihood P(features/class) with Laplace smoothing
    likelihood = np.zeros((n_classes, n_features, 2))
    for i, c in enumerate(classes):
        x_c = x[y == c]
        for j in range(n_features):
            # Count occurrences of feature values (0, 1) for this class
            count_0 = np.sum(x_c[:, j] == 0)
            count_1 = np.sum(x_c[:, j] == 1)
            total = x_c.shape[0]

            # Apply laplace smoothing
            likelihood[i, j, 0] = (count_0 + 1) / (total+2)
            likelihood[i, j, 1] = (count_1 + 1) / (total+2)

    return prior, likelihood, classes

def predict_proba_naive_bayes(x, prior, likelihood, classes):
    """
    Calculate the class probabilities using Naive Bayes
    :param x:
    :param prior:
    :param likelihood:
    :param classes:
    :return: probability for each sample and each class
    """
    n_samples = x.shape[0]
    n_classes = len(classes)

    # initialize probabilities array
    probas = np.zeros((n_samples, n_classes))

    # Calculate P(class|features) for each sample
    for i in range(n_samples):
        for c in range(n_classes):
            # Start with prior probability (in log space to avoid underflow)
            log_proba = np.log(prior[c])

            # Multiply by likelihood of each feature
            for j in range(x.shape[1]):
                feature_value = x[i, j]
                log_proba += np.log(likelihood[c, j, feature_value])

            probas[i, c] = log_proba

    # Convert log probabilities to probabilities and normalize
    max_log_proba = np.max(probas, axis=1, keepdims=True)
    probas = np.exp(probas - max_log_proba)  # Subtract max for numerical stability
    probas /= np.sum(probas, axis=1, keepdims=True)

    return probas

def predict_naive_bayes(X, prior, likelihood, classes):
    """
    Make class predictions using Naive Bayes
    Returns:
    - predictions: predicted class for each sample
    """
    probas = predict_proba_naive_bayes(X, prior, likelihood, classes)
    return np.argmax(probas, axis=1)


# Train Naive Bayes model using functions
prior_nb, likelihood_nb, classes_nb = train_naive_bayes(x_train_bin, y_train_binary)

# Predict on train and test sets
y_train_pred_nb = predict_naive_bayes(x_train_bin, prior_nb, likelihood_nb, classes_nb)
y_test_pred_nb = predict_naive_bayes(x_test_bin, prior_nb, likelihood_nb, classes_nb)

# Get probability predictions for ROC curve
y_train_prob_nb = predict_proba_naive_bayes(x_train_bin, prior_nb, likelihood_nb, classes_nb)[:, 1]
y_test_prob_nb = predict_proba_naive_bayes(x_test_bin, prior_nb, likelihood_nb, classes_nb)[:, 1]


# ========== ROC CURVE IMPLEMENTATION ==========
print("Creating ROC curve from scratch...")

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    """Plot ROC curve from scratch and calculate AUC"""
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate TPR and FPR
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate AUC
    auc = 0
    for i in range(len(fpr_list) - 1):
        auc += (fpr_list[i+1] - fpr_list[i]) * (tpr_list[i+1] + tpr_list[i]) / 2

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} (AUC = {auc:.3f})')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()

    return auc
