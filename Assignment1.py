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
train_labels_df = pd.read_csv('train_labels.csv', header=None)
x_train = train_images_df.values
y_train = train_labels_df.values.ravel()

# Load testing data
test_images_df = pd.read_csv('test_images.csv', header=None)
test_labels_df = pd.read_csv('test_labels.csv', header=None)
x_test = test_images_df.values
y_test = test_labels_df.values.ravel()

print(f'Num of rows, and columns in training images data : {x_train.shape}')
print(f'Num of rows, and columns in training labels data : {y_train.shape}')

print(f'First entry in training data : {x_train[1]}')

# Printing out the unique classes in our label files. Should match with below
print("Class distribution in training set: ", np.unique(y_train))
# Class distribution in training set:  [0 1 2 3 4 5 6 7 8 9]
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Filtering Data for Trouser(1) and Pullover(2)
# train_filter = np.where((y_train == 1) | (y_train == 2))
# test_filter = np.where((y_test == 1) | (y_test == 2))

# x_train_filtered = x_train[train_filter]
# y_train_filtered = y_train[train_filter]
# x_test_filtered = x_test[test_filter]
# y_test_filtered = y_test[test_filter]

# print(f"Filtered dataset size: {x_train_filtered.shape[0]} training samples, {x_test_filtered.shape[0]} testing samples")

### ================== BINARIZATION OF DATA ====================
# Threshold, 'del', to be 127
def binarize_images(images, threshold=127):
    # convert grayscale images 0-255 into binary, either 1 or 0
    return (images >= threshold).astype(int)

# Default binarization with threshold 127
x_train_bin = binarize_images(x_train)
# print(f'Training Data Binarized: {x_train_bin}')
print(f'First entry in binarized training data: {x_train_bin[1]}')
x_test_bin = binarize_images(x_test)
print(f'Testing Data Binarized: {x_test_bin}')

trouser_class = 1
pullover_class = 2

# Filter to only have trousers and pullovers
filtering_indices_train = np.where((y_train == trouser_class) | (y_train == pullover_class))
filtering_indices_test = np.where((y_test == trouser_class) | (y_test == pullover_class))

# Applying filter to images and labels data
x_train_filtered = x_train_bin[filtering_indices_train]
y_train_filtered = y_train[filtering_indices_train]

x_test_filtered = x_test_bin[filtering_indices_test]
y_test_filtered = y_test[filtering_indices_test]

# Printing to see what our data looks like now
print(f'Number of training samples (Trousers and Pullovers only): {len(x_train_filtered)}')
print(f'Number of testing samples (Trousers and Pullovers only): {len(x_test_filtered)}')
print(f'Classes in filtered training set: {np.unique(y_train_filtered)}')

# Let's visualize one example of each class after filtering and binarization
trouser_idx = np.where(y_train_filtered == trouser_class)[0][0]
pullover_idx = np.where(y_train_filtered == pullover_class)[0][0]

def display_image(pixels, title):
    # Reshape the 784 pixels to 28x28 image
    img = pixels.reshape(28, 28)
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='binary')
    plt.title(title)
    plt.show()

display_image(x_train_filtered[trouser_idx], f'Binarized Trouser (Class {trouser_class})')
display_image(x_train_filtered[pullover_idx], f'Binarized Pullover (Class {pullover_class})')



# ### ============== NAIVE BAYES CLASSIFICATION ===============
print("\nImplementing Naive Bayes from scratch...")
# # Apply Naive Bayes Classification to the data
# # Using Naive Bayes, we will classify if the image belongs in 'Trouser' or in 'Pullover'
# # We must compute P(class/features) through code - no library allowed
# # Make Class Predictions based on the higher value of these computed probabilities
#
# def train_naive_bayes(x, y):
#     """
#     Train Naive Bayes Classifier
#     :param x:
#     :param y:
#     :return:
#     - prior: prior probabilities for each class
#     - likelihood: likelihood probabilities P(feature/class)
#     - classes: unique class labels
#     """
#     classes = np.unique(y)
#     n_samples, n_features = x.shape
#     n_classes = len(classes)
#
#     # calculating prior probabilities P(class)
#     prior = np.zeros(n_classes)
#     for i, c in enumerate(classes):
#         prior[i] = np.sum(y == c) / n_samples
#
#     # Calculate likelihood P(features/class) with Laplace smoothing
#     likelihood = np.zeros((n_classes, n_features, 2))
#     for i, c in enumerate(classes):
#         x_c = x[y == c]
#         for j in range(n_features):
#             # Count occurrences of feature values (0, 1) for this class
#             count_0 = np.sum(x_c[:, j] == 0)
#             count_1 = np.sum(x_c[:, j] == 1)
#             total = x_c.shape[0]
#
#             # Apply laplace smoothing
#             likelihood[i, j, 0] = (count_0 + 1) / (total+2)
#             likelihood[i, j, 1] = (count_1 + 1) / (total+2)
#
#     return prior, likelihood, classes
#
# def predict_proba_naive_bayes(x, prior, likelihood, classes):
#     """
#     Calculate the class probabilities using Naive Bayes
#     :param x:
#     :param prior:
#     :param likelihood:
#     :param classes:
#     :return: probability for each sample and each class
#     """
#     n_samples = x.shape[0]
#     n_classes = len(classes)
#
#     # initialize probabilities array
#     probas = np.zeros((n_samples, n_classes))
#
#     # Calculate P(class|features) for each sample
#     for i in range(n_samples):
#         for c in range(n_classes):
#             # Start with prior probability (in log space to avoid underflow)
#             log_proba = np.log(prior[c])
#
#             # Multiply by likelihood of each feature
#             for j in range(x.shape[1]):
#                 feature_value = x[i, j]
#                 log_proba += np.log(likelihood[c, j, feature_value])
#
#             probas[i, c] = log_proba
#
#     # Convert log probabilities to probabilities and normalize
#     max_log_proba = np.max(probas, axis=1, keepdims=True)
#     probas = np.exp(probas - max_log_proba)  # Subtract max for numerical stability
#     probas /= np.sum(probas, axis=1, keepdims=True)
#
#     return probas
#
# def predict_naive_bayes(X, prior, likelihood, classes):
#     """
#     Make class predictions using Naive Bayes
#     Returns:
#     - predictions: predicted class for each sample
#     """
#     probas = predict_proba_naive_bayes(X, prior, likelihood, classes)
#     return np.argmax(probas, axis=1)
#
#
# # Train Naive Bayes model using functions
# prior_nb, likelihood_nb, classes_nb = train_naive_bayes(x_train_bin, y_train_binary)
#
# # Predict on train and test sets
# y_train_pred_nb = predict_naive_bayes(x_train_bin, prior_nb, likelihood_nb, classes_nb)
# y_test_pred_nb = predict_naive_bayes(x_test_bin, prior_nb, likelihood_nb, classes_nb)
#
# # Get probability predictions for ROC curve
# y_train_prob_nb = predict_proba_naive_bayes(x_train_bin, prior_nb, likelihood_nb, classes_nb)[:, 1]
# y_test_prob_nb = predict_proba_naive_bayes(x_test_bin, prior_nb, likelihood_nb, classes_nb)[:, 1]
#
#
# # ========== ROC CURVE IMPLEMENTATION ==========
# print("Creating ROC curve from scratch...")
#
# def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
#     """Plot ROC curve from scratch and calculate AUC"""
#     thresholds = np.linspace(0, 1, 100)
#     tpr_list = []
#     fpr_list = []
#
#     for threshold in thresholds:
#         y_pred = (y_prob >= threshold).astype(int)
#
#         # Calculate TPR and FPR
#         tp = np.sum((y_pred == 1) & (y_true == 1))
#         fp = np.sum((y_pred == 1) & (y_true == 0))
#         tn = np.sum((y_pred == 0) & (y_true == 0))
#         fn = np.sum((y_pred == 0) & (y_true == 1))
#
#         tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
#         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#
#         tpr_list.append(tpr)
#         fpr_list.append(fpr)
#
#     # Calculate AUC
#     auc = 0
#     for i in range(len(fpr_list) - 1):
#         auc += (fpr_list[i+1] - fpr_list[i]) * (tpr_list[i+1] + tpr_list[i]) / 2
#
#     # Plot ROC curve
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr_list, tpr_list, 'b-', linewidth=2)
#     plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'{title} (AUC = {auc:.3f})')
#     plt.grid(True)
#
#     if save_path:
#         plt.savefig(save_path)
#
#     plt.show()
#
#     return auc
