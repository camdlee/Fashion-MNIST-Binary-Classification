#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

# need to calculate the Prior probability - represents how likely each class is to occur before we look at any features (pixels)
# What is the chance an image is a Trouser (or pullover) if I do not know anything about it?
# Matters because of potential class imbalance in data, Bayesian foundation requires prior probability as part of equation
# incorporates the existing knowledge of frequency of classes in prediction

# need likelihood calculation to indicate how compatible the observed pixel values are with each class
# in other words "If this image is a trouser, how likely would I be able to see this specific pattern of pixels?"
# Matters because it measures how well the observed features match what we'd expect for each class
# Some features (pixels) may be more important than others, like a center pixel vs an outer pixel
# forms core of predictive model - patterns that distinguish one class from another

# Naive Bayes combines both - Prior tells us what we know before looking at image (class frequencies),
# likelihood tells us how well the images matches each class pattern
# posterior is the final probability after considering both prior knowledge and evidence

# Prior Probabilities
n_samples = len(x_train_filtered)
classes = np.unique(y_train_filtered)
n_classes = len(classes)
print(f'')

# Initialize arrays to store our probabilities
class_priors = np.zeros(n_classes)
print(f'Class priors initialized: {class_priors}')
pixel_probs = np.zeros((n_classes, x_train_filtered.shape[1], 2))
print(f'Pixel probabilities initialized: {pixel_probs}')

for i, c in enumerate(classes):
    # every sample of this class
    x_c = x_train_filtered[y_train_filtered == c]
    print(f'Instance of class in training data: {x_c}')

    # prior probability = count of class / total samples in data
    class_priors[i] = len(x_c) / n_samples
    print(f'Calculating class prior for {i}, current length of x_c: {len(x_c)}, current n_samples: {n_samples}')
    print(f'Class prior {i} = {class_priors[i]}')

    # calc prob of each pixel being 1 for this class
    # add 1 for Laplace smoothing (avoid zero probabilities)
    n_samples_c = len(x_c)
    print(f'N samples c: {n_samples_c}')

    # count how many times each pixel is 1 in this class
    pixel_one_counts = np.sum(x_c, axis=0) + 1 # +1 is Laplace smoothing
    print(f'Pixel one counts: {pixel_one_counts}')

    # probability of pixel being 1 given the class (with laplace smoothing)
    pixel_probs[i, :, 1] = pixel_one_counts / (n_samples_c + 2)
    print(f'Pixel probability of being 1: {pixel_probs[i, :, 1]}')

    # probability of pixel being 0 given the class (with laplace smoothing)
    pixel_probs[i, :, 0] = 1 - pixel_probs[i, :, 1]
    print(f'Pixel probability of being 0: {pixel_probs[i, :, 0]}')

print("Prior probabilities: ")
for i, c in enumerate(classes):
    print(f'Class {c}: {class_priors[i]:.4f}')

# function to predict probabilities using our trained model
def predict_proba(x, classes, class_priors, pixel_probs):
    n_samples = x.shape[0]
    n_classes = len(classes)

    # initialize log probabilities array
    log_probs = np.zeros((n_samples, n_classes))

    # calculate log probabilities for each class
    for i, c in enumerate(classes):
        # start with log of class prior
        class_prior = np.log(class_priors[i])

        # calculate log likelihood for each pixel
        pixel_probs_for_values = np.zeros(x.shape)
        for j in range(n_samples):

            for k in range(x.shape[1]):
                # select either P(pixel=0|class) or P(pixel=1|class) based on actual pixel value
                pixel_probs_for_values[j, k] = pixel_probs[i, k, x[j, k]]

        # sum the log probabilities of all pixels
        log_probs[:, i] = class_prior + np.sum(np.log(pixel_probs_for_values), axis=1)

    # Convert from log probabilities to actual probabilities
    # Subtract max for numerical stability
    max_log_probs = np.max(log_probs, axis=1, keepdims=True)
    ex_probs = np.exp(log_probs - max_log_probs)
    probs = ex_probs / np.sum(ex_probs, axis=1, keepdims=True)

    return probs

# Function to predict class labels
def predict(x, classes, class_priors, pixel_probs):
    probs = predict_proba(x, classes, class_priors, pixel_probs)
    return classes[np.argmax(probs, axis=1)]

# Use prediction function to see what the predicted y values would be
y_train_pred = predict(x_train_filtered, classes, class_priors, pixel_probs)
y_test_pred = predict(x_test_filtered, classes, class_priors, pixel_probs)

# calculate the accuracy of the predictions to filtered data
# check y predicted against true values
train_accuracy = np.mean(y_train_filtered == y_train_pred)
test_accuracy = np.mean(y_test_filtered == y_test_pred)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")


# -------------------- Calculate Receiver Operating Characteristics (ROC) manually and plot ROC curve --------------------
# ROC curve is graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied
def calculate_roc_curve(y_true, probs, positive_class):
    # parameters : y_true=true labels, probs = predicted probabilities for the pos class, positive class = class to treat as positive
    # Convert to binary problem (1 for positive class, 0 for others)
    y_binary = (y_true == positive_class).astype(int)

    # sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    y_binary_sorted = y_binary[sorted_indices]

    # Calculated TPR and FPR for different thresholds
    # TPR = true positive rate = (true positives)/(total actual positives) = recall
    # FPR = false positive rate = (false positives)/(total actual negatives)
    tpr_list = []
    fpr_list = []

    tp = 0
    fp = 0
    n_positive = np.sum(y_binary)
    n_negative = len(y_binary) - n_positive

    # starting with threshold high enough that no samples are classified as positive
    tpr_list.append(0)
    fpr_list.append(0)

    # add point for each sample as we lower the threshold
    for i in range(len(y_binary_sorted)):
        if y_binary_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_positive
        fpr = fp / n_negative

        # Don't add duplicate posts (when multiple samples have same probability)
        if i < len(y_binary_sorted) - 1 and probs[sorted_indices[i]] == probs[sorted_indices[i + 1]]:
            continue

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate AUC using the trapezoidal rule
    area_under_curve = 0
    for i in range(1, len(fpr_list)):
        area_under_curve += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return tpr_list, fpr_list, area_under_curve

# calculate probabilities for the roc curve
test_probs = predict_proba(x_test_filtered, classes, class_priors, pixel_probs)
print(f'Probabilities for ROC curve: {test_probs}')

plt.figure(figsize=(10,6))

colors = ['blue', 'red']
class_names = ['Trouser', 'Pullover']

for i, c in enumerate(classes):
    # Get the column index for this class in the probabilities array
    class_idx = np.where(classes == c)[0][0]

    # Calculate ROC curve
    tpr, fpr, auc = calculate_roc_curve(y_test_filtered, test_probs[:, class_idx], c)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('naive_bayes_roc_curve.png')
plt.show()

# Find correctly classified and misclassified examples
correct_indices = np.where(y_test_pred == y_test_filtered)[0]
incorrect_indices = np.where(y_test_pred != y_test_filtered)[0]

# Display a few examples
if len(correct_indices) > 0:
    idx = correct_indices[0]
    class_name = "Trouser" if y_test_filtered[idx] == trouser_class else "Pullover"
    display_image(x_test_filtered[idx], f"Correctly Classified: {class_name}")

if len(incorrect_indices) > 0:
    idx = incorrect_indices[0]
    true_class = "Trouser" if y_test_filtered[idx] == trouser_class else "Pullover"
    pred_class = "Trouser" if y_test_pred[idx] == trouser_class else "Pullover"
    display_image(x_test_filtered[idx], f"Misclassified: True={true_class}, Pred={pred_class}")


### ====================== DECISION TREE MACHINE LEARNING MODEL ===========================
# Use Decision tree technique to differentiate classify samples between trouser and pullover
# You should use gini index for discriminatory features and maximum tree height should be 10

# initialize decision tree with gini index and max depth 10
dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=10)

# train decision tree
dt_classifier.fit(x_train_filtered, y_train_filtered)

# prediction
y_train_prediction_dt = dt_classifier.predict(x_train_filtered)
y_test_prediction_dt = dt_classifier.predict(x_test_filtered)

# evaluating accuracy of decision tree prediction
train_accuracy_dt = accuracy_score(y_train_filtered, y_train_prediction_dt)
test_accuracy_dt = accuracy_score(y_test_filtered, y_test_prediction_dt)

# calculating recall and precision
precision_dt = precision_score(y_test_filtered, y_train_prediction_dt)
recall_dt = recall_score(y_test_filtered, y_test_prediction_dt)

print(f'Decision Tree training accuracy: {train_accuracy_dt}')
print(f'Decision Tree testing accuracy: {test_accuracy_dt}')


### ====================== COMPARISON AND ANALYSIS ===========================
# Using both training and testing sets, compute class-wise accuracy, precision, and recall of both methods
#

# where the models differ in results / prediction
incorrect_decision_tree = np.where(y_test_prediction_dt != y_test_filtered)[0]
incorrect_naive_bayes = np.where(y_test_pred != y_test_filtered)[0]

for i in incorrect_decision_tree:
    if i not in incorrect_naive_bayes:
        print(f'Index {i}: Naive Bayes correct, Decision Tree incorrect')
        display_image(x_test_filtered[i], "Naive Bayes correct, DT wrong")
        break

for i in incorrect_naive_bayes:
    if i not in incorrect_decision_tree:
        print(f'Index {i}: Decision Tree correct, Naive Bayes incorrect')
        display_image(x_test_filtered[i], "Decision Tree correct, Naive Bayes incorrect")
        break


## Based on the accuracy on training and test sets, comment if any of the classifiers suffer from underfitting or overfitting. Explain why so.
# Neither classifiers suffer from underfitting nor overfitting. The NB training accuracy was 93.18% and the testing accuracy was 93.40%. The Decision Tree training accuracy was 99.16% and the testing accuracy was 97.35%.
# For either model to be underfitting, both the training and testing accuracy would have to be similar values that are low. However, both have a high accuracy percentage. Furthermore, for the models to be
# overfitted, they would have to be too accurate on the training data and less accurate on the testing data. However, both models have a relatively high accuracy for training and testing. There is a slightly greater difference
# between the Decision tree training and testing accuracy compared to the Naive Bayes accuracies, so the Decision tree may be slightly more overfitted compared to the Naive Bayes classifier.

## Based on the data distribution, should we choose accuracy as the main metric? or should it be precision or recall? Explain the reason.
## The main metric should be able to assess model’s correct performance (No diplomatic answers).

## Compare the strengths and weaknesses of each model. From the test set, you can share some examples correctly predicted by one model but mispredicted by another.