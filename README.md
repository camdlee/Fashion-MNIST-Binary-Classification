# Fashion-MNIST Binary Classification

Project Overview

This project implements a machine learning pipeline for image classification using the Fashion-MNIST dataset (specifically trousers vs. pullovers). It explores multiple classification approaches and evaluates their performance using standard metrics. The work is structured as a Jupyter Notebook with supporting Python code.

Key Features

Data Preprocessing

Loads training and testing datasets from CSV files.

Performs binarization of images by applying a threshold (127) to pixel values.

Includes utility functions to display and visualize images.

Implemented Functions

binarize_images: Converts grayscale image data into binary form.

display_image: Helper function to visualize individual images from the dataset.

calculate_roc_curve: Computes ROC curve points and calculates the area under the curve (AUC) for model evaluation.

Models & Techniques

Naive Bayes Classification: Applies probabilistic classification on the binarized dataset.

Decision Tree Classifier: Implements a decision tree with the Gini index as the splitting criterion and a maximum depth of 10 to classify trousers vs. pullovers.

Evaluation Metrics

Accuracy of classifiers on training and test sets.

ROC Curve and AUC (Area Under Curve) for comparing model performance.

Technologies Used

Python (NumPy, Pandas, Matplotlib, Scikit-learn)

Jupyter Notebook for development and visualization