# Through the Youth Eyes: Training Depression Detection Algorithms with Eye Tracking Data
### Repository ID: 9049

## Authors

- Derick A. Lagunes-Ramírez
- Gabriel González-Serna
- Leonor Rivera-Rivera
- Nimrod González-Franco
- María Y. Hernández-Pérez
- José A. Reyes-Ortiz

## Files

This repository contains the files needed to replicate the results presented in the paper _Through the Youth Eyes: Training Depression Detection Algorithms with Eye Tracking Data_.

### **selectedFeaturesDataset.csv**

This dataset file in CSV format contains data from eye-tracking tests involving 172 young participants aged 15 to 21 years (mean age = 16.53) randomly selected from two high schools in Mexico. After excluding 33 participants based on inclusion criteria (age between 15 and 21, no medication use, no eye-related issues, and consent to participate), the dataset comprises 139 participants. They are classified into two groups based on their DASS-21 depression scale scores: **Healthy Control (n=82)** and **Depressed (n=57)**.

The data were cleaned, and the most significant features were selected using U-tests. To balance classes, data augmentation techniques were applied. The final dataset includes 21 selected features, including:

- **Pupil size metrics**: Maximum values, skewness, and kurtosis for both the left and right pupils.
- **Eye fixation metrics**: Fixation count, variance and standard deviation of both X and Y coordinates, maximum and minimum values of the X coordinate (side-to-side view), and mean, skewness, and kurtosis of the Y coordinate (up-and-down view).
- **Additional metrics**: Blink count, blink rate per minute, saccade velocity, and saccade latency.

The target variable `Class` (0 for the control group, 1 for the depressed group) is also included, resulting in a dataset with 228 data vectors (114 for each group).

This dataset was used to train supervised machine learning models for binary classification, aiming to predict whether a participant is classified as "depressed" or "healthy control" based on their eye-tracking data.

### **MLTest.py**

This Python 3 script implements and evaluates four machine learning algorithms using the **selectedFeaturesDataset.csv** dataset.

#### Algorithms Used

The following machine learning algorithms were chosen for this study:

- **Support Vector Machine (SVM)**: Selected for its strong performance in high-dimensional spaces.
- **Random Forest (RF)**: Known for its robustness and ability to reduce overfitting.
- **Multilayer Perceptron (MLP)**: A neural network model chosen for its ability to capture complex, non-linear patterns.
- **Gradient Boosting (GB)**: Selected for its high accuracy and sequential error-correcting nature.

These models were evaluated and compared in terms of accuracy, generalization, and interpretability.

#### Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** from Scikit-learn to find the optimal parameters for each algorithm. The dataset was split into training and testing sets (80% for training, 20% for testing), and stratified sampling was used to maintain the same class proportions in both sets.

A portion of the training data was set aside for cross-validation. The best hyperparameters were selected based on training set performance, and the final model was trained with the optimal parameters and evaluated on the testing set.

#### Hyperparameters Used

The fine-tuned hyperparameters for each algorithm are as follows:

- **SVM**:
  - Regularization parameter \( C = 0.1 \)
  - Polynomial kernel, degree = 4
  - Coefficient \( \text{coef0} = 1 \)
  - Gamma = 0.1
  
- **Random Forest**:
  - max_depth = None (unlimited)
  - min_samples_leaf = 1
  - min_samples_split = 2
  - n_estimators = 200
  - random_state = 11

- **Gradient Boosting**:
  - Learning rate = 0.1
  - max_depth = 3
  - min_samples_split = 3
  - n_estimators = 200

- **MLP**:
  - Batch size = 16
  - Epochs = 30
  - Dropout rate = 0.25
  - Hidden layer sizes = (64, 32, 16, 8)
  - Optimizer = rmsprop
  - Learning rate = 0.001

To use the script, ensure you have the necessary Python libraries specified at the start of the `.py` file, then execute the script. After processing, two `.txt` files will be generated, reporting the results of all ML algorithms with performance metrics, including accuracy, recall, F1 score, and MCC.
