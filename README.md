# DETDS

The Depression Eye-Tracking Data Set (DETDS) is a dataset collected from 172 young participants aged 15 to 21 years (mean = 16.53), randomly selected from two different high schools in Mexico. Ethical approval for the study was obtained from Mexico’s Instituto Nacional de Salud Pública (INSP) ethics committee, with informed consent provided by all participants, their parents, and the school authorities.

After excluding 33 participants based on the inclusion criteria (age between 15 and 21, no medication use, no eye-related issues, and consent to participate), the dataset consists of 139 participants, classified into two groups based on their scores from the DASS-21 depression scale: **Healthy Control (n=82)** and **Depressed (n=57)**.

The data were cleaned, and the most significant features were selected using U-tests. Based on these results, eye-tracking metrics that showed significant differences between the two groups were included in the final dataset, which consists of 21 features. These features include:

- **Pupil size metrics**: Maximum values, skewness, and kurtosis for both the left and right pupils.
- **Eye fixation metrics**: Fixation count, variance and standard deviation of both X and Y coordinates, maximum and minimum values of the X coordinate (side-to-side view), and mean, skewness, and kurtosis of the Y coordinate (up-and-down view).
- **Additional metrics**: Blink count, blink rate per minute, saccade velocity, and saccade latency.

The dataset also includes a target variable `Class` (0 for the control group, 1 for the depressed group), consisting of 228 data vectors (114 for each group).

This dataset was used to train supervised machine learning models for binary classification, with the goal of predicting whether a participant is classified as "depressed" or "healthy control" based on their eye-tracking data.

### Algorithms Used

The following machine learning algorithms were selected for this study:

- **Support Vector Machine (SVM)**: Chosen for its strong performance in high-dimensional spaces.
- **Random Forest (RF)**: Known for its robustness and ability to reduce overfitting.
- **Multilayer Perceptron (MLP)**: A neural network model chosen for its ability to capture complex, non-linear patterns.
- **Gradient Boosting (GB)**: Selected for its high accuracy and sequential nature, which corrects errors iteratively.

These models were evaluated and compared in terms of accuracy, generalization, and interpretability.

### Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** from Scikit-learn to find the optimal parameter settings for each algorithm. The dataset was split into training and testing sets (80% for training, 20% for testing), and stratified sampling was used to maintain the same proportion of each class in both sets.

A portion of the training data was set aside for cross-validation, and the best hyperparameters were selected based on the performance on the training set. The final model was trained using the full training set with the selected hyperparameters, and its performance was evaluated on the testing set.

### Hyperparameters Used

The hyperparameters for each algorithm were fine-tuned as follows:

- **SVM**:
  - Regularization parameter C = 0.1
  - Polynomial kernel, degree = 4
  - Coefficient coef0 = 1
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

### Results

The models were evaluated based on their classification performance, with metrics including accuracy, precision, recall, F1-score, and Matthews Correlation Coefficient (MCC). These metrics provide insights into the models' ability to correctly classify the participants into the two groups based on their eye-tracking data.

---

For more details, see the dataset and the accompanying Python code used for model training and evaluation.