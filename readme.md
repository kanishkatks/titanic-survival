# Titanic Survival Prediction (SVC Optimization)

This project aims to predict passenger survival on the RMS Titanic using the classic dataset available through Seaborn. The notebook walks through a typical machine learning workflow, focusing on data preprocessing and optimizing a Support Vector Classifier (SVC) using Randomized Search Cross-Validation.

## Workflow

The analysis follows these key steps:

1.  **Data Loading & Cleaning:**
    *   The Titanic dataset is loaded using `seaborn.load_dataset('titanic')`.
    *   The redundant `alive` column is dropped.
    *   Duplicate rows are removed using `data.drop_duplicates()`.
    *   Missing values are assessed. The `deck` column is dropped due to a high number of missing values.
    *   Remaining missing values in `age`, `embarked`, and `embark_town` are imputed using `sklearn.impute.SimpleImputer` with the `most_frequent` strategy.

2.  **Train/Test Split:**
    *   The data is split into features (`X`) and the target variable (`y` - 'survived').
    *   `sklearn.model_selection.train_test_split` is used to create training (70%) and testing (30%) sets.

3.  **Feature Encoding:**
    *   Categorical features (`sex`, `embarked`, `class`, `who`, `adult_male`, `embark_town`, `alone`) are identified.
    *   `sklearn.preprocessing.OneHotEncoder` is applied to these features on the training set (`fit_transform`) and then transformed on the test set (`transform`). Key parameters used:
        *   `drop='if_binary'`: Avoids multicollinearity for binary features.
        *   `sparse_output=False`: Returns a dense NumPy array.
        *   `handle_unknown='ignore'`: Prevents errors if the test set contains categories not seen in training.
    *   The original categorical columns are dropped, and the encoded features are concatenated back to the respective DataFrames (`features_train`, `features_test`).

4.  **Feature Scaling:**
    *   `sklearn.preprocessing.RobustScaler` is chosen for scaling, as it's less sensitive to outliers.
    *   The scaler is fitted *only* on the training features (`fit`) and then applied to both training and test sets (`transform`).

5.  **Baseline Model:**
    *   A simple `sklearn.svm.SVC` with a `kernel='linear'` is evaluated using 5-fold cross-validation (`sklearn.model_selection.cross_val_score`) on the scaled training data to establish a baseline accuracy.

6.  **Hyperparameter Tuning:**
    *   `sklearn.model_selection.RandomizedSearchCV` is used to optimize the `SVC` model.
    *   The search space includes:
        *   `kernel`: ['rbf']
        *   `C`: Continuous uniform distribution (e.g., `stats.uniform(0.01, 100)`)
        *   `gamma`: Log-uniform distribution (e.g., `stats.loguniform(0.001, 100)`)
    *   The search runs for a specified number of iterations (`n_iter`) with 5-fold cross-validation (`cv=5`), optimizing for `accuracy`.

7.  **Evaluation:**
    *   The best estimator found by `RandomizedSearchCV` (`random_search.best_estimator_`) is selected.
    *   This optimized model is used to make predictions on the *scaled test set*.
    *   Performance is evaluated using:
        *   `sklearn.metrics.classification_report`: Shows precision, recall, F1-score, and support for each class.
        *   `sklearn.metrics.ConfusionMatrixDisplay`: Visualizes the confusion matrix on the test set.

## Results

*   **Baseline Accuracy:** The initial linear SVC achieved a cross-validated accuracy of approximately **[Insert your baseline cross_val_score result here, e.g., ~80.1%]** on the training data.
*   **Tuned Model Accuracy (CV):** Randomized Search identified optimal hyperparameters, resulting in a cross-validated accuracy of **[Insert your random_search.best_score_ result here, e.g., ~82%]** on the training data.
*   **Test Set Performance:** The final tuned model achieved an accuracy of **[Insert accuracy from your classification_report on the test set]** on the unseen test data. The classification report and confusion matrix provide further details on its predictive performance for both survivors and non-survivors.

## Technologies Used

*   Python 3
*   Pandas
*   NumPy
*   Seaborn
*   Scikit-learn
*   Jupyter Notebook / Lab

## How to Run

1.  Ensure you have the required libraries installed (`pip install pandas numpy seaborn scikit-learn jupyterlab matplotlib`).
2.  Clone the repository or download the `.ipynb` file.
3.  Run the Jupyter Notebook cells sequentially. The notebook handles data loading, preprocessing, model training, tuning, and evaluation.
