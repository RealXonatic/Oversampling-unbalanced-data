import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from mlxtend.plotting import plot_decision_regions

from collections import Counter

# Loading the dataset
file_path = '/Users/jaheergoulam/Desktop/Implementation Pyton/dataset/main_file_RFDR_selected_features_1-3_New.csv'
df = pd.read_csv(file_path)

# Creating new classes: 1 remains 1, and 2, 3, 4 merge into 2
df['NEW_CLASS'] = df['CLASE'].apply(lambda x: 1 if x == 1 else 2)

# Function to perform PCA, apply SMOTE-ENN, and calculate confusion matrix, global accuracy, and macro-accuracy
def plot_pca_and_metrics_with_smoteenn(df, n_iterations=20):
    all_results_before_smoteenn = []
    all_results_after_smoteenn = []

    for i in range(n_iterations):
        # Splitting features and target
        X = df.drop(columns=['CLASE', 'OJO', 'USER', 'NEW_CLASS']).values
        y = df['NEW_CLASS'].values

        # Splitting into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        # Apply PCA
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Parameter grid for SVM without SMOTE-ENN
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        }

        # Using make_scorer to define macro-accuracy as the scoring metric
        def macro_accuracy(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
            accuracy_class_1 = cm[0, 0] / cm[0].sum()
            accuracy_class_2 = cm[1, 1] / cm[1].sum() if cm.shape[0] > 1 else 0
            return (accuracy_class_1 + accuracy_class_2) / 2

        scorer = make_scorer(macro_accuracy)

        # Cross-validation for SVM without SMOTE-ENN
        grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)

        # Training and finding the best SVM model without SMOTE-ENN
        grid_search_svm.fit(X_train_pca, y_train)

        # Training the best SVM model without SMOTE-ENN on all training data
        best_model_svm = grid_search_svm.best_estimator_
        best_model_svm.fit(X_train_pca, y_train)

        # Predicting and evaluating on test data for SVM without SMOTE-ENN
        y_pred_svm = best_model_svm.predict(X_test_pca)

        # Calculating confusion matrix and accuracies after SVM without SMOTE-ENN
        cm_after_svm = confusion_matrix(y_test, y_pred_svm, labels=[1, 2])
        accuracy_after_svm = accuracy_score(y_test, y_pred_svm)

        # Calculating accuracy for each class after SVM without SMOTE-ENN
        accuracy_class_1_after_svm = cm_after_svm[0, 0] / cm_after_svm[0].sum() if cm_after_svm[0].sum() > 0 else 0
        accuracy_class_2_after_svm = cm_after_svm[1, 1] / cm_after_svm[1].sum() if cm_after_svm.shape[0] > 1 and cm_after_svm[1].sum() > 0 else 0
        macro_accuracy_after_svm = (accuracy_class_1_after_svm + accuracy_class_2_after_svm) / 2

        results_after_svm = {
            'Accuracy Globale': accuracy_after_svm,
            'Accuracy Classe 1': accuracy_class_1_after_svm,
            'Accuracy Classe 2': accuracy_class_2_after_svm,
            'Macro Accuracy': macro_accuracy_after_svm,
        }

        all_results_before_smoteenn.append(results_after_svm)

        # Calculate class counts for y_train
        class_counts = np.bincount(y_train)

        # Defining the pipeline for SMOTE-ENN + SVM
        pipeline = Pipeline([
            ('smoteenn', SMOTEENN(random_state=42)),
            ('classifier', SVC(random_state=42))
        ])

        # Initial parameter grid for SMOTE-ENN + SVM
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        }

        # Cross-validation for SMOTE-ENN + SVM
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Training the best model SMOTE-ENN + SVM on all training data
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Predicting and evaluating on test data
        y_pred = best_model.predict(X_test)

        # Applying ADASYN to the training data for visualization after ADASYN
        X_res, y_res = best_model.named_steps['smoteenn'].fit_resample(X_train, y_train)
        X_res_pca = pca.transform(X_res)

        # Extracting the classifier parameters
        classifier_params = best_model.named_steps['classifier'].get_params()
        classifier_params.pop('random_state')

        # Extracting the classifier from the pipeline and training it on the PCA-transformed data
        classifier = SVC(random_state=42, **classifier_params)
        classifier.fit(X_res_pca, y_res)

        # Calculating confusion matrix and accuracies after SMOTE-ENN
        cm_after_smoteenn = confusion_matrix(y_test, y_pred, labels=[1, 2])
        accuracy_after_smoteenn = accuracy_score(y_test, y_pred)

        # Calculating accuracy for each class after SMOTE-ENN
        accuracy_class_1_after = cm_after_smoteenn[0, 0] / cm_after_smoteenn[0].sum() if cm_after_smoteenn[0].sum() > 0 else 0
        accuracy_class_2_after = cm_after_smoteenn[1, 1] / cm_after_smoteenn[1].sum() if cm_after_smoteenn.shape[0] > 1 and cm_after_smoteenn[1].sum() > 0 else 0
        macro_accuracy_after_smoteenn = (accuracy_class_1_after + accuracy_class_2_after) / 2

        results_after_smoteenn = {
            'Accuracy Globale': accuracy_after_smoteenn,
            'Accuracy Classe 1': accuracy_class_1_after,
            'Accuracy Classe 2': accuracy_class_2_after,
            'Macro Accuracy': macro_accuracy_after_smoteenn,
        }

        all_results_after_smoteenn.append(results_after_smoteenn)

    mean_results_before_smoteenn = pd.DataFrame(all_results_before_smoteenn).mean().to_dict()
    mean_results_after_smoteenn = pd.DataFrame(all_results_after_smoteenn).mean().to_dict()

    # Convert the lists to DataFrames
    df_before_smoteenn = pd.DataFrame(all_results_before_smoteenn)
    df_after_smoteenn = pd.DataFrame(all_results_after_smoteenn)

    # Add an index column starting from 1 to n_iterations
    df_before_smoteenn.insert(0, 'Iteration', range(1, len(df_before_smoteenn) + 1))
    df_after_smoteenn.insert(0, 'Iteration', range(1, len(df_after_smoteenn) + 1))

    # Add mean row to the DataFrames
    df_before_smoteenn.loc['Mean'] = df_before_smoteenn.mean()
    df_before_smoteenn['Iteration'] = df_before_smoteenn['Iteration'].astype(object)
    df_before_smoteenn.at['Mean', 'Iteration'] = 'Mean'

    df_after_smoteenn.loc['Mean'] = df_after_smoteenn.mean()
    df_after_smoteenn['Iteration'] = df_after_smoteenn['Iteration'].astype(object)
    df_after_smoteenn.at['Mean', 'Iteration'] = 'Mean'

    # Add a separator row for better readability
    separator_row = pd.Series([''] * len(df_before_smoteenn.columns), index=df_before_smoteenn.columns)
    combined_df = pd.concat([df_before_smoteenn, pd.DataFrame([separator_row]), df_after_smoteenn], ignore_index=True)

    # Save the combined DataFrame to CSV
    combined_df.to_csv('combined_results_smote_enn.csv', index=False)

    # Visualizations before and after SMOTE-ENN
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Decision regions plot
    plt.figure(figsize=(12, 6))

    # Plotting decision regions before smoteenn
    plt.subplot(1, 2, 1)
    plot_decision_regions(X_reduced, y, clf=best_model_svm, legend=2)
    plt.title('Régions de décision avant smoteenn')

    # Plotting decision regions after smoteenn
    plt.subplot(1, 2, 2)
    plot_decision_regions(X_res_pca, y_res, clf=classifier, legend=2)
    plt.title('Régions de décision après smoteenn')

    plt.tight_layout()
    plt.savefig('visualisation_pca_smote_enn_svm.png')
    plt.show()

    return mean_results_before_smoteenn, mean_results_after_smoteenn, df_before_smoteenn, df_after_smoteenn, combined_df

# Generate results
mean_results_before_smoteenn, mean_results_after_smoteenn, df_before_smoteenn, df_after_smoteenn, combined_df = plot_pca_and_metrics_with_smoteenn(df)

print("Mean results before SMOTE-ENN:")
print(mean_results_before_smoteenn)

print("Mean results after SMOTE-ENN:")
print(mean_results_after_smoteenn)

print("Results saved to 'combined_results_smote_enn.csv' and PCA visualizations saved as 'visualisation_pca_smote_enn_svm.png'")