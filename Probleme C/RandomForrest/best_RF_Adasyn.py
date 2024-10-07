import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from mlxtend.plotting import plot_decision_regions
from collections import Counter

# Loading the dataset
file_path = '/Users/jaheergoulam/Desktop/Implementation Pyton/dataset/main_file_RFDR_selected_features_1-3_New.csv'
df = pd.read_csv(file_path)

# Creating new classes: 1 remains 1, and 2, 3, 4 merge into 2
df['NEW_CLASS'] = df['CLASE'].apply(lambda x: 1 if x == 1 else 2)

# Function to perform PCA, apply ADASYN, and calculate confusion matrix, global accuracy, and macro-accuracy
def plot_pca_and_metrics_with_adasyn(df, n_iterations=20):
    all_results_before_adasyn = []
    all_results_after_adasyn = []

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

        # Parameter grid for Random Forest without ADASYN
        param_grid_rf = {
            'n_estimators': [50, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [4, 10],
            'min_samples_leaf': [2, 4]
        }

        # Using make_scorer to define macro-accuracy as the scoring metric
        def macro_accuracy(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
            accuracy_class_1 = cm[0, 0] / cm[0].sum()
            accuracy_class_2 = cm[1, 1] / cm[1].sum() if cm.shape[0] > 1 else 0
            return (accuracy_class_1 + accuracy_class_2) / 2

        scorer = make_scorer(macro_accuracy)

        # Cross-validation for Random Forest without ADASYN
        grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)

        # Training and finding the best Random Forest model without ADASYN
        grid_search_rf.fit(X_train_pca, y_train)

        # Training the best Random Forest model without ADASYN on PCA-transformed data
        best_model_rf = grid_search_rf.best_estimator_
        best_model_rf.fit(X_train_pca, y_train)

        # Predicting and evaluating on PCA-transformed test data
        y_pred_rf = best_model_rf.predict(X_test_pca)

        # Calculating confusion matrix and accuracies after Random Forest without ADASYN
        cm_after_rf = confusion_matrix(y_test, y_pred_rf, labels=[1, 2])
        accuracy_after_rf = accuracy_score(y_test, y_pred_rf)

        # Calculating accuracy for each class after Random Forest without ADASYN
        accuracy_class_1_after_rf = cm_after_rf[0, 0] / cm_after_rf[0].sum() if cm_after_rf[0].sum() > 0 else 0
        accuracy_class_2_after_rf = cm_after_rf[1, 1] / cm_after_rf[1].sum() if cm_after_rf.shape[0] > 1 and cm_after_rf[1].sum() > 0 else 0
        macro_accuracy_after_rf = (accuracy_class_1_after_rf + accuracy_class_2_after_rf) / 2

        results_after_rf = {
            'Accuracy Globale': accuracy_after_rf,
            'Accuracy Classe 1': accuracy_class_1_after_rf,
            'Accuracy Classe 2': accuracy_class_2_after_rf,
            'Macro Accuracy': macro_accuracy_after_rf,
        }

        all_results_before_adasyn.append(results_after_rf)

        # Calculate class counts for y_train
        class_counts = np.bincount(y_train)

        # Defining the pipeline for ADASYN + Random Forest
        pipeline = Pipeline([
            ('adasyn', ADASYN(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Initial parameter grid for ADASYN + Random Forest
        param_grid = {
            'adasyn__n_neighbors': [5, 3],
            'classifier__n_estimators': [50, 150],
            'classifier__max_depth': [None, 10, 25],
            'classifier__min_samples_split': [4, 10],
            'classifier__min_samples_leaf': [2, 4]
        }

        # Cross-validation for ADASYN + Random Forest
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Training the best model ADASYN + Random Forest on all training data
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        #print(grid_search.best_params_) #To get the parameters of the best model

        # Predicting and evaluating on test data
        y_pred = best_model.predict(X_test)

        # Applying ADASYN to the training data for visualization after ADASYN
        X_res, y_res = best_model.named_steps['adasyn'].fit_resample(X_train, y_train)
        X_res_pca = pca.transform(X_res)

        # Extracting the classifier parameters
        classifier_params = best_model.named_steps['classifier'].get_params()
        classifier_params.pop('random_state')

        # Extracting the classifier from the pipeline and training it on the PCA-transformed data
        classifier = RandomForestClassifier(random_state=42, **classifier_params)
        classifier.fit(X_res_pca, y_res)

        # Calculating confusion matrix and accuracies after ADASYN
        cm_after_adasyn = confusion_matrix(y_test, y_pred, labels=[1, 2])
        accuracy_after_adasyn = accuracy_score(y_test, y_pred)

        # Calculating accuracy for each class after ADASYN
        accuracy_class_1_after = cm_after_adasyn[0, 0] / cm_after_adasyn[0].sum() if cm_after_adasyn[0].sum() > 0 else 0
        accuracy_class_2_after = cm_after_adasyn[1, 1] / cm_after_adasyn[1].sum() if cm_after_adasyn.shape[0] > 1 and cm_after_adasyn[1].sum() > 0 else 0
        macro_accuracy_after_adasyn = (accuracy_class_1_after + accuracy_class_2_after) / 2

        results_after_adasyn = {
            'Accuracy Globale': accuracy_after_adasyn,
            'Accuracy Classe 1': accuracy_class_1_after,
            'Accuracy Classe 2': accuracy_class_2_after,
            'Macro Accuracy': macro_accuracy_after_adasyn,
        }

        all_results_after_adasyn.append(results_after_adasyn)

    mean_results_before_adasyn = pd.DataFrame(all_results_before_adasyn).mean().to_dict()
    mean_results_after_adasyn = pd.DataFrame(all_results_after_adasyn).mean().to_dict()

    # Convert the lists to DataFrames
    df_before_adasyn = pd.DataFrame(all_results_before_adasyn)
    df_after_adasyn = pd.DataFrame(all_results_after_adasyn)

    # Add an index column starting from 1 to n_iterations
    df_before_adasyn.insert(0, 'Iteration', range(1, len(df_before_adasyn) + 1))
    df_after_adasyn.insert(0, 'Iteration', range(1, len(df_after_adasyn) + 1))

    # Add mean row to the DataFrames
    df_before_adasyn.loc['Mean'] = df_before_adasyn.mean()
    df_before_adasyn['Iteration'] = df_before_adasyn['Iteration'].astype(object)
    df_before_adasyn.at['Mean', 'Iteration'] = 'Mean'

    df_after_adasyn.loc['Mean'] = df_after_adasyn.mean()
    df_after_adasyn['Iteration'] = df_after_adasyn['Iteration'].astype(object)
    df_after_adasyn.at['Mean', 'Iteration'] = 'Mean'

    # Add a separator row for better readability
    separator_row = pd.Series([''] * len(df_before_adasyn.columns), index=df_before_adasyn.columns)
    combined_df = pd.concat([df_before_adasyn, pd.DataFrame([separator_row]), df_after_adasyn], ignore_index=True)

    # Save the combined DataFrame to CSV
    combined_df.to_csv('combined_results_adasyn_rf.csv', index=False)

    # Visualizations using plot_decision_regions before and after ADASYN
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Decision regions plot
    plt.figure(figsize=(12, 6))

    # Plotting decision regions before ADASYN
    plt.subplot(1, 2, 1)
    plot_decision_regions(X_reduced, y, clf=best_model_rf, legend=2)
    plt.title('Régions de décision avant ADASYN')

    # Plotting decision regions after ADASYN
    plt.subplot(1, 2, 2)
    plot_decision_regions(X_res_pca, y_res, clf=classifier, legend=2)
    plt.title('Régions de décision après ADASYN')

    plt.tight_layout()
    plt.savefig('decision_regions_adasyn_rf.png')
    plt.show()

    return mean_results_before_adasyn, mean_results_after_adasyn, df_before_adasyn, df_after_adasyn, combined_df

# Generate results
mean_results_before_adasyn, mean_results_after_adasyn, df_before_adasyn, df_after_adasyn, combined_df = plot_pca_and_metrics_with_adasyn(df)

print("Mean results before ADASYN:")
print(mean_results_before_adasyn)

print("Mean results after ADASYN:")
print(mean_results_after_adasyn)

print("Results saved to 'combined_results_adasyn_rf.csv' and decision regions visualizations saved as 'decision_regions_adasyn_rf.png'")