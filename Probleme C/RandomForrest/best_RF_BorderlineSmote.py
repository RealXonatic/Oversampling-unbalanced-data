import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from mlxtend.plotting import plot_decision_regions
from collections import Counter

# Loading the dataset
file_path = '/Users/jaheergoulam/Desktop/Implementation Pyton/dataset/main_file_RFDR_selected_features_1-3_New.csv'
df = pd.read_csv(file_path)

# Creating new classes: 1 remains 1, and 2, 3, 4 merge into 2
df['NEW_CLASS'] = df['CLASE'].apply(lambda x: 1 if x == 1 else 2)

# Function to perform PCA, apply BorderlineSMOTE, and calculate confusion matrix, global accuracy, and macro-accuracy
def plot_pca_and_metrics_with_borderlinesmote(df, n_iterations=20):
    all_results_before_borderlinesmote = []
    all_results_after_borderlinesmote = []

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
        
        # Parameter grid for Random Forest without BorderlineSMOTE
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
        }

        # Using make_scorer to define macro-accuracy as the scoring metric
        def macro_accuracy(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
            accuracy_class_1 = cm[0, 0] / cm[0].sum()
            accuracy_class_2 = cm[1, 1] / cm[1].sum() if cm.shape[0] > 1 else 0
            return (accuracy_class_1 + accuracy_class_2) / 2

        scorer = make_scorer(macro_accuracy)

        # Cross-validation for Random Forest without BorderlineSMOTE
        grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)

        # Training and finding the best Random Forest model without BorderlineSMOTE
        grid_search_rf.fit(X_train_pca, y_train)

        # Training the best Random Forest model without BorderlineSMOTE on all training data
        best_model_rf = grid_search_rf.best_estimator_
        best_model_rf.fit(X_train_pca, y_train)

        # Predicting and evaluating on test data for Random Forest without BorderlineSMOTE
        y_pred_rf = best_model_rf.predict(X_test_pca)

        # Calculating confusion matrix and accuracies after Random Forest without BorderlineSMOTE
        cm_after_rf = confusion_matrix(y_test, y_pred_rf, labels=[1, 2])
        accuracy_after_rf = accuracy_score(y_test, y_pred_rf)

        # Calculating accuracy for each class after Random Forest without BorderlineSMOTE
        accuracy_class_1_after_rf = cm_after_rf[0, 0] / cm_after_rf[0].sum() if cm_after_rf[0].sum() > 0 else 0
        accuracy_class_2_after_rf = cm_after_rf[1, 1] / cm_after_rf[1].sum() if cm_after_rf.shape[0] > 1 and cm_after_rf[1].sum() > 0 else 0
        macro_accuracy_after_rf = (accuracy_class_1_after_rf + accuracy_class_2_after_rf) / 2

        results_after_rf = {
            'Accuracy Globale': accuracy_after_rf,
            'Accuracy Classe 1': accuracy_class_1_after_rf,
            'Accuracy Classe 2': accuracy_class_2_after_rf,
            'Macro Accuracy': macro_accuracy_after_rf,
        }

        all_results_before_borderlinesmote.append(results_after_rf)

        # Calculate class counts for y_train
        class_counts = np.bincount(y_train)

        # Defining the pipeline for BorderlineSMOTE + Random Forest
        pipeline = Pipeline([
            ('borderlinesmote', BorderlineSMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Initial parameter grid for BorderlineSMOTE + Random Forest
        param_grid = {
            'borderlinesmote__k_neighbors': [5, 3],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [5, 10],
        }

        # Cross-validation for BorderlineSMOTE + Random Forest
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, n_jobs=-1, return_train_score=True)
        grid_search.fit(X_train, y_train)

        # Training the best model BorderlineSMOTE + Random Forest on all training data
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        #print(grid_search.best_params_) # To get the parameters of the best model

        # Predicting and evaluating on test data
        y_pred = best_model.predict(X_test)

        # Applying ADASYN to the training data for visualization after ADASYN
        X_res, y_res = best_model.named_steps['borderlinesmote'].fit_resample(X_train, y_train)
        X_res_pca = pca.transform(X_res)

        # Extracting the classifier parameters
        classifier_params = best_model.named_steps['classifier'].get_params()
        classifier_params.pop('random_state')

        # Extracting the classifier from the pipeline and training it on the PCA-transformed data
        classifier = RandomForestClassifier(random_state=42, **classifier_params)
        classifier.fit(X_res_pca, y_res)

        # Calculating confusion matrix and accuracies after BorderlineSMOTE
        cm_after_borderlinesmote = confusion_matrix(y_test, y_pred, labels=[1, 2])
        accuracy_after_borderlinesmote = accuracy_score(y_test, y_pred)

        # Calculating accuracy for each class after BorderlineSMOTE
        accuracy_class_1_after = cm_after_borderlinesmote[0, 0] / cm_after_borderlinesmote[0].sum() if cm_after_borderlinesmote[0].sum() > 0 else 0
        accuracy_class_2_after = cm_after_borderlinesmote[1, 1] / cm_after_borderlinesmote[1].sum() if cm_after_borderlinesmote.shape[0] > 1 and cm_after_borderlinesmote[1].sum() > 0 else 0
        macro_accuracy_after_borderlinesmote = (accuracy_class_1_after + accuracy_class_2_after) / 2

        results_after_borderlinesmote = {
            'Accuracy Globale': accuracy_after_borderlinesmote,
            'Accuracy Classe 1': accuracy_class_1_after,
            'Accuracy Classe 2': accuracy_class_2_after,
            'Macro Accuracy': macro_accuracy_after_borderlinesmote,
        }

        all_results_after_borderlinesmote.append(results_after_borderlinesmote)

    mean_results_before_borderlinesmote = pd.DataFrame(all_results_before_borderlinesmote).mean().to_dict()
    mean_results_after_borderlinesmote = pd.DataFrame(all_results_after_borderlinesmote).mean().to_dict()

    # Convert the lists to DataFrames
    df_before_borderlinesmote = pd.DataFrame(all_results_before_borderlinesmote)
    df_after_borderlinesmote = pd.DataFrame(all_results_after_borderlinesmote)

    # Add an index column starting from 1 to n_iterations
    df_before_borderlinesmote.insert(0, 'Iteration', range(1, len(df_before_borderlinesmote) + 1))
    df_after_borderlinesmote.insert(0, 'Iteration', range(1, len(df_after_borderlinesmote) + 1))

    # Add mean row to the DataFrames
    df_before_borderlinesmote.loc['Mean'] = df_before_borderlinesmote.mean()
    df_before_borderlinesmote['Iteration'] = df_before_borderlinesmote['Iteration'].astype(object)
    df_before_borderlinesmote.at['Mean', 'Iteration'] = 'Mean'

    df_after_borderlinesmote.loc['Mean'] = df_after_borderlinesmote.mean()
    df_after_borderlinesmote['Iteration'] = df_after_borderlinesmote['Iteration'].astype(object)
    df_after_borderlinesmote.at['Mean', 'Iteration'] = 'Mean'

    # Add a separator row for better readability
    separator_row = pd.Series([''] * len(df_before_borderlinesmote.columns), index=df_before_borderlinesmote.columns)
    combined_df = pd.concat([df_before_borderlinesmote, pd.DataFrame([separator_row]), df_after_borderlinesmote], ignore_index=True)

    # Save the combined DataFrame to CSV
    combined_df.to_csv('combined_results_borderlinesmote.csv', index=False)

    # Visualizations before and after BorderlineSMOTE
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Decision regions plot
    plt.figure(figsize=(12, 6))

    # Plotting decision regions before BorderlineSmote
    plt.subplot(1, 2, 1)
    plot_decision_regions(X_reduced, y, clf=best_model_rf, legend=2)
    plt.title('Régions de décision avant BorderlineSmote')

    # Plotting decision regions after BorderlineSmote
    plt.subplot(1, 2, 2)
    plot_decision_regions(X_res_pca, y_res, clf=classifier, legend=2)
    plt.title('Régions de décision après BorderlineSmote')
    
    plt.tight_layout()
    plt.savefig('visualisation_pca_borderlinesmote_random_forest.png')
    plt.show()

    return mean_results_before_borderlinesmote, mean_results_after_borderlinesmote, df_before_borderlinesmote, df_after_borderlinesmote, combined_df

# Generate results
mean_results_before_borderlinesmote, mean_results_after_borderlinesmote, df_before_borderlinesmote, df_after_borderlinesmote, combined_df = plot_pca_and_metrics_with_borderlinesmote(df)

print("Mean results before BorderlineSMOTE:")
print(mean_results_before_borderlinesmote)

print("Mean results after BorderlineSMOTE:")
print(mean_results_after_borderlinesmote)

print("Results saved to 'combined_results_borderlinesmote.csv' and PCA visualizations saved as 'visualisation_pca_borderlinesmote_random_forest.png'")