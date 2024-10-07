import pandas as pd


# Load CSV files
df_logistic = pd.read_csv('LogisticRegression/combined_model_results.csv')
df_rf = pd.read_csv('RandomForrest/combined_model_results_rf.csv')
df_svm = pd.read_csv('svm/combined_model_results_svm.csv')

# Add a column to specify the model
df_logistic['Model_Type'] = 'Logistic Regression'
df_rf['Model_Type'] = 'Random Forest'
df_svm['Model_Type'] = 'SVM'

# Combine the results
combined_df = pd.concat([df_logistic, df_rf, df_svm], ignore_index=True)

# Keep only one "Before SMOTE" row
before_smote_row = combined_df[combined_df['Model'] == 'Before SMOTE'].iloc[0].to_frame().T
combined_df = combined_df[combined_df['Model'] != 'Before SMOTE']
before_smote_row['Model_Type'] = ''
combined_df = pd.concat([combined_df, before_smote_row], ignore_index=True)

# Sort by Macro Accuracy
combined_df_sorted = combined_df.sort_values(by='Macro Accuracy', ascending=False)

# Save combined results to CSV
combined_df_sorted.to_csv('sorted_combined_model_results.csv', index=False)

print("Combined and sorted results saved to 'sorted_combined_model_results.csv'")