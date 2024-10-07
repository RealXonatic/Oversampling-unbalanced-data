import pandas as pd


# Load CSV files
df_adasyn_svm = pd.read_csv('combined_results_adasyn.csv')
df_smote_svm = pd.read_csv('combined_results_smote.csv')
df_borderline_smote_svm = pd.read_csv('combined_results_borderlinesmote.csv')
df_smote_enn_svm = pd.read_csv('combined_results_smote_enn.csv')
df_random_oversampler_svm = pd.read_csv('combined_results_randomoversampler.csv')

# Extract mean results after applying the methods
mean_adasyn_svm_after = df_adasyn_svm[df_adasyn_svm['Iteration'] == 'Mean'].iloc[1].copy()
mean_smote_svm_after = df_smote_svm[df_smote_svm['Iteration'] == 'Mean'].iloc[1].copy()
mean_borderline_smote_svm_after = df_borderline_smote_svm[df_borderline_smote_svm['Iteration'] == 'Mean'].iloc[1].copy()
mean_smote_enn_svm_after = df_smote_enn_svm[df_smote_enn_svm['Iteration'] == 'Mean'].iloc[1].copy()
mean_random_oversampler_svm_after = df_random_oversampler_svm[df_random_oversampler_svm['Iteration'] == 'Mean'].iloc[1].copy()

# Extract results before SMOTE (assuming these are the first rows with 'Mean' in the 'Iteration' column)
mean_results_before_smote_svm = df_adasyn_svm[df_adasyn_svm['Iteration'] == 'Mean'].iloc[0].copy()

# Add a column for the model name
mean_adasyn_svm_after['Model'] = 'ADASYN'
mean_smote_svm_after['Model'] = 'SMOTE'
mean_borderline_smote_svm_after['Model'] = 'Borderline SMOTE'
mean_smote_enn_svm_after['Model'] = 'SMOTE-ENN'
mean_random_oversampler_svm_after['Model'] = 'RandomOverSampler'
mean_results_before_smote_svm['Model'] = 'Before SMOTE'

# Combine results
combined_results_svm = pd.concat([mean_adasyn_svm_after.to_frame().T, mean_smote_svm_after.to_frame().T, mean_borderline_smote_svm_after.to_frame().T, mean_smote_enn_svm_after.to_frame().T, mean_random_oversampler_svm_after.to_frame().T], ignore_index=True)

# Sort by Macro Accuracy
combined_results_svm = combined_results_svm.sort_values(by='Macro Accuracy', ascending=False)

# Append the mean results before SMOTE at the end
combined_results_svm = pd.concat([combined_results_svm, mean_results_before_smote_svm.to_frame().T], ignore_index=True)

# Reset index
combined_results_svm.reset_index(drop=True, inplace=True)

# Add an index column
combined_results_svm.index.name = 'Rank'

# Save combined results to CSV
combined_results_svm.to_csv('combined_model_results_svm.csv', index=True)

print("Results saved to 'combined_model_results_svm.csv'")