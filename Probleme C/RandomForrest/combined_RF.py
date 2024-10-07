import pandas as pd


# Load CSV files
df_adasyn_rf = pd.read_csv('combined_results_adasyn_rf.csv')
df_smote_rf = pd.read_csv('combined_results_smote.csv')
df_borderline_smote_rf = pd.read_csv('combined_results_borderlinesmote.csv')
df_smote_enn_rf = pd.read_csv('combined_results_smote_enn.csv')
df_random_oversampler_rf = pd.read_csv('combined_results_random_oversampler.csv')

# Extract mean results after applying the methods
mean_adasyn_rf_after = df_adasyn_rf[df_adasyn_rf['Iteration'] == 'Mean'].iloc[1].copy()
mean_smote_rf_after = df_smote_rf[df_smote_rf['Iteration'] == 'Mean'].iloc[1].copy()
mean_borderline_smote_rf_after = df_borderline_smote_rf[df_borderline_smote_rf['Iteration'] == 'Mean'].iloc[1].copy()
mean_smote_enn_rf_after = df_smote_enn_rf[df_smote_enn_rf['Iteration'] == 'Mean'].iloc[1].copy()
mean_random_oversampler_rf_after = df_random_oversampler_rf[df_random_oversampler_rf['Iteration'] == 'Mean'].iloc[1].copy()

# Extract results before SMOTE (assuming these are the first rows with 'Mean' in the 'Iteration' column)
mean_results_before_smote_rf = df_adasyn_rf[df_adasyn_rf['Iteration'] == 'Mean'].iloc[0].copy()

# Add a column for the model name
mean_adasyn_rf_after['Model'] = 'ADASYN'
mean_smote_rf_after['Model'] = 'SMOTE'
mean_borderline_smote_rf_after['Model'] = 'Borderline SMOTE'
mean_smote_enn_rf_after['Model'] = 'SMOTE-ENN'
mean_random_oversampler_rf_after['Model'] = 'RandomOverSampler'
mean_results_before_smote_rf['Model'] = 'Before SMOTE'

# Combine results
combined_results_rf = pd.concat([mean_adasyn_rf_after.to_frame().T, mean_smote_rf_after.to_frame().T, mean_borderline_smote_rf_after.to_frame().T, mean_smote_enn_rf_after.to_frame().T, mean_random_oversampler_rf_after.to_frame().T], ignore_index=True)

# Sort by Macro Accuracy
combined_results_rf = combined_results_rf.sort_values(by='Macro Accuracy', ascending=False)

# Append the mean results before SMOTE at the end
combined_results_rf = pd.concat([combined_results_rf, mean_results_before_smote_rf.to_frame().T], ignore_index=True)

# Reset index
combined_results_rf.reset_index(drop=True, inplace=True)

# Add an index column
combined_results_rf.index.name = 'Rank'

# Save combined results to CSV
combined_results_rf.to_csv('combined_model_results_rf.csv', index=True)

print("Results saved to 'combined_model_results_rf.csv'")