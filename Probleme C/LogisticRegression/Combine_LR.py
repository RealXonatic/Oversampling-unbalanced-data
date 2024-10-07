import pandas as pd


# Load CSV files
df_adasyn = pd.read_csv('combined_results_adasyn.csv')
df_smote = pd.read_csv('combined_results_smote.csv')
df_borderline_smote = pd.read_csv('combined_results_borderline_smote.csv')
df_smote_enn = pd.read_csv('combined_results_smoteenn.csv')
df_random_oversampler = pd.read_csv('combined_results_randomoversampler.csv')

# Extract mean results after applying the methods
mean_adasyn_after = df_adasyn[df_adasyn['Iteration'] == 'Mean'].iloc[1].copy()
mean_smote_after = df_smote[df_smote['Iteration'] == 'Mean'].iloc[1].copy()
mean_borderline_smote_after = df_borderline_smote[df_borderline_smote['Iteration'] == 'Mean'].iloc[1].copy()
mean_smote_enn_after = df_smote_enn[df_smote_enn['Iteration'] == 'Mean'].iloc[1].copy()
mean_random_oversampler_after = df_random_oversampler[df_random_oversampler['Iteration'] == 'Mean'].iloc[1].copy()

# Extract results before SMOTE (assuming these are the first rows with 'Mean' in the 'Iteration' column)
mean_results_before_smote = df_adasyn[df_adasyn['Iteration'] == 'Mean'].iloc[0].copy()

# Add a column for the model name
mean_adasyn_after['Model'] = 'ADASYN'
mean_smote_after['Model'] = 'SMOTE'
mean_borderline_smote_after['Model'] = 'Borderline SMOTE'
mean_smote_enn_after['Model'] = 'SMOTE-ENN'
mean_random_oversampler_after['Model'] = 'RandomOverSampler'
mean_results_before_smote['Model'] = 'Before SMOTE'

# Combine results
combined_results = pd.concat([mean_adasyn_after.to_frame().T, mean_smote_after.to_frame().T, mean_borderline_smote_after.to_frame().T, mean_smote_enn_after.to_frame().T, mean_random_oversampler_after.to_frame().T], ignore_index=True)

# Sort by Macro Accuracy
combined_results = combined_results.sort_values(by='Macro Accuracy', ascending=False)

# Append the mean results before SMOTE at the end
combined_results = pd.concat([combined_results, mean_results_before_smote.to_frame().T], ignore_index=True)

# Reset index
combined_results.reset_index(drop=True, inplace=True)

# Add an index column
combined_results.index.name = 'Rank'

# Save combined results to CSV
combined_results.to_csv('combined_model_results.csv', index=True)


print("Results saved to 'highlighted_model_results.csv'")