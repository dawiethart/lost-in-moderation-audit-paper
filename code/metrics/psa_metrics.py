import os

import pandas as pd
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.cm as cm

##### Compile Aggreate PSA Classification #####

# Load csv containing the labels
token_real = pd.read_csv('../../input/psa/psa_real_data_majority.csv')
token_synthetic = pd.read_csv('../../input/psa/psa_synthetic_data_majority.csv')
token_real

# change psa_id to id
token_synthetic = token_synthetic.rename(columns={'psa_id': 'id'}).copy()
token_real = token_real.rename(columns={'psa_id': 'id'}).copy()

# Base directory where the API folders are located
base_dir = '../../output/API_classification'

# Initialize two empty DataFrames to collect results for real and synthetic datasets
psa_merged_real = pd.DataFrame()
psa_merged_synthetic = pd.DataFrame()

# List of APIs
apis = ['amazon', 'google', 'microsoft', 'gpt']

for api in apis:
    # Paths for the real dataset
    real_maj_path = os.path.join(base_dir, api, 'real', f'{api}_real_maj.csv')
    real_min_path = os.path.join(base_dir, api, 'real', f'{api}_real_min.csv')

    # Paths for the synthetic dataset
    synthetic_maj_path = os.path.join(base_dir, api, 'synthetic', f'{api}_synthetic_maj.csv') 
    synthetic_min_path = os.path.join(base_dir, api, 'synthetic', f'{api}_synthetic_min.csv')
    
    final_columns = ['api', 'id', 'text_maj', 'text_min', 'score_maj', 'score_min', 'score_difference', 'true_label_maj']

    # Process real data
    df_real_maj = pd.read_csv(real_maj_path)
    df_real_min = pd.read_csv(real_min_path)
    
    if df_real_maj.columns.str.contains('ids').any():	
        df_real_maj['id'] = df_real_maj.ids

    if df_real_min.columns.str.contains('ids').any():	
        df_real_min['id']  = df_real_min.ids

    merged_real = pd.merge(df_real_maj, df_real_min, on='id', suffixes=('_maj', '_min'))
    merged_real['score_difference'] = merged_real['score_maj'] - merged_real['score_min']
    merged_real['api'] = api
    merged_real = merged_real[final_columns]
    psa_merged_real = pd.concat([psa_merged_real, merged_real], ignore_index=True)
    
    # Process synthetic data if available
    df_synthetic_maj = pd.read_csv(synthetic_maj_path)
    df_synthetic_min = pd.read_csv(synthetic_min_path)
    merged_synthetic = pd.merge(df_synthetic_maj, df_synthetic_min, on='id', suffixes=('_maj', '_min'))
    merged_synthetic['score_difference'] = merged_synthetic['score_maj'] - merged_synthetic['score_min']
    merged_synthetic['api'] = api
    merged_synthetic = merged_synthetic[final_columns]
    psa_merged_synthetic = pd.concat([psa_merged_synthetic, merged_synthetic], ignore_index=True)

psa_merged_real.columns
psa_merged_synthetic.columns

# count observations by api
psa_merged_real.groupby('api').size()
psa_merged_synthetic.groupby('api').size()

# merge with token data to get group column
psa_merged_real = pd.merge(psa_merged_real, token_real[['id', 'group']], on='id')
psa_merged_real

# print only google merged real
psa_merged_real[psa_merged_real['api'] == 'microsoft']

psa_merged_synthetic = pd.merge(psa_merged_synthetic, token_synthetic[['id', 'group']], on='id')
psa_merged_synthetic

# inspect groups
psa_merged_real['group'].value_counts()

# drop white and christian
psa_merged_real = psa_merged_real[psa_merged_real['group'] != 'white']
psa_merged_real = psa_merged_real[psa_merged_real['group'] != 'christians']


##### Compute PSA Statistics #####

# Function to compute mean and 95% confidence interval

def compute_psa_statistics(data):
    results_list = []
    
    # Group by both 'group' and 'api'
    for (api, group, true_label_maj), group_data in data.groupby(['api', 'group', 'true_label_maj']):
        mean = group_data['score_difference'].mean()
        sem = group_data['score_difference'].sem()  # Standard Error of the Mean
        ci_lower, ci_upper = stats.t.interval(0.95, df=len(group_data) - 1, loc=mean, scale=sem)
        error = ci_upper - mean
        
        # Append results for this group and API to the list as a dictionary
        results_list.append({
            'api': api,
            'group': group,
            'toxic': true_label_maj,
            'mean_difference': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'error': error,
            'N': len(group_data)
        })

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df

# For real
grouped_real_stats = compute_psa_statistics(psa_merged_real)
grouped_real_stats

# For synthetics
grouped_synthetic_stats = compute_psa_statistics(psa_merged_synthetic)
grouped_synthetic_stats

##### Visualise #####

# Define output directory
output_dir = "../../output/psa"

# Define a continuous color map and marker styles for the APIs
color_map = plt.get_cmap('viridis')  # A continuous colormap
markers = ['o', 's', '^', '*']  # Different markers for each API

def plot_and_save(grouped_data, data_type, x_axis_limits):
    groups = grouped_data['group'].unique().tolist()
    toxic_values = grouped_data['toxic'].unique()

    # Define the colors based on groups
    steps = np.linspace(0, 1, len(groups))
    colors = {group: color_map(step) for group, step in zip(groups, steps)}   
    upper_groups = [group.upper() for group in groups]

    for toxic_value in toxic_values:
        toxic_data = grouped_data[grouped_data['toxic'] == toxic_value]
        api_data = {}
        for api in toxic_data['api'].unique():
            api_data[api] = {
                'mean_difference': toxic_data[toxic_data['api'] == api]['mean_difference'].tolist(),
                'mean_error': toxic_data[toxic_data['api'] == api]['error'].tolist(),
                'group': toxic_data[toxic_data['api'] == api]['group'].tolist()
            }

        fig, ax = plt.subplots(figsize=(12, 8))
        # API Labels with "OpenAI" instead of "GPT"
        api_labels = ['Amazon', 'Google', 'OpenAI', 'Microsoft']
        for i, (api, marker) in enumerate(zip(api_data.keys(), markers)):
            trans = Affine2D().translate(0.0, -i * 0.2 + 0.2 * (len(api_data) - 1) / 2) + ax.transData
            
            for j, group in enumerate(api_data[api]['group']):
                color = colors[group]  # Assign color based on the group
                ax.errorbar(api_data[api]['mean_difference'][j], 
                            [group.upper()], 
                            xerr=api_data[api]['mean_error'][j], 
                            marker=marker, 
                            mec=color, markerfacecolor=color, linestyle="none", 
                            label=api_labels[i] if j == 0 else "", 
                            ecolor=color, transform=trans, markersize=8)

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

        plt.xlabel(f'Mean Counterfactual Token Fairness Score', fontsize=18)
        plt.legend(fontsize=14)  # Adjusted font size for the legend
        plt.xticks(fontsize=14) 
        plt.yticks(fontsize=14)

        plt.xlim(x_axis_limits)

        # Remove grid lines
        ax.grid(False)

        # Modernize the look by using a white background and removing unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        output_file_path = os.path.join(output_dir, f'{data_type}_toxic_{toxic_value}_plot.png')
        plt.savefig(output_file_path, bbox_inches='tight', dpi=300)

# Plot and save for real data
plot_and_save(grouped_real_stats, 'real', (-0.15, 0.15))

# Plot and save for synthetic data
plot_and_save(grouped_synthetic_stats, 'synthetic', (-0.3, 0.3))
