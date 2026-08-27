import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

#load data 
HX = pd.read_json("hatexplain.json")
HX = HX.transpose()

#Renname columns end index
HX = HX.rename(columns={'post_id':'ids', 'post_tokens': 'text'})
HX['id'] = ['HX' + str(i) for i in range(1, len(HX) + 1)]
HX.set_index(['id'], inplace=True)
HX['text'] = [" ".join(HX.iloc[i]["text"]) for i in range(len(HX))]

#Sort by groups 
identity_groups = ['Asian', 'Islam', 'Jewish', 'African', 'Homosexual', 'Women', 'Caucasian', 'None']

targets_HX = [[HX.iloc[i]["annotators"][j]['target'] for j in range(3)] for i in range(len(HX))]
targets_HX = [targets_HX[i][0]+targets_HX[i][1]+targets_HX[i][2] for i in range(len(HX))]

label = []
target = []
for i in range(len(HX)):
    t = [targets_HX[i].count(identity_groups[j]) for j in range(8)]
    if t.count(max(t)) == 1:
        target.append(identity_groups[t.index(max(t))])
    else:
        target.append('Unequal')
    labels_HX = [HX.iloc[i]["annotators"][j]['label'] for j in range(3)]
    if labels_HX.count("normal") == 3:
        label.append(0)
    elif labels_HX.count("normal") == 0:
        label.append(1)
    else:
        label.append(2)

HX['true_label'] = label
HX['group'] = target

#Get where entries with unambigous group-identity and label(two or more agreeing)
HX = HX[(HX['true_label'] < 2) & (HX['group'] != 'Unequal')]

def load_data(dataset_name, api):
    file_paths = {
        'gpt': f'../../output/API_classification/gpt/{dataset_name}_aggregate/gpt_{dataset_name}_aggregate.csv',
        'amazon': f'../../output/API_classification/amazon/{dataset_name}_aggregate/amazon_{dataset_name}_aggregate.csv',
        'google': f'../../output/API_classification/google/{dataset_name}_aggregate/google_{dataset_name}_aggregate.csv',
        'microsoft': f'../../output/API_classification/microsoft/{dataset_name}_aggregate/microsoft_{dataset_name}_aggregate.csv',
        'perspective': f'../../output/API_classification/perspective/{dataset_name}_aggregate/perspective_{dataset_name}_aggregate.csv'
    }
    return pd.read_csv(file_paths[api])

def get_false(dataset_name, api, distance):
    file = load_data(dataset_name, api)
    index = file.index[file.loc[:, 'predicted_label'] - file.loc[:, 'true_label'] == distance].tolist()
    wrong = pd.DataFrame(file.loc[index])
    wrong = wrong.set_index(np.array(list(range(len(wrong)))))
    ids = 'id' if api == 'microsoft' else 'ids'
    return wrong[[ids, 'text', 'score', 'predicted_label', 'true_label']]

# Load SHAP values from CSV files
shap_amazon = pd.read_csv('../../output/API_classification/shap/amazon/hatexplain/amazon_hatexplain.csv', delimiter= ',')
shap_google = pd.read_csv('../../output/API_classification/shap/google/hatexplain/google_hatexplain.csv', delimiter= ',')
shap_gpt = pd.read_csv('../../output/API_classification/shap/gpt/hatexplain/gpt_hatexplain.csv', delimiter= ',')

# Example of setting a threshold for SHAP values
threshold = 0.5


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
# Function to binarize SHAP values based on the threshold
def binarize_shap_values(shap_values, threshold):
    return (shap_values >= threshold).astype(int)

# Function to calculate IOU
def calculate_iou(human_rationale, shap_pred):
    intersection = np.sum(np.logical_and(human_rationale, shap_pred))
    union = np.sum(np.logical_or(human_rationale, shap_pred))
    iou = intersection / union if union != 0 else 0
    return iou

# Function to apply a sliding window to average SHAP values
def apply_sliding_window(shap_values, target_length):
    shap_values = np.array(shap_values)
    window_size = len(shap_values) // target_length
    averaged_shap_values = np.array([
        np.mean(shap_values[i*window_size:(i+1)*window_size])
        for i in range(target_length)
    ])
    return averaged_shap_values

# Function to normalize SHAP values to a range from 0 to 1
def normalize_shap_values(shap_values):
    min_val = np.min(shap_values)
    max_val = np.max(shap_values)
    normalized_shap_values = (shap_values - min_val) / (max_val - min_val) if max_val != min_val else shap_values
    return normalized_shap_values

# General function to calculate metrics
def calculate_metrics(shap_values, human_rationale, threshold):
    shap_bin = binarize_shap_values(shap_values, threshold)
    # binarize human rationale
    human_rationale = binarize_shap_values(human_rationale, threshold)
    
    # Calculate IOU
    iou = calculate_iou(human_rationale, shap_bin)
    

    # Calculate Precision, Recall, and F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(human_rationale, shap_bin, average='binary')
    
    # Calculate AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(human_rationale, shap_values)
    auprc = auc(recall_curve, precision_curve)
    
    return iou, precision, recall, f1, auprc

# Unified function to process SHAP values for any API
def process_shap_values(api_name, shap_csv_path, HX, false_distance=1):
    # Load the SHAP values for the given API
    shap_values_df = pd.read_csv(shap_csv_path, delimiter=',')
    # load false predictions -> 1 = false positive, -1 = false negative
    false_predictions = get_false('hatexplain', api_name, false_distance)

    shap_values_df = shap_values_df[shap_values_df['id'].isin(false_predictions['id'])]
    results = []
    for id, data in HX.iterrows():
        # Use the average of the annotators' rationales
        if len(data['rationales']) == 0:
            continue
        human_rationale = np.mean(np.array(data['rationales']), axis=0)
        
        shap_values = shap_values_df[shap_values_df['id'] == id]['shap_values'].apply(json.loads).values.tolist()
        print(shap_values)
        if len(shap_values) == 0:
            continue
        #make nans 0    
        shap_values = [np.nan_to_num(np.array(shap_values[i])) for i in range(len(shap_values))]
        shap_values = np.array(shap_values[0])
        print(shap_values)
        # Remove first and last SHAP values
        shap_values = shap_values[1:-1]
        
        # Apply sliding window to average SHAP values to match the length of human rationales
        if len(shap_values) != len(human_rationale):
            shap_values = apply_sliding_window(shap_values, len(human_rationale))
        shap_values = shap_values * false_distance
        shap_values = normalize_shap_values(shap_values)
        print(data['text'])
        print(shap_values)
        print(human_rationale)
        iou, precision, recall, f1, auprc = calculate_metrics(shap_values, human_rationale, threshold)
        
        results.append({
            'id': id,
            f'iou_{api_name}': iou,
            f'precision_{api_name}': precision,
            f'recall_{api_name}': recall,
            f'f1_{api_name}': f1,
            f'auprc_{api_name}': auprc,
        })
    
    return pd.DataFrame(results)

# Calculate results for each API
amazon_results = process_shap_values('amazon', '../../output/API_classification/shap/amazon/hatexplain/amazon_hatexplain.csv', HX)
google_results = process_shap_values('google', '../../output/API_classification/shap/google/hatexplain/google_hatexplain.csv', HX)
gpt_results = process_shap_values('gpt', '../../output/API_classification/shap/gpt/hatexplain/gpt_hatexplain.csv', HX)

# Combine results
combined_results = amazon_results.merge(google_results, on='id', how='outer').merge(gpt_results, on='id', how='outer')

# Display results
print(combined_results)

# Optionally, save results to a CSV file
combined_results.to_csv('shap_vs_human_rationales_comparison.csv', index=False)
