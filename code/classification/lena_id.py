import pandas as pd
import numpy as np
from collections import Counter
import json

def og(api, dataset):
    """
    Maps tokenized texts to their original IDs based on the dataset.
    """
    # Adjust dataset name and id column based on specific conditions
    shap_values_df = pd.read_csv(f'../../output/API_classification/shap/{api}/{dataset}/{api}_{dataset}.csv', delimiter=',') #With old ids
    id_column = 'ids' if dataset != 'toxigen' else 'id'
    
    dataset_suffix = '_human_annotated' if dataset == 'toxigen' else ''
    dataset_path = f"../../input/data/{dataset}{dataset_suffix}_aggregate.csv"
    
    
    original_data = pd.read_csv(dataset_path)
    

    text_to_right_id = {text: og_id for text, og_id in original_data[['text', id_column]].values} #ID aus Aggregate File
    #print({value: count for value, count in Counter(text_to_right_id.keys()).items() if count > 1})
 
    shap_values_df['og_id']= shap_values_df['tokenized'].apply(json.loads).apply("".join).map(text_to_right_id) #Map the new ids to old shap file
    shap_values_df = shap_values_df.drop_duplicates(subset=['og_id'])  # drop texts which are duplicates
    
    #print(shap_values_df['og_id'].value_counts())
    

    shap_values_df = shap_values_df[['tokenized', 'shap_values', 'base_values', 'og_id']]
    shap_values_df.to_csv(f'../../output/API_classification/shap/{api}/{dataset}/{api}_{dataset}_ID.csv')


og('amazon', 'hatexplain')
og('gpt', 'hatexplain')
og('microsoft', 'hatexplain')
og('google', 'hatexplain')
og('perspective', 'hatexplain')


og('amazon', 'toxigen')
og('gpt', 'toxigen')
og('microsoft', 'toxigen')
og('google', 'toxigen')
og('perspective', 'toxigen')






