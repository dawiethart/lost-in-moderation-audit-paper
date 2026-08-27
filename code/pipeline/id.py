import pandas as pd
import numpy as np
import json

def get_id_key_mapping(shap_values_df, dataset_name):
    """
    Maps tokenized texts to their original IDs based on the dataset.
    """
    # Adjust dataset name and id column based on specific conditions
    id_column = 'ids' if dataset_name != 'toxigen' else 'id'
    dataset_suffix = '_human_annotated' if dataset_name == 'toxigen' else ''
    dataset_path = f"../../input/data/{dataset_name}{dataset_suffix}_aggregate.csv"
    
    
    original_data = pd.read_csv(dataset_path)
    tokenized_texts = shap_values_df['tokenized'].apply(json.loads).apply("".join)

   
    ids = shap_values_df.get('id', shap_values_df.get('og_id')).values
    

    text_to_id = {text: og_id for text, og_id in original_data[['text', id_column]].values}

    id_key = {}
    missing_texts = []

    # Map each tokenized text to its corresponding original ID if it exists
    for i, text in enumerate(tokenized_texts):
        if text in text_to_id:
            id_key[ids[i]] = text_to_id[text]
        else:
            missing_texts.append(text)

    print(f"Dataset '{dataset_name}' - Missing matches: {len(missing_texts)}")
    return id_key, missing_texts

def og(api, dataset):
    shap_values_df = pd.read_csv(f'../../output/API_classification/shap/{api}/{dataset}/{api}_{dataset}.csv', delimiter=',')
    shap_values_df.to_csv(f'../../output/API_classification/shap/{api}/{dataset}/{api}_{dataset}.csv')
    missing, id_key = get_id_key_mapping(shap_values_df, dataset)
    id_column = 'ids' if dataset != 'toxigen' else 'id'

    shap_values_df['og_id']= shap_values_df[id_column].map(id_key)
    
        # drop texts which are in the missing list
    shap_values_df = shap_values_df[~shap_values_df['tokenized'].isin(missing)]
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
#og('google', 'toxigen')
og('perspective', 'toxigen')






