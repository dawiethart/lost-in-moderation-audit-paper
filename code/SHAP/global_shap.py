import numpy as np
import pandas as pd
import random
import torch
import sys
import logging
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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

def embed_words(words):
    """
    Embeds a list of words using the BERT model.

    Args:
        words (list): List of words or sentences to embed.

    Returns:
        torch.Tensor: The average embedding of each input.
    """
    encoding = tokenizer.batch_encode_plus(
        words,                    # List of input texts
        padding=True,             # Pad to the maximum sequence length
        truncation=True,          # Truncate to the maximum sequence length if necessary
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True   # Add special tokens CLS and SEP
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average over the sequence length
    
    return embeddings

def cluster_embeddings(words, embeddings, threshold=0.9):
    """
    Clusters word embeddings based on cosine similarity.

    Args:
        words (list): List of words corresponding to embeddings.
        embeddings (torch.Tensor): Tensor of word embeddings.
        threshold (float): Cosine similarity threshold for clustering.

    Returns:
        list: A list of clusters, where each cluster is a list of words.
    """
    # Convert embeddings to numpy for faster operations
    embeddings_np = embeddings.numpy()
    
    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embeddings_np)
    
    # Track which tokens have been clustered
    clustered = np.zeros(len(words), dtype=bool)
    clusters = []

    for i in tqdm(range(len(words)), desc="Clustering embeddings"):
        if not clustered[i]:
            # Find all words with cosine similarity above threshold
            similar_indices = np.where(cosine_sim_matrix[i] >= threshold)[0]
            clusters.append([words[j] for j in similar_indices])
            clustered[similar_indices] = True

    return clusters

def shap(api, dataset, distance):
    shap_try = pd.read_csv(f'../../output/API_classification/shap/{api}/{dataset}/{api}_{dataset}.csv', delimiter= ',')
    f = get_false(dataset, api, distance)
    #remove columns with nan values in shape_values
    shap_try = shap_try[shap_try['shap_values'].notna()]
    try:
        shap_try['tokenized'] = shap_try['tokenized'].apply(json.loads)
        shap_try['shap_values'] = shap_try['shap_values'].apply(json.loads)
    except:
        pass
    if api == 'microsoft':
        f['ids'] = f['id']
    
    index = shap_try['id'].index[shap_try['id'].isin(f['ids'])] 
    
    shap_try = shap_try.iloc[index]
    
    tokens = sum(shap_try['tokenized'].values.tolist(), []) 
   
    values = np.array(sum(shap_try['shap_values'].values.tolist(), []))
    print('Number of tokens:', len(tokens))
    # Embed tokens using BERT
    embeddings = embed_words(tokens)
    print('Embeddings shape:', embeddings.shape)
    print("Clustering tokens based on cosine similarity...")
    # Cluster tokens based on cosine similarity
    clusters = cluster_embeddings(tokens, embeddings, threshold=0.9)
    print('Number of clusters:', len(clusters))
    print("Calculating mean SHAP values for each cluster...")
    safe = []
    for cluster in clusters:
        indices = [tokens.index(token) for token in cluster]
        mean_value = values[indices].mean()
        safe.append([', '.join(cluster), mean_value])

    safe = pd.DataFrame(safe, columns=['tokens', 'values'])
    safe['tokens'] = safe['tokens'].apply(lambda x: x.strip())

    if shap_try.empty:
        if distance == 1:
            print('No entries for false positive found')
        elif distance == -1:
            print('No entries for false negative found')
    else:
        safe = safe.sort_values(by='values', ascending=(distance == 1))
        filename = f'../../output/API_classification/shap/{api}/{dataset}/global_shap_{api}_{dataset}_{"fp" if distance == 1 else "fn"}.csv'
        safe.to_csv(filename, index=False)

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        logging.info("Please provide arguments as follows: top_shap.py --api <api> --dataset <dataset_name>")
        return
    if args[2] == '--api':
        api = args[3]
    else:
        logging.exception("Please provide dataset name with the argument --api")
        return
    if args[0] == '--dataset':
        dataset = args[1]
    else:
        logging.exception("Please provide sample fraction with the argument --dataset")
        return

    shap(api, dataset, distance=1) 
    shap(api, dataset, distance=-1)

if __name__ == '__main__':
    main()