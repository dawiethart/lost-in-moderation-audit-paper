import random
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Set a random seed
random_seed = 42
random.seed(random_seed)
 
# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embedding(words):
    # The function returns a dictionary containing the token IDs and attention masks
    encoding = tokenizer.batch_encode_plus(
        words,                    # List of input texts
        padding=True,              # Pad to the maximum sequence length
        truncation=True,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP
    )
    
    input_ids = encoding['input_ids']  # Token IDs
    attention_mask = encoding['attention_mask']  # Attention mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        sentence__embedding = outputs.last_hidden_state.mean(dim=1)  # This contains the embeddings
    return sentence__embedding

#Load data
out_file_words = f'../../output/API_classification/shap/gpt/toxigen/gpt_toxigen_top_shap.csv'
shap_words = pd.read_csv(out_file_words)

clustered = []
words = list(shap_words['1'])

while words != []:
    example_list = [words[i] for i in range(len(words)) if cosine_similarity(embedding(words[0]), embedding(words[i]))[0][0] >= 0.99]
    words = [words[i] for i in range(len(words)) if words[i] not in example_list]
    clustered.append(example_list)
    #print(clustered)

labels = [clustered[i][0] for i in range(len(clustered))]
sizes = [len(clustered[i]) for i in range(len(clustered))]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)
fig.savefig('../../output/API_classification/shap/gpt/toxigen/gpt_toxigen_graph.png')