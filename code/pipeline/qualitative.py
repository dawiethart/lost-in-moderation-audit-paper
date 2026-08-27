import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import pandas as pd
import emoji
import json

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
    ids = 'id' if api == 'microsoft' or api == 'perspective' else 'ids'
    return wrong[[ids, 'text', 'score', 'predicted_label', 'true_label']]

def sentence_shap(shap_values, start, path):

    # Make empty file path

    # Split the sentence into tokens
    tokens = shap_values['tokenized'].apply(json.loads).values.tolist()[start:start+5]
    # Token values
    token_values = shap_values['shap_values'].apply(json.loads).values.tolist()[start:start+5]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(max([len(i) for i in tokens])*0.5, 5))  # Adjust figure size based on the number of tokens

    # Set the axis limits (a bit wider for the Legend)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 5)

    #Set the font to read emojis
    emoji_font = FontProperties(fname=r'C:\Windows\Fonts\seguiemj.ttf')

    # Hide the axes
    ax.axis('off')

    # Calculate the width of each token
    total_length = [sum(len(token) for token in i) for i in tokens]
    token_widths = [[len(token) / total_length[i] for token in tokens[i]] for i in range(len(tokens))]
    color_values = [(np.array(token_values[i])+ abs(min(token_values[i]))) / (max(token_values[i])-min(token_values[i])) for i in range(len(tokens))]
    
    #Make the Legend
    y = 0.25
    for i in range(10):
        ax.add_patch(Rectangle((1.1, y), 0.02, 0.45, color=plt.cm.bwr(i*0.1), alpha = 0.7))
        ax.text(1.14, y+0.2, round(i*0.1,1), ha='center', va='center', fontsize=8, color='black')#, fontproperties = prop)
        y += 0.45

    # Plot each token with a background color based on its value
    for i in range(len(tokens)):
        x = 0
        ax.text(0.02 , 0.7+i, shap_values['og_id'].iloc[start+i], ha='center', va='center', fontsize=8, color='black')
        for token, value, width in zip(tokens[i], color_values[i], token_widths[i]):
            ax.add_patch(Rectangle((x, 0.3+i), width, 0.3, color=plt.cm.bwr(value), alpha = 0.7))
            ax.text(x + width / 2, 0.45+i, token, ha='center', va='center', fontsize=8, color='black', fontproperties=emoji_font)  # Adjust font size
            x += width  
    plt.savefig(f'{path}/{int(start/5)}.png', bbox_inches='tight')
    
    #plt.plot()
    plt.close()

def safe_sentences(api, dataset, distance):
    shap_values = pd.read_csv(f'../../output/API_classification/shap/{api}/{dataset}/{api}_{dataset}_ID.csv', delimiter=',')
    false = get_false(dataset, api, distance)
    id = 'id' if api == 'microsoft' or api == 'perspective' else 'ids'
    shap_values = shap_values[shap_values['og_id'].isin(false[id])]
    shap = shap_values['shap_values'].apply(json.loads).tolist()
    nan = [shap_values['og_id'].iloc[i] for i in range(len(shap_values)) if 'NaN' in shap_values['shap_values'].iloc[i] or (np.array(shap[i])==0).all()]
    shap_values = shap_values[~shap_values['og_id'].isin(nan)]
    
    f = 'fn' if distance == -1 else 'fp'

    path = f'../../output/API_classification/shap/{api}/{dataset}/images_{f}'
    if not os.path.exists(path): #Create Path, if it does'nt exist
        os.makedirs(path)
    files = os.listdir(path)
    for file in files: #Remove all old files
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for i in range(0,len(shap_values), 5):
        sentence_shap(shap_values, i, path)

for api in ['amazon', 'google', 'gpt', 'microsoft', 'perspective']:    
    for distance in [1,-1]:
        print(f'Toxigen is running for {api} with distance = {distance}...')
        safe_sentences(api, 'toxigen', distance)
        print(f'Hatexplain is running for {api} with distance = {distance}...')
        safe_sentences(api, 'hatexplain', distance)
    print(f'Done for {api}!')