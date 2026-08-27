from googleapiclient import discovery
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import time 
import os

def APIperspective(prompt):

    API_KEY = os.environ.get("PERSPECTIVE_API_KEY", "")

    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )

    analyze_request = {
    'comment': { 'text': prompt },
    'requestedAttributes': {'TOXICITY': {}, 'INSULT': {}, 'THREAT': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    
    return [response["attributeScores"][i]['summaryScore']['value'] for i in ['THREAT', 'INSULT', 'TOXICITY']]

def perspective_worker(text):
    try:
        time.sleep(1)
        values = APIperspective(text)
        score = max(values)
        dec = int(max(values) > 0.5)
        return score, dec
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None, None
    
def hate_evaluation_perspective(text_list, labels, ids, max_workers=1):
    logging.info('Generating Perspective Scores')

    all_scores = [None] * len(text_list)
    all_labels = [None] * len(text_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(perspective_worker, text): idx for idx, text in enumerate(text_list)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            # wait one second
            idx = futures[future]
            score, dec = future.result()
            all_scores[idx] = score
            all_labels[idx] = dec
            
    
    df = pd.DataFrame({
        'id': ids[:len(all_scores)],
        'text': text_list,
        'score': all_scores,
        'predicted_label': all_labels,
        'true_label': labels[:len(all_scores)]
    })    

    return df