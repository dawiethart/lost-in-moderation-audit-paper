import pandas as pd
import numpy as np
import sys
import os
import spacy
from apiGPT import APIgpt
from apiAmazon import APIamazon
from apiGoogle import hate_evaluation_google
from apiMicrosoft import hate_evaluation_microsoft
import logging
import transformers
from transformers import AutoTokenizer
import shap

def pred_prob(text, api):
    """
    INPUT: list with strings
    OUTPUT: numpy array
    """
    if api == 'gpt':
        score = []
        gpt_output_list = APIgpt([st for st in text])
        #print(gpt_output_list)
        #print("---")
        for gpt_output in gpt_output_list:
            max_column = gpt_output['category_scores'].idxmax() # find maximum score (but why? I thought this is only about hate?!)
            idx  = gpt_output.index.get_loc(max_column)
            score.append(gpt_output.iat[idx,1])
            #print(score)
        out =  [[score[i], 1-score[i]] for i in range(len(score))] # score and compliment (1 minus score), i.e. flagged, not to be flagged
        #print(text)
        #print(out)
        #exit()
    elif api == 'google':
        out = hate_evaluation_google(text, np.ones(len(text)), np.full(len(text), 'INDEX'))
        out = np.array([[1 - (out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0), 
                out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0] for i in range(len(out))])
    elif api == 'microsoft':
        out = hate_evaluation_microsoft(text, np.ones(len(text)), np.full(len(text), 'INDEX'))
        out = np.array([[1 - (out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0), 
                out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0] for i in range(len(out))])
    elif api == 'amazon':
        score = []
        amazon_list = [APIamazon(tx) for tx in text]
        for amazon in amazon_list:
            max_column = amazon[0]['Score'].idxmax()
            idx = amazon[0].index.get_loc(max_column)
            score.append(amazon[0].iat[idx, 1])
    return [[score[i], 1-score[i]] for i in range(len(score))]

def pred_prob_wrapper_api(api):
    return lambda text: pred_prob(text, api)

#Mask data

def make_html(dataset, id):
    if dataset == 'toxigen':
        dataset_name = 'toxigen_human_annotated'
    else:
         dataset_name = dataset
    df = pd.read_csv(f'../../input/data/{dataset_name}_aggregate.csv')
    index = df['id'].where(df['id'] == id).dropna().index

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True) # use_fast -> word level simpler tokenizer
    masker = shap.maskers.Text(tokenizer = tokenizer, mask_token = '[MASK]')

    # Calculate Shap Values
    for api in ['gpt', 'amazon', 'google', 'microsoft']:
        shap_val = shap.Explainer(pred_prob_wrapper_api(api), masker= masker)
        explanation = shap_val(list(df['text'][index]))

        a = shap.plots.text(explanation, display = False) 

        with open(f'../../output/html/shap_{dataset_name}_{api}_{id}.html', 'w') as f:
            f.write(a)

def main():
	args = sys.argv[1:]
	if len(args) == 0:
		logging.info("Please provide arguments as follows: html_shap.py --dataset <dataset_name> --text_id <id>")
		return
	if args[0] == '--dataset':
		dataset_name = args[1]
	else:
		logging.exception("Please provide dataset name with the argument --dataset")
		return
	if args[2] == '--text_id':
		id = args[3]
	else:
		logging.exception("Please provide sample fraction with the argument --text_id")
		return

	make_html(dataset_name, id)

if __name__ == '__main__':
	main()