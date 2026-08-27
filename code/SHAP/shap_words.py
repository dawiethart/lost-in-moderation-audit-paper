import pandas as pd
import numpy as np
import sys
import logging
import os
import spacy
from apiGPT import APIgpt
from apiAmazon import hate_evaluation_amazon
from apiGoogle import hate_evaluation_google
from apiMicrosoft import hate_evaluation_microsoft
from apiPerspective import hate_evaluation_perspective
import logging
import transformers
from transformers import AutoTokenizer
import shap
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from spacy.lang.en.stop_words import STOP_WORDS
import json

# To hide that annoying warning at the beginning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#nlp = spacy.load("en_core_web_sm")

# Initialize logging and set up log file
if not os.path.exists('../../output/log'):
	os.makedirs('../../output/log')

class ExcludeHttpsFilter(logging.Filter):
	def filter(self, record):
		return not record.getMessage().startswith('HTTPS')

datestr = pd.Timestamp.now().strftime('%Y-%m-%d')
timestr = pd.Timestamp.now().strftime('%H-%M')
logname = f'../../output/log/SHAP_{datestr}_{timestr}.log'
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
					datefmt='%H:%M:%S',
					handlers=[
						logging.FileHandler(logname, mode='a'),
						#logging.StreamHandler(sys.stdout) #uncomment this if you want console logs
					])

logging.info("Running SHAP analysis ")
logger = logging.getLogger()
logger.addFilter(ExcludeHttpsFilter())

### data loading
def load_data(dataset_name, api):
	file_paths = {
		'gpt': f'../../output/API_classification/gpt/{dataset_name}_aggregate/gpt_{dataset_name}_aggregate.csv',
		'amazon': f'../../output/API_classification/amazon/{dataset_name}_aggregate/amazon_{dataset_name}_aggregate.csv',
		'google': f'../../output/API_classification/google/{dataset_name}_aggregate/google_{dataset_name}_aggregate.csv',
		'microsoft': f'../../output/API_classification/microsoft/{dataset_name}_aggregate/microsoft_{dataset_name}_aggregate.csv',
		'perspective': f'../../output/API_classification/perspective/{dataset_name}_aggregate/perspective_{dataset_name}_aggregate.csv'
		}
	
	return pd.read_csv(file_paths[api])

#def remove_stopwords(text):
#	 doc = nlp(text)
#	 return " ".join([token.text for token in doc if not token.is_stop])

def get_mislabeled_data(dataset_name, api, label_difference, should_remove_stopwords=False):
	file = load_data(dataset_name, api)
	index = file.index[file['predicted_label'] - file['true_label'] == label_difference].tolist()  # find indexes for mismatches
	mislabeled_data = file.loc[index].reset_index(drop=True)  # reset index for the new DataFrame
	ids_column = 'id' if (api == 'microsoft' or api == 'perspective') else 'ids'
	#if api == 'perspective':
	#	# has this string form now: [0.007637825, 0.00979289, 0.024613906], want to have the maximum
	#	temp = mislabeled_data['score'].str.extract(r'\[(.*)\]').astype(str).apply(lambda x: x.str.split(',').apply(lambda y: max([float(i) for i in y])))
	#	mislabeled_data['score'] = temp
	#	return mislabeled_data[[ids_column, 'text', 'score', 'predicted_label', 'true_label']]
		
	# I realized that this makes not sense here, because the API will receive the cleaned version. We would have to create a custom shapley class to avoid stopwords instead.
	#if should_remove_stopwords:
	#	 mislabeled_data['text'] = mislabeled_data['text'].apply(remove_stopwords)

	return mislabeled_data[[ids_column, 'text', 'score', 'predicted_label', 'true_label']]

#def get_tokenized_words(sentences):
#	 nlp = spacy.load("en_core_web_sm")
#	 tokenized =[[nlp(sen)] for sen in sentences]
#	 return tokenized

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
                 out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0] 
                for i in range(len(out))])
		elif api == 'microsoft':
			out = hate_evaluation_microsoft(text, np.ones(len(text)), np.full(len(text), 'INDEX'))
			out = np.array([[1 - (out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0), 
                 out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0] 
                for i in range(len(out))])
#out = np.array([[1-out.loc[i,'score'], out.loc[i,'score']] for i in range(len(out))])
		elif api == 'amazon':
			out = hate_evaluation_amazon(text, np.ones(len(text)), np.full(len(text), 'INDEX'))
			out = np.array([[1 - (out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0), 
                 out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0] 
                for i in range(len(out))])
		elif api == 'perspective':
			out = hate_evaluation_perspective(text, np.ones(len(text)), np.full(len(text), 'INDEX'))
			print(out)
			out = np.array([[1 - (out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0), 
				 out.loc[i, 'score'] if out.loc[i, 'score'] is not None else 0] 
				for i in range(len(out))])
		return	out # give probability not to be flagged, and probability to be flagged

def pred_prob_wrapper_api(api):
	return lambda text: pred_prob(text, api)

def shap_explain(df, api):

	# Mask data
	#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True) # use_fast -> word level simpler tokenizer
	masker = shap.maskers.Text(tokenizer = tokenizer, mask_token = '[MASK]') # BERT models use [MASK], RoBERTa and BART use <mask>, GPT does not have a mask token. Since we have no idea what the APIs do internally, I would ust use [MASK].

	# Calculate Shap Values
	shap_val = shap.Explainer(pred_prob_wrapper_api(api), masker = masker)
	#print(list(df['text']))
	#print("START")

	explanation = shap_val(list(df['text']), batch_size=20, silent=False)
	# this is processed sequentially! So shap_val does all variations of one sentence before doing the next
	# we handle parallel processing in shap_aggregate() now. So actually list() is not needed here.

	#print("DONE")
	# Transform to output
	values = [explanation.values[i][:, 1].tolist() for i in range(len(explanation))]
	tokens = [explanation.data[i].tolist() for i in range(len(explanation))]
	base_values = [explanation.base_values[i].tolist()for i in range(len(explanation))]

	return pd.DataFrame({'tokenized': tokens, 'shap_values': values, 'base_values': base_values})


def shap_aggregate(df, dataset_name, frac=1, firstN=None, lastN = None, api = 'microsoft'):
	# see if a file already exists
	output_file = f'../../output/API_classification/shap/{api}/{dataset_name}/{api}_{dataset_name}.csv'

	if os.path.exists(output_file):
		df_existing = pd.read_csv(output_file)
		if len(df_existing) == len(df.sample(frac=frac)):
			logging.info("All IDs were already processed and have a result. Skipping.")
			return

	os.makedirs(os.path.dirname(output_file), exist_ok=True) # since we're only doing one api at a time, we only need this

	logging.info("Processing shap values of dataset")

	if firstN and lastN: # I added this for debugging. It can be removed later.
		df_sampled = df[max(0, len(df)-lastN):min(len(df)-1, firstN)]
	elif firstN:
		df_sampled = df[0:min(len(df)-1, firstN)]
	elif lastN:
		df_sampled = df[max(0, len(df)-lastN):len(df)-1]
	else:
		df_sampled = df.sample(frac=frac, random_state=42)
 
	if df_sampled.columns.str.contains('ids').any():
		df_sampled['id'] = df_sampled.ids

	if df_sampled.columns.str.contains('id').any():
		df_sampled['ids'] = df_sampled.id

	df_sampled.set_index('ids', inplace=True)
	#print("RUNNING WITH:")
	#print(df_sampled)

	'''
	python shap_words.py --dataset toxigen --sample_frac 0.005 --api gpt

	Took roughly 3-4 minutes for 30 entries with the configuration below.
	'''

	# 4 threads -> 3:39
	# 8 threads -> 2:08
	# 16 threads -> Rate limit exceeded

	DF_ENTRIES_PARALLEL = 1 # decrease this if the API returns rate warning
	SLEEP_SECONDS_BETWEEN_ENTRY_RUN = 1 # increase this if the API returns rate warnings

	futures = []
	results = []

	with ThreadPoolExecutor(max_workers=DF_ENTRIES_PARALLEL) as executor, tqdm(total=len(df_sampled), desc=f"--> Processing < {dataset_name} > with < {api} > and parallelism of < {DF_ENTRIES_PARALLEL} >") as progress_bar:
		for i in range(len(df_sampled)):
			while len(futures) >= DF_ENTRIES_PARALLEL: # run this while the max number of futures are running
				for future in as_completed(futures):
					result = future.result()
					results.append(result)
					futures.remove(future)
					progress_bar.update(1) # since we're processing in parallel, we have to inform tqdm to update the progress bar
					time.sleep(SLEEP_SECONDS_BETWEEN_ENTRY_RUN)
					break
			#print("Doing:")
			#print(df_sampled.iloc[[i]])
			future = executor.submit(shap_explain, df_sampled.iloc[[i]], api) # If one future completes of there are non, add the next one (double bracket to keep dataframe format)
			futures.append(future)
			# save some results to disk to avoid losing everything in case of a crash --> json.dumps() is used to store the list as a string
			print(results)
			if results:
					temp_results = pd.concat(results).reset_index(drop=True)
					temp_results['id'] = df_sampled['id'].iloc[i]
					temp_results['tokenized'] = temp_results['tokenized'].apply(json.dumps)
					temp_results['shap_values'] = temp_results['shap_values'].apply(json.dumps)
					temp_results['base_values'] = temp_results['base_values'].apply(json.dumps)
					if os.path.exists(output_file):
						df_existing = pd.read_csv(output_file, index_col=0)
						temp_results = pd.concat([df_existing, temp_results], ignore_index=False)
					temp_results.to_csv(output_file, index=True)
					logging.info(f"Saved {api} results for aggregate dataset to {output_file}")
					results = []

		# if the for loop completes and there's still a future running, this code ensures that we wait for it to complete
		#for future in as_completed(futures):
		#	result = future.result()
		#	results.append(result)
		#	progress_bar.update(1) # since we're processing in parallel, we have to inform tqdm to update the progress bar
		#	time.sleep(SLEEP_SECONDS_BETWEEN_ENTRY_RUN)

		#for i in range(len(results)):
				#results['tokenized'] = list(results['tokenized'])
				#results['shap_values'] = list(results['shap_values'])

	if results:
		all_results = pd.concat(results).reset_index(drop=True)
		all_results.index = df_sampled[len(df_sampled)-len(results)-1:len(df_sampled)-1].index
		all_results['tokenized'] = all_results['tokenized'].apply(json.dumps)
		all_results['shap_values'] = all_results['shap_values'].apply(json.dumps)
		all_results['base_values'] = all_results['base_values'].apply(json.dumps)

		if os.path.exists(output_file):
			df_existing = pd.read_csv(output_file, index_col=0)
			all_results = pd.concat([df_existing, all_results], ignore_index=False)

		all_results.to_csv(output_file, index=True)
		logging.info(f"Saved {api} results for aggregate dataset to {output_file}")
	else:
		logging.warning("No results to save. The results list is empty.")

def run_shap_analysis(dataset_name, api, sample_frac):
	valid_datasets = ['toxigen', 'jigsaw', 'sbic', 'hatexplain']

	if dataset_name not in valid_datasets:
		logging.exception("Please provide a valid dataset name. Choose from: toxigen, jigsaw, sbic, hatexplain")
		return

	logging.info(f"Running SHAP analysis for {dataset_name} dataset")
	logging.info(f"Loading data for {dataset_name} dataset")
	fn = get_mislabeled_data(dataset_name, api, label_difference = -1)
	fp = get_mislabeled_data(dataset_name, api, label_difference = 1)
	# TODO: So far this is not beeing accounted for. What was the plan here? If you want to evaluate fn and fp seprately, then we should write to different files.
	logging.info(f"Data loaded for {dataset_name} dataset")

	# not needed because shap atomatically uses the tokenizer provided to shap.maskers.Text
	#fn['tokenized'] = get_tokenized_words(fn['text'])
	#fp['tokenized'] = get_tokenized_words(fp['text'])
	#logging.info(f"Tokenized words extracted for {dataset_name} dataset")

	shap_aggregate(fn, dataset_name, frac=sample_frac, firstN=None, lastN = len(fn)-477, api=api)
	shap_aggregate(fp, dataset_name, frac=sample_frac, firstN=None, lastN = None, api=api) # caution: this one is small (at least for toxigen) so 0.001 causes an empty list
	logging.info(f"SHAP analysis completed for {dataset_name} dataset")

def main():
	args = sys.argv[1:]
	if len(args) == 0:
		logging.info("Please provide arguments as follows: shap.py --dataset <dataset_name> --sample_frac <sample_frac> --api <api_name>")
		return
	if args[0] == '--dataset':
		dataset_name = args[1]
	else:
		logging.exception("Please provide dataset name with the argument --dataset")
		return
	if args[2] == '--sample_frac':
		sample_frac = float(args[3])
	else:
		logging.exception("Please provide sample fraction with the argument --sample_frac")
		return
	if len(args) > 4 and args[4] == '--api':
		api = args[5]
	else:
		api = 'all'

	run_shap_analysis(dataset_name, api, sample_frac)

if __name__ == '__main__':
	main()
