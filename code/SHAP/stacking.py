import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def data_prep(dataset_name):
    out_gpt = pd.read_csv(f'../../output/API_classification/gpt/{dataset_name}/gpt_{dataset_name}.csv')
    out_google = pd.read_csv(f'../../output/API_classification/google/{dataset_name}/google_{dataset_name}.csv')
    out_amazon = pd.read_csv(f'../../output/API_classification/amazon/{dataset_name}/amazon_{dataset_name}.csv')
    out_microsoft = pd.read_csv(f'../../output/API_classification/microsoft/{dataset_name}/microsoft_{dataset_name}.csv')

    out_microsoft = out_microsoft.rename(columns={'id': 'ids'})

    # merge first two dataframes
    merge = out_gpt.merge(out_google, on='ids', how='inner')
    # rename predicted label columns
    merge = merge.rename(columns={'predicted_label_x': 'predicted_label_gpt', 'predicted_label_y': 'predicted_label_google', 'true_label_x': 'true_label'})
    # only use the columns we need
    merge = merge[['ids', 'predicted_label_gpt', 'predicted_label_google', 'true_label']]
    # merge with the third dataframe
    merge = merge.merge(out_microsoft, on='ids', how='inner')
    # rename predicted label columns
    merge = merge.rename(columns={'predicted_label': 'predicted_label_microsoft', 'true_label_x': 'true_label'})
    # only use the columns we need
    merge = merge[['ids', 'predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'true_label']]
    # merge with the fourth dataframe
    merge = merge.merge(out_amazon, on='ids', how='inner')
    # rename predicted label columns
    merge = merge.rename(columns={'predicted_label': 'predicted_label_amazon', 'true_label_x': 'true_label'})
    # only use the columns we need
    merge = merge[['ids', 'predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'predicted_label_amazon', 'true_label']]
    

    dim = len(out_gpt['ids'])-1
    # stack the dataframes
    stack = np.dstack([merge['predicted_label_gpt'], merge['predicted_label_google'], merge['predicted_label_microsoft'], merge['predicted_label_amazon'], merge['true_label']])
    
    #stack by id (concetenate the predictions of the 4 models and the true label by id

    #stack = np.dstack([out_gpt['predicted_label'].iloc[:dim], out_google['predicted_label'].iloc[:dim], out_microsoft['predicted_label'].iloc[:dim], out_amazon['predicted_label'].iloc[:dim], out_gpt['true_label'].iloc[:dim]])
    return pd.DataFrame(stack[0,0:dim], columns='GPT Google Microsoft Amazon true_label'.split())


def meta_learn(train_dataset_name):
    stack_df = data_prep(train_dataset_name)

    #train meta-learner on training data
    meta_learner = LogisticRegression()
    return meta_learner.fit(stack_df.iloc[:,0:3], stack_df['true_label'])

def accuracy(train_dataset_name, test_dataset_name, first = False):
    stack_test_df = data_prep(test_dataset_name)
    dim = len(stack_test_df['GPT'])

    if first == True:
        acc_gpt=np.count_nonzero(stack_test_df['GPT']-stack_test_df['true_label']==0)/dim
        acc_google=np.count_nonzero(stack_test_df['Google']-stack_test_df['true_label']==0)/dim
        acc_amazon=np.count_nonzero(stack_test_df['Amazon']-stack_test_df['true_label']==0)/dim
        acc_microsoft=np.count_nonzero(stack_test_df['Microsoft']-stack_test_df['true_label']==0)/dim
        print("Accuracy from GPT:" ,acc_gpt)
        print("Accuracy from Google:" ,acc_google)
        print("Accuracy from Amazon:" ,acc_amazon)
        print("Accuracy from Microsoft:" ,acc_microsoft)
    
    #make meta learner prediction
    pred_meta_learner = meta_learn(train_dataset_name).predict(stack_test_df.iloc[:,0:3])

    acc_learner = accuracy_score(stack_test_df['true_label'], pred_meta_learner)

    print("Accuracy from Meta Learner:" ,acc_learner)
    #return acc_gpt, acc_google, acc_amazon, acc_microsoft, acc_learner


print('Jigsaw')
accuracy('hatexplain_aggregate', 'jigsaw_aggregate', first=True)
print('Megaspeech')
accuracy('hatexplain_aggregate', 'megaspeech_aggregate', first=True)
print('Toxigen')
accuracy('hatexplain_aggregate', 'toxigen_aggregate', first=True)