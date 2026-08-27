import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def data_prep(dataset_name):
    out_gpt = pd.read_csv(f'../../output/API_classification/gpt/{dataset_name}/gpt_{dataset_name}.csv')
    out_google = pd.read_csv(f'../../output/API_classification/google/{dataset_name}/google_{dataset_name}.csv')
    out_amazon = pd.read_csv(f'../../output/API_classification/amazon/{dataset_name}/amazon_{dataset_name}.csv')
    out_microsoft = pd.read_csv(f'../../output/API_classification/microsoft/{dataset_name}/microsoft_{dataset_name}.csv')
    out_microsoft = out_microsoft.rename(columns={'id': 'ids'})

    merge = out_gpt.merge(out_google, on='ids', how='inner')
    #merge score instead of predicted label

    merge = merge.rename(columns={'score_x': 'score_gpt', 'score_y': 'score_google', 'true_label_x': 'true_label'})
    merge = merge[['ids', 'score_gpt', 'score_google', 'true_label']]
    merge = merge.merge(out_microsoft, on='ids', how='inner')
    merge = merge.rename(columns={'score': 'score_microsoft', 'true_label_x': 'true_label'})
    merge = merge[['ids', 'score_gpt', 'score_google', 'score_microsoft', 'true_label']]
    merge = merge.merge(out_amazon, on='ids', how='inner')
    merge = merge.rename(columns={'score': 'score_amazon', 'true_label_x': 'true_label'})
    merge = merge[['ids', 'score_gpt', 'score_google', 'score_microsoft', 'score_amazon', 'true_label']]
    merge = merge.dropna()
    
    dim = len(merge['ids']) - 1
    # stack the dataframes
    stack = np.dstack([merge['score_gpt'], merge['score_google'], merge['score_microsoft'], merge['score_amazon'], merge['true_label']])
    #stack by id (concetenate the predictions of the 4 models and the true label by id
    #stack = np.dstack([out_gpt['predicted_label'].iloc[:dim], out_google['predicted_label'].iloc[:dim], out_microsoft['predicted_label'].iloc[:dim], out_amazon['predicted_label'].iloc[:dim], out_gpt['true_label'].iloc[:dim]])
    return pd.DataFrame(stack[0, 0:dim], columns='GPT Google Microsoft Amazon true_label'.split())


def meta_learn(train_dataset_name):
    print('Training Meta Learner')
    stack_df = data_prep(train_dataset_name)
    
    # Train different models
    X = stack_df.iloc[:, 0:4]
    y = stack_df['true_label']

    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        trained_model = model.fit(X, y)
        trained_models[name] = trained_model

    return trained_models


def accuracy(train_dataset_name, test_dataset_name, first=False):
    stack_test_df = data_prep(test_dataset_name)
     # treshold > 0.5 with pandas
    stack_test_df = stack_test_df.assign(GPT = stack_test_df['GPT'].apply(lambda x: 1 if x > 0.5 else 0))
    stack_test_df = stack_test_df.assign(Google = stack_test_df['Google'].apply(lambda x: 1 if x > 0.5 else 0))
    stack_test_df = stack_test_df.assign(Microsoft = stack_test_df['Microsoft'].apply(lambda x: 1 if x > 0.5 else 0))
    stack_test_df = stack_test_df.assign(Amazon = stack_test_df['Amazon'].apply(lambda x: 1 if x > 0.5 else 0))
    X_test = stack_test_df.iloc[:, 0:4]
   


    y_test = stack_test_df['true_label']

    if first:
        for col in X_test.columns:
            acc = accuracy_score(y_test, X_test[col])
            print(f"Accuracy from {col}:", acc)

    # Train models and get predictions
    trained_models = meta_learn(train_dataset_name)
    for name, model in trained_models.items():
        pred_meta_learner = model.predict(X_test)
        # treshold > 0.5    
        pred_meta_learner = np.where(pred_meta_learner > 0.5, 1, 0)
        acc_learner = accuracy_score(y_test, pred_meta_learner)
        print(f"Accuracy from Meta Learner ({name}):", acc_learner)


print('Jigsaw')
accuracy('hatexplain_aggregate', 'jigsaw_aggregate', first=True)
print('Megaspeech')
accuracy('hatexplain_aggregate', 'megaspeech_aggregate', first=True)
accuracy('jigsaw_aggregate', 'megaspeech_aggregate', first=True)
print('Toxigen')
accuracy('hatexplain_aggregate', 'toxigen_aggregate', first=True)
accuracy('jigsaw_aggregate', 'megaspeech_aggregate', first=True)

