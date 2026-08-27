import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline

def data_prep(dataset_name):
    out_gpt = pd.read_csv(f'../../output/API_classification/gpt/{dataset_name}/gpt_{dataset_name}.csv')
    out_google = pd.read_csv(f'../../output/API_classification/google/{dataset_name}/google_{dataset_name}.csv')
    out_amazon = pd.read_csv(f'../../output/API_classification/amazon/{dataset_name}/amazon_{dataset_name}.csv')
    out_microsoft = pd.read_csv(f'../../output/API_classification/microsoft/{dataset_name}/microsoft_{dataset_name}.csv')

    out_microsoft = out_microsoft.rename(columns={'id': 'ids'})

    merge = out_gpt.merge(out_google, on='ids', how='inner')
    merge = merge.rename(columns={'predicted_label_x': 'predicted_label_gpt', 'predicted_label_y': 'predicted_label_google', 'true_label_x': 'true_label', 'text_x': 'text'})
    merge = merge[['ids', 'text', 'predicted_label_gpt', 'predicted_label_google', 'true_label']]
    merge = merge.merge(out_microsoft, on='ids', how='inner')
    merge = merge.rename(columns={'predicted_label': 'predicted_label_microsoft', 'true_label_x': 'true_label', 'text_x': 'text'})
    merge = merge[['ids', 'text', 'predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'true_label']]
    merge = merge.merge(out_amazon, on='ids', how='inner')
    merge = merge.rename(columns={'predicted_label': 'predicted_label_amazon', 'true_label_x': 'true_label', 'text_x': 'text'})
    merge = merge[['ids', 'text', 'predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'predicted_label_amazon', 'true_label']]
    merge = merge.dropna()
    return merge


def meta_learn(train_dataset_name):
    stack_df = data_prep(train_dataset_name)
    
    X_text = stack_df['text']
    X_meta = stack_df[['predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'predicted_label_amazon']]
    y = stack_df['true_label']

    # Combine text and meta features
    combined_features = pd.concat([X_text, X_meta], axis=1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'text'),
            ('meta', 'passthrough', ['predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'predicted_label_amazon'])
        ]
    )

    # Use a pipeline to combine preprocessor and a classifier
    text_clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SGDClassifier())
    ])
    text_clf.fit(combined_features, y)

    return text_clf


def accuracy(train_dataset_name, test_dataset_name, first=False):
    stack_test_df = data_prep(test_dataset_name)
    X_test_text = stack_test_df['text']
    X_test_meta = stack_test_df[['predicted_label_gpt', 'predicted_label_google', 'predicted_label_microsoft', 'predicted_label_amazon']]
    y_test = stack_test_df['true_label']

    if first:
        for col in X_test_meta.columns:
            acc = accuracy_score(y_test, X_test_meta[col])
            print(f"Accuracy from {col}:", acc)

    # Train models and get predictions
    meta_model = meta_learn(train_dataset_name)
    X_test_combined = pd.concat([X_test_text, X_test_meta], axis=1)
    pred_meta_learner = meta_model.predict(X_test_combined)
    acc_learner = accuracy_score(y_test, pred_meta_learner)
    print(f"Accuracy from Meta Learner (SGDClassifier with TF-IDF):", acc_learner)


print('Jigsaw')
accuracy('toxigen_aggregate', 'jigsaw_aggregate', first=True)
print('Megaspeech')
accuracy('jigsaw_aggregate', 'megaspeech_aggregate', first=True)
print('Toxigen')
accuracy('jigsaw_aggregate', 'toxigen_aggregate', first=True)
