import pandas as pd
import numpy as np

#load data 
HX = pd.read_json("../../input/data/hatexplain_human_annotated.json")
HX = HX.transpose()

#Renname columns end index
HX = HX.rename(columns={'post_id':'ids', 'post_tokens': 'text'})
HX['id'] = ['HX' + str(i) for i in range(1, len(HX) + 1)]
HX.set_index(['id'], inplace=True)
HX['text'] = [" ".join(HX.iloc[i]["text"]) for i in range(len(HX))]

#Sort by groups 
identity_groups = ['Asian', 'Islam', 'Jewish', 'African', 'Homosexual', 'Women', 'Caucasian', 'None']

targets_HX = [[HX.iloc[i]["annotators"][j]['target'] for j in range(3)] for i in range(len(HX))]
targets_HX = [targets_HX[i][0]+targets_HX[i][1]+targets_HX[i][2] for i in range(len(HX))]

label = []
target = []
for i in range(len(HX)):
    t = [targets_HX[i].count(identity_groups[j]) for j in range(8)]
    if t.count(max(t)) == 1:
        target.append(identity_groups[t.index(max(t))])
    else:
        target.append('Unequal')
    labels_HX = [HX.iloc[i]["annotators"][j]['label'] for j in range(3)]
    if labels_HX.count("normal") == 3:
        label.append(0)
    elif labels_HX.count("normal") == 0:
        label.append(1)
    else:
        label.append(2)

HX['true_label'] = label
HX['group'] = target

#Get where entries with unambigous group-identity and label(two or more agreeing)
HX = HX[(HX['true_label'] < 2) & (HX['groups'] != 'Unequal')]

#Safe aggregate Data

HX_aggregate = HX.sort_values('true_label')[['text', 'true_label']]
HX_aggregate.to_csv("../../input/data/hatexplain_aggregate.csv", index=False)


group_mapping = {
    'Asian': 'asian',
    'Islam': 'muslim',
    'Jewish': 'jewish',
    'African': 'black',
    'Homosexual': 'lgbtq',
    'Women': 'woman',
    'Caucasian': 'white',
}

HX['group'].replace(group_mapping, inplace = True)

#Safe group data
HX_group = HX[(HX['groups'] != 'None')].sort_values('group')[['text', 'group', 'true_label']]
HX_group.to_csv("../../input/data/hatexplain_group.csv", index=False)