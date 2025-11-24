from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np 
import ml_collections

#Build two non-overlapping test sets

config = ml_collections.ConfigDict()
config.labels_file = "labels/diagnosis.csv"
config.seed = 0

labels_df = pd.read_csv(config.labels_file, index_col=0)
indices = labels_df.index
labels = labels_df.values.flatten()
labels_dict = dict(zip(indices, labels))

# 89 samples in test_indices
_, test_indices, _, _ = train_test_split(indices, labels, shuffle=True,test_size=0.2, random_state=config.seed, stratify=labels)

indices = list(set(indices).difference(set(test_indices)))
labels = [l for i,l in labels_dict.items() if i in indices]


# 89 samples in test_indices_2
_, test_indices_2, _, _ = train_test_split(indices, labels, shuffle=True,test_size=0.251, random_state=config.seed, stratify=labels)


print(set(test_indices).intersection(set(test_indices_2)))

np.savetxt("test_indices_file.csv",test_indices,delimiter =",", fmt ='% s')
np.savetxt("test_indices_2_file.csv",test_indices_2,delimiter =",", fmt ='% s')




