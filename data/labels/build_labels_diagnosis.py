from os.path import join
import os
from tqdm import tqdm
import pandas
import numpy as np

FOLDER_DIR = 'CLUSSTER-Benin/images'

files = [f for f in os.listdir(FOLDER_DIR)] 

# Get all dataset names
datasets = sorted({f.split('_')[0] for f in files})
print("Number of datasets", len(datasets))


df = pandas.read_csv("CLUSSTER-Benin/clinical_data/CLUSSTERBenin-ClinicalDataForResea_DATA_2023-05-24_1630.csv")
patient_id = 'record_id'
test_results = 'tb_rif_genexpert'
with open('diagnosis.csv', 'w') as file:
    file.write('record_id,tb_rif_genexpert\n')
    for index, row in df.iterrows():
        label = row[test_results]
        if np.isnan(label) == False and label != 3.0:
            file.write(str(row[patient_id])[3:] +',' + str(int(label-1))+'\n')
        elif label == 3.0:
            file.write(str(row[patient_id])[3:] +',' + str(int(label-2))+'\n')

df = pandas.read_csv("diagnosis.csv")
patient_labels = set([str(i) for i in df['record_id']] )

patient_images = set(datasets)

print('# patients with a label: ', len(patient_labels))
print('# patients with images: ', len(patient_images))

print('Patients with images but no label: ', patient_images.difference(patient_labels))
print('Patients with label but no image: ', patient_labels.difference(patient_images))
print('# patients in dataset: ', len(patient_labels.intersection(patient_images)))

df = pandas.read_csv('diagnosis.csv', index_col=0)
df = df.drop([int(i) for i in patient_labels.difference(patient_images)])
df.to_csv('diagnosis.csv')
