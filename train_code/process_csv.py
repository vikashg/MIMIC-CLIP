import os
import numpy as np
from pandas import read_csv
from tqdm import tqdm
import json

def creating_negative_classes(dieases_classes):
    """
    Create negative classes for each disease class
    """
    for disease in dieases_classes:
        print(disease)
        new_class = 'no_' + disease
        print(new_class)
        dieases_classes = np.append(dieases_classes, new_class)
    return dieases_classes

def convert_csv_to_json(filename):
    fid = open(filename, 'r')
    df = read_csv(fid)
    fid.close()
    base_dir = '/workspace/data/image'
    data_list = []
    for index, row in tqdm(df.iterrows()):
        image_fn = os.path.join(base_dir, row['DicomPath'])
        label = row[4:].values.flatten().tolist()
        _data = {'image': image_fn, 'label': label}
        data_list.append(_data)
    num_data = len(data_list)
    num_train = int(num_data * 0.8)
    num_val = int(num_data * 0.1)
    num_test = num_data - num_train - num_val
    train_data = data_list[:num_train]
    val_data = data_list[num_train:num_train+num_val]
    test_data = data_list[num_train+num_val:]
    data_dict = {'train': train_data, 'val': val_data, 'test': test_data}
    json_fn = filename.replace('.csv', '.json')
    fid = open(json_fn, 'w')
    json.dump(data_dict, fid, indent = 2)
    fid.close()


def preprocess_csv(filename):
    fid = open(filename, 'r')
    df = read_csv(fid)
    fid.close()
    disease_classes = df.columns[4:]
    print(disease_classes)
    all_classes = creating_negative_classes(disease_classes)
    print(all_classes)

    #get columns with value 1
    # if column has value -1, then create a new negative column and set value to 1
    for index, row in df.iterrows():
        for disease in disease_classes:
            if row[disease] == -1:
                new_disease = 'no_' + disease
                df.loc[index, new_disease] = 1
                df.loc[index, disease] = np.nan
            if row[disease] == 0:
                new_disease = 'no_' + disease
                df.loc[index, new_disease] = 1
                df.loc[index, disease] = 1

    df.replace(np.nan, 0, inplace=True)
    df.to_csv('/workspace/data/image/CXLSeg-segmented-processed.csv', index=False)



def main():
    filename = '/workspace/data/image/CXLSeg-segmented.csv'
    #preprocess_csv(filename)
    processed_filename = '/workspace/data/image/CXLSeg-segmented-processed.csv'
    convert_csv_to_json(processed_filename)



if __name__ == "__main__":
    main()
