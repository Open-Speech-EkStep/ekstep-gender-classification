import numpy as np
import pandas as pd

def make_data(npz_path):
    data = np.load(npz_path)
    embeddings, duration, labels, paths = data['embeds'], data['duration'], data['gender'], data['file_paths']
    source_name = [p.split('/')[-5] for p in paths]

    labels = np.where(labels=='Male', 0, labels)
    labels = np.where(labels=='Female', 1, labels)

    embeddings = embeddings.astype('float16')
    labels = labels.astype('int16')

    data_df = pd.DataFrame()

    for i in range(256):
        data_df['feature_'+str(i)] = embeddings[:,i]
        
    data_df.insert(loc=0, column='labels', value=labels)
    data_df.insert(loc=1, column='duration', value=duration)
    data_df.insert(loc=2, column='source_name', value=source_name)
    data_df.insert(loc=3,column='file_paths', value=paths)

    #data_df.to_csv('data.csv', index=False)

    return data_df