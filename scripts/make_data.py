import numpy as np
import pandas as import pd

def make_data(embeddings):
    data = np.load(embeddings)
    embeddings, duration, labels, paths = data['embeds'], data['duration'], data['gender'], data['file_paths']

    labels = np.where(labels=='male', 0, labels)
    labels = np.where(labels=='female', 1, labels)

    embeddings = embeddings.astype('float16')
    labels = labels.astype('int16')

    data_df = pd.DataFrame()

    for i in range(256):
        data_df['feature_'+str(i)] = embeddings[:,i]
        
    data_df.insert(loc=0, column='labels', value=labels)
    data_df.insert(loc=1, column='duration', value=duration)
    data_df.insert(loc=2, column='source_name', value=source_name)
    data_df.insert(loc=3,column='file_paths', value=paths)

    data_df.to_csv('data.csv', index=False)