from sklearn.svm import SVC
import joblib
import pandas as pd
import argparse
from make_data import make_data

def save_model(model, filepath):
    joblib.dump(model, filepath)

def make_X_y(data_df):
    #data = pd.read_csv(data_csv)
    X = data_df.drop(['labels', 'source_name', 'file_paths', 'duration'], axis=1).values
    y = data_df['labels'].values
    return X, y

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--npz-file",required=True)
	args = parser.parse_args()
	
	data_df = make_data(args.npz_file)
	train_X, train_y = make_X_y(data_df)
	clf_SVC = SVC(kernel='rbf', gamma=0.01, C=100, random_state=2020, probability=True)
	clf_SVC.fit(train_X, train_y)
	save_model(clf_SVC, './clf_SVC.sav')
	print("Model has been saved to ./clf_SVC.sav ")