from sklearn.svm import SVC
import joblib
import pandas as pd

def save_model(model, filepath):
    joblib.dump(model, filepath)

def make_X_y(data_csv):
    data = pd.read_csv(data_csv)
    X = data.drop(['labels', 'source_name', 'file_paths', 'duration'], axis=1).values
    y = data['labels'].values
    return X, y

if __name__ == "__main__":
    train_X, train_y = make_X_y(data_csv)
    clf_SVC = SVC(kernel='rbf', gamma=0.01, C=100, random_state=2020, probability=True)
    clf_SVC.fit(train_X, train_y)
    save_model(clf_SVC, 'model/clf_SVC.sav')