import os
import pandas as pd
import numpy as np
import argparse
import time
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        help="path to model"
    )

    parser.add_argument(
        "--csv-path",
        default=None,
        type=str,
        help="csv file path containig file paths of audios"
    )

    parser.add_argument(
        "--file-mode",
        default=False,
        type=bool,
        help="True to see results for single audio file"
    )

    parser.add_argument(
        "--file-path",
        default=None,
        type=str,
        help="file path of audio"
    )

    parser.add_argument(
        "--save-dir",
        default='./',
        type=str,
        help="location to save prediction file"
    )

    return parser

def load_model(model_path):
    return joblib.load(model_path)

def get_embed(voice_enc, file):
    return np.asarray(voice_enc.embed_utterance(preprocess_wav(file))).reshape(1, -1)

def get_prediction(voice_enc,model, file):
    if os.path.exists(file):
        X = get_embed(voice_enc,file)
        return model.predict(X)[0]
    else:
        raise Exception(f"File path does not exist {file}")

def main(args):
    if os.path.exists(args.model_path):
        model = load_model(args.model_path)
    voice_enc = VoiceEncoder()
    if args.file_mode:
        print(f"Predicted gender : {get_prediction(voice_enc,model, args.file_path)}")
    else:
        df = pd.read_csv(args.csv_path, header=None, names=['file_paths'])
        df['predicted_gender'] = Parallel(n_jobs=-1)(delayed(get_prediction)(voice_enc,model, file_path) for file_path in tqdm(df['file_paths'].values))
        df.to_csv(os.path.join(args.save_dir, 'predictions.csv'), header=False, index=False)
        print(f"Inference Completed")

if __name__ == "__main__":
    s = time.time()
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    print(f"Time taken {time.time() - s} seconds")