# python create_embeddings.py --source-dir './data/' --source-dir-pattern '*/*/clean/*.wav' --output-file-name './gender_embeds.npz'

from resemblyzer import preprocess_wav, VoiceEncoder
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
import numpy as np
import sox
import argparse

def audio_paths(directory, pattern):
    print('Using dir {}'.format(directory + pattern))
    return glob.glob(directory + pattern)


def encoder(source_dir, source_dir_pattern, embed_file_name):

    file_paths = audio_paths(source_dir, source_dir_pattern)
    print('Number of files: {}'.format(len(file_paths)))
    # Example file_path: /Users/neerajchhimwal/ekstep-speech-recognition/gender_identification/data/*/male/
    gender = [i.split('/')[-3].split('_')[-1] for i in file_paths]

    print(gender[0], file_paths[0])

    embed_duration = Parallel(n_jobs=24)(delayed(sox.file_info.duration)(file) for file in tqdm(file_paths))
    processed_wavs = Parallel(n_jobs=-1)(delayed(preprocess_wav)(i) for i in tqdm(file_paths))
    vocoder = VoiceEncoder()
    encodings = Parallel(n_jobs=-1)(delayed(vocoder.embed_utterance)(i) for i in tqdm(processed_wavs))
    print('Creating embeddings')
    encodings = np.array(encodings)
    np.savez_compressed(embed_file_name, embeds=encodings, file_paths=file_paths, gender=gender, duration=embed_duration)
    print('Encodings mapped to filepaths have been saved at {}'.format(embed_file_name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-dir-pattern", required=True)
    parser.add_argument("--output-file-name", required=True)
    args = parser.parse_args()

    src_dir = args.source_dir
    src_dir_pattern = args.source_dir_pattern
    embed_file_name = args.output_file_name
    encoder(src_dir, src_dir_pattern, embed_file_name)

if __name__ == "__main__":
	main()