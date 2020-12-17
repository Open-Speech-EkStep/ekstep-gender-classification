import unittest
from scripts.create_embeddings import encoder
from os import path
from glob import glob
import numpy as np


class EmbeddingsTests(unittest.TestCase):
    def test_should_create_embeddings_for_clean_test_files(self):
        source_dir = '../resources/test_data/'
        source_dir_pattern = '*/*/*/clean/*.wav'
        embed_file_name = '../resources/test_outputs/sample_embed.npz'

        encoder(source_dir, source_dir_pattern, embed_file_name)
        self.assertTrue(path.exists(embed_file_name))
        clean_files = glob(source_dir + source_dir_pattern)
        self.assertEqual(len(clean_files), 16)

    def test_embed_file_created_has_4_columns(self):
        embed_file_name = '../resources/test_outputs/sample_embed.npz'
        embed_file = np.load(embed_file_name)
        self.assertEqual(embed_file.files, ['embeds', 'file_paths', 'gender', 'duration'])

    def test_should_extract_gender_from_dir_and_save_in_embed_file(self):
        embed_file_name = '../resources/test_outputs/sample_embed.npz'
        embed_file = np.load(embed_file_name)
        gender_list = embed_file['gender']
        self.assertTrue('male' in gender_list and 'female' in gender_list)
        num_male_samples = [i for i in gender_list if i == 'male']
        num_female_samples = [i for i in gender_list if i == 'female']

        self.assertEqual(len(num_male_samples), 8)
        self.assertEqual(len(num_female_samples), 8)

    def test_should_extract_audio_duration_and_save_in_embed_file(self):
        embed_file_name = '../resources/test_outputs/sample_embed.npz'
        embed_file = np.load(embed_file_name)
        durations = embed_file['duration']
        dtypes = [str(type(dur)) for dur in durations]
        self.assertTrue(all([True for dtype in dtypes if 'float64' in dtype]))

    def test_should_extract_file_paths_and_save_in_embed_file(self):
        embed_file_name = '../resources/test_outputs/sample_embed.npz'
        embed_file = np.load(embed_file_name)
        file_paths = embed_file['file_paths']
        self.assertTrue(all([path.exists(file) for file in file_paths]))

    def test_embeddings_have_256_dimensions(self):
        embed_file_name = '../resources/test_outputs/sample_embed.npz'
        embed_file = np.load(embed_file_name)
        embeds = embed_file['embeds']
        self.assertEqual(embeds.shape, (16, 256))


if __name__ == '__main__':
    unittest.main()
