import unittest
from scripts.inference import get_prediction, get_prediction_csv_mode, load_model, get_prediction_from_npz_file
from resemblyzer import VoiceEncoder, preprocess_wav
from os import path


class InferenceTests(unittest.TestCase):
    def test_returns_int_value_for_gender_in_file_mode(self):
        file_path = '../resources/test_data/bucket_sample/female/202009141109586182/clean/1_24_378file-idqUQWlr1113g.wav'
        model_path = '../model/clf_svc.sav'
        model = load_model(model_path)
        voice_enc = VoiceEncoder()
        prediction = get_prediction(voice_enc, model, file_path)
        self.assertEqual(prediction, 1)

    def test_returns_csv_with_predicted_genders_in_csv_mode(self):
        csv_path = '../resources/sample_csv_mode_data.csv'
        model_path = '../model/clf_svc.sav'
        save_dir = '../resources/test_outputs/'
        model = load_model(model_path)
        voice_enc = VoiceEncoder()
        get_prediction_csv_mode(voice_enc, model, csv_path, save_dir)
        self.assertTrue(path.exists(save_dir + 'predictions.csv'))

    def test_returns_csv_with_predicted_genders_from_npz_file(self):
        npz_file_path = '../resources/tarini.npz'
        model_path = '../model/clf_svc.sav'
        save_dir = '../resources/test_outputs/'
        model = load_model(model_path)
        file_vs_prediction_df = get_prediction_from_npz_file(model, npz_file_path, save_dir)
        # self.assertTrue(path.exists(save_dir+'predictions_from_npz.csv'))
        gender_values = set(list(file_vs_prediction_df['gender']))
        self.assertEqual(list(file_vs_prediction_df.columns), ['file_paths', 'gender'])
        self.assertEqual({0, 1}, gender_values)


if __name__ == '__main__':
    unittest.main()
