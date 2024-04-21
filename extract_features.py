import surfboard
import pandas as pd
import numpy as np

from surfboard.sound import Waveform
import warnings
from datetime import datetime

from surfboard import feature_extraction, sound
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features, extract_features_from_waveform

from config import config

mfcc_with_arg = {'mfcc': {'n_mfcc': 26, 'n_fft_seconds': 0.08, 'hop_length_seconds': 0.02}}


# config = {
#     'log_melspec': {
#         'hop_length_seconds': 0.02,
#         'n_fft_seconds': 0.08,
#         'n_mels': 26,
#     },
#     'loudness_slidingwindow': {
#         'frame_length_seconds': 1.0,
#         'hop_length_seconds': 0.25
#     }
# }

def extact_features_from_file(file_name):

    sound_path = file_name + '.wav'
    waveforms = [Waveform(sound_path)]
    waveform = Waveform(sound_path)
    print(f"Extracting features from {file_name}\n")
    timeSt = datetime.now()

    # sample = extract_features(waveforms = waveforms ,
    #                           components_list = [mfcc_with_arg, 'rms'],
    #                           statistics_list=["mean", "std"])

    sample_dict = extract_features_from_waveform(components_list = [mfcc_with_arg, 'rms'],
                              statistics_list=["mean", "std"], waveform = waveform)

    sample = pd.DataFrame([sample_dict])

    print(f"Sample formed. Shape {sample.shape} timing {datetime.now() - timeSt}\n\n")
    sample.to_csv(file_name + '.csv', index = False)
    return sample


def extact_features_from_samples(file_names):

    path_lists = []
    for file_name in file_names:

        path_list = pd.read_csv(file_name)['path']
        print(f"Extracting features for {len(path_list)} records\n")
        timeSt = datetime.now()
        sample = (
            feature_extraction.extract_features_from_paths(
                path_list,
                components_list=[{key: config[key]} for key in config],
                statistics_list=["mean", "std"],
            )
            .replace(-np.inf, np.nan)
            # .fillna(method="bfill")
        )
        print(f"Sample formed. Shape {sample.shape} timing {datetime.now() - timeSt}\n\n")
        sample.to_csv(file_names[file_name])

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    file_path = 'uploads/TMPg0qq0q/record2024-04-20IDg0qq0qno0'
    df = extact_features_from_file(file_path)
    print(df.shape)
    print(df.head())

    # file_names = {'path_train.csv' : 'X_train_features.csv',
    #               'path_test.csv': 'X_test_features.csv',
    #               'path_final_test.csv': 'X_final_test_features.csv'}
    #
    # extact_features_from_samples(file_names)