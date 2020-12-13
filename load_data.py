import os
import sys
import pandas as pd


class DataLoader(object):
    input_filepath = os.path.join(os.getcwd(), 'input_dataset', 'dataset.csv')

    def locate_input_data(self):
        try:
            os.path.exists(self.input_filepath)
        except FileNotFoundError:
            print('[LOAD] Error on loading the dataset...')
            sys.exit(1)

    def load_dataset(self):
        print("[LOAD] Loading the dataset...")

        # `pd.read_csv` checks if the input data is empty
        return pd.read_csv(self.input_filepath)


data_loader = DataLoader()