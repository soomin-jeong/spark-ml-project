import os
import pandas as pd


class DataLoader(object):
    input_filepath = os.path.join(os.getcwd(), 'input_dataset', 'dataset.csv')

    def locate_input_data(self):
        try:
            os.path.exists(self.input_filepath)
        except FileNotFoundError:
            print('[LOAD] Error on loading the dataset...')

    def load_dataset(self):
        print ("[LOAD] Loading the dataset...")
        # check if the inpu t data is empty
        return pd.read_csv(self.input_filepath)


data_loader = DataLoader()