
from load_data import data_loader as dl
from process_data import data_processor as dp
from predict_data import data_predictor as dpre


class ArrivalDelayMachineLearningRunner(object):
    def load_data(self):
        dl.locate_input_data()
        return dl.load_dataset()

    def process_data(self, input_dataset):
        return dp.run_all_data_processing(input_dataset)

    def predict(self, processed_data):
        dpre.predict_dummy_function(processed_data)

    def run(self):
        input_dataset = self.load_data()
        processed_data = self.process_data(input_dataset)
        self.predict(processed_data)


runner = ArrivalDelayMachineLearningRunner()
runner.run()
