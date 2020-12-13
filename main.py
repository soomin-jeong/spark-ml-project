
from load_data import data_loader as dl
from process_data import data_processor as dp
from predict_data import data_predictor as dpre

class ArrivalDelayMachineLearningRunner(object):
    def load_data(self):
        dl.locate_input_data()
        return dl.load_dataset()

    def process_data(self, input_dataset):
        input_dataset = dp.drop_duplicated_data(input_dataset)
        input_dataset = dp.convert_datatypes(input_dataset)
        input_dataset = dp.drop_forbidden_variables(input_dataset)
        input_dataset = dp.remove_null_arr_delay(input_dataset)
        input_dataset = dp.remove_cancelled_flights(input_dataset)
        input_dataset = dp.remove_cancelleation_code(input_dataset)
        input_dataset = dp.convert_datatypes(input_dataset)
        return input_dataset

    def predict(self, processed_data):
        dpre.predict_dummy_function(processed_data)

    def run(self):
        input_dataset = self.load_data()
        processed_data = self.process_data(input_dataset)
        self.predict(processed_data)


runner = ArrivalDelayMachineLearningRunner()
runner.run()
