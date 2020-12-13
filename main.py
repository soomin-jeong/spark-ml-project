
from tool.load_data import data_loader as dl
from tool.process_data import data_processor as dp


class ArrivalDelayMachineLearningRunner(object):
    def load_data(self):
        dl.locate_input_data()
        return dl.load_dataset()

    def process_data(self, input_dataset):
        input_dataset = dp.remove_null_arr_delay(input_dataset)
        input_dataset = dp.remove_cancelled_flights(input_dataset)
        input_dataset = dp.remove_cancelleation_code(input_dataset)
        return input_dataset

    def predict(self, process_data):
        pass

    def run(self):
        input_dataset = self.load_data()
        processed_data = self.process_data(input_dataset)
        self.predict(processed_data)


runner = ArrivalDelayMachineLearningRunner()
runner.run()
