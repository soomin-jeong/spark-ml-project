

class DataLoader(object):

    def __init__(self, spark):
        self.spark = spark

    def load_dataset(self, filepath):
        print("[LOAD] Loading the dataset...")
        print("[LOAD] This should take some time for large datasets.")
        return self.spark.read.csv(filepath, header=True)
