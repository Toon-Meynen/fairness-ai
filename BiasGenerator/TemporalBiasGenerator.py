from DataGenerator import BiasGenerator
import numpy as np

class TemporalBiasGenerator(BiasGenerator):
    def __init__(self, parameter, primary_distribution, hype_distribution, use_index_as_time=True):
        super().__init__()
        # Hype distribution interval is [0, 1] by default.
        # Primary distribution uses the output of hype distribution as a parameter
        # uses index as notion of time
        self.parameter = parameter
        self.pdis = primary_distribution
        self.hdis = hype_distribution
        self.index_time = use_index_as_time

    def apply(self, data):
        if self.index_time:
            data.df()[self.parameter] = self.pdis(self.hdis(range(len(data.df()))))
            return data
        else:
            if "time" not in data.df():
                data.df()["time"] = range(len(data.df()))
            scaled_time = data.df()["time"].to_list()
            try:
                scaled_time = np.asarray([d.timedelta for d in scaled_time])
            except Exception:
                pass
            scaled_time = (scaled_time - np.min(scaled_time)) / np.ptp(scaled_time)
            data.df()[self.parameter] = self.pdis(self.hdis(scaled_time))
            return data
