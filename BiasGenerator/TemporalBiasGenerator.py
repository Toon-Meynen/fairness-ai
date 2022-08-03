from DataGenerator import BiasGenerator
import numpy as np

class TemporalBiasGenerator(BiasGenerator):
    """
    The temporal bias generator updates or adds a column by adding a temporal causation. This is done using two
    distributions. We have the primary distribution and a hype distribution. The hype distribution denotes
    the trend over time, the primary distribution denotes the actual distribution of the new parameter. To create
    this causation between the hype and primary distributions we ensure that the outcome of the hype distribution
    is used as a parameter for the primary distribution. Thus to create a datapoint at time t we use the following
    formula: primary(hype(t)). By default this time t is the index of the item, however this can also be a column
    named "time"

    Attributes
    ----------
    parameter : str
        The parameter we want to (re)-introduce with a temporal bias
    primary_distribution: statistical distribution
        The primary distribution of the new parameter
    hype_distribution: statistical distribution
        The hype distribution denotes the temporal parameter for the primary distribution
    use_index_as_time: bool
        determins if you use a column "time" as time, or the index

    Methods
    -------
    apply(data)
        Applies the bias on a given dataset, returns the biased set.
    """
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
