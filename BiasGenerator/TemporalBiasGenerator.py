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
    def __init__(self, parameter, primary_distribution, hype_distribution, use_index_as_time=True, domain=None):
        super().__init__()
        # Hype distribution interval is [0, 1] by default.
        # Primary distribution uses the output of hype distribution as a parameter
        # uses index as notion of time
        self.parameter = parameter
        self.pdis = primary_distribution
        self.hdis = hype_distribution
        self.index_time = use_index_as_time
        self.domain = domain

    def apply(self, data):
        data = data.copy() # ensure original object isn't changed

        if self.index_time:
            data.df(weight=True)[self.parameter] = self.pdis(self.hdis(range(len(data.df(weight=True)))))
        else:
            if "time" not in data.df(weight=True):
                data.df(weight=True)["time"] = range(len(data.df(weight=True)))
            scaled_time = data.df(weight=True)["time"].to_list()
            try:
                scaled_time = np.asarray([d.timedelta for d in scaled_time])
            except Exception:
                pass
            scaled_time = (scaled_time - np.min(scaled_time)) / np.ptp(scaled_time)
            data.df(weight=True)[self.parameter] = self.pdis(self.hdis(scaled_time))

        if self.domain is not None:
            # if smaller, set to min value
            data.df(weight=True).loc[(data.df(weight=True)[self.parameter] < min(self.domain)), self.parameter] = min(self.domain)
            # if larger, set to max value
            data.df(weight=True).loc[(data.df(weight=True)[self.parameter] > max(self.domain)), self.parameter] = max(self.domain)
            #
        return data
