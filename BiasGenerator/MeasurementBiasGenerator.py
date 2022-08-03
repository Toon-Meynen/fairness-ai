from DataGenerator import BiasGenerator
import scipy.stats
import numpy as np


class MeasurementBiasGenerator(BiasGenerator):
    """
    introduces a measurement bias on a given parameter, value pair.

    Attributes
    ----------
    parameter : str
        bias is introduced w.r.t. this parameter, value pair
    pvalue : int
        bias is introduced w.r.t. this parameter, value pair
    measurement: str
        the parameter we want to add a measurement error to
    bias_strength: float
        propability for a condidate datapoint to be altered
    weight: dict
        weight["measurement_error"]
            a statistical distribution that is added to the datapoint we alter, giving a new datapoint.
        weight["invalid_ratio"]
            a probability that a selected datapoint is set to a completely random value within its original domain

    Methods
    -------
    apply(data)
        Applies the bias on a given dataset, returns the biased set.
    """
    def __init__(self, parameter, parameter_value, measurement, weight=None, bias_strength=0.3):
        super().__init__()
        # Here bias strength indicates the probability in the rows containing the correct biased parameter
        # and value. That the measurement will be increased or decreased by one. Thus the lowest and highest
        # class have only half the effect of this strength as these can only increase or decrease
        self.parameter = parameter
        self.pvalue = parameter_value

        self.measurement = measurement
        self.bias_strength = bias_strength
        if weight is None:
            weight = dict()
        self.weight = weight

        # if weight["measurement_error"] is a list, convert to scipy stats object
        self._discrete = False
        if "measurement_error" in self.weight:
            if type(self.weight["measurement_error"]) is list:
                self._discrete = True
                self.weight["measurement_error"] = scipy.stats.rv_discrete(values=(range(len(self.weight["measurement_error"])), self.weight["measurement_error"]))

    def apply(self, data):
        # ensure weight dict is complete, fill with default values where empty
        if "invalid_ratio" not in self.weight:
            self.weight["invalid_ratio"] = 0.01
        if "measurement_error" not in self.weight:
            self.weight["measurement_error"] = scipy.stats.norm(0, 1)

        values = data.df()[self.measurement].unique()

        # select the measurements that encountered extra error
        tmp = data.df().loc[(data.df()[self.parameter] == self.pvalue)].sample(frac=self.bias_strength)
        # add measurement error
        if self._discrete:
            data.df().loc[tmp.index.values, self.measurement] = self.weight["measurement_error"].rvs(size=len(tmp))
        else:
            data.df().loc[tmp.index.values, self.measurement] += np.round(self.weight["measurement_error"].rvs(size=len(tmp))).astype(int)
        # if smaller, set to min value
        data.df().loc[(data.df()[self.measurement] < min(values)), self.measurement] = min(values)
        # if larger, set to max value
        data.df().loc[(data.df()[self.measurement] > max(values)), self.measurement] = max(values)
        # add invalid ratio to selected measurements
        tmp = tmp.sample(frac=self.weight["invalid_ratio"])
        data.df().loc[tmp.index.values, self.measurement] = np.random.randint(min(values), max(values) + 1, len(tmp))
        return data
