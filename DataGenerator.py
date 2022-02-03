from Data import Data
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import scipy.stats
from pgmpy.factors.continuous import LinearGaussianCPD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class DataGenerator:
    def __init__(self, data_generator, protected_attributes, labels):
        self.data_generator = data_generator
        self._prot_attr = protected_attributes
        self._labels = labels

    # generates n items
    def simulate(self, n=10):
        return Data(df=self.data_generator.simulate(n), protected_attributes=self._prot_attr, labels=self._labels)


class BiasGenerator:
    def __init__(self):
        pass

    def apply(self, data):
        return data


class SamplingBiasGenerator(BiasGenerator):
    def __init__(self, parameter, parameter_value, weight=None, bias_strength=0.3):
        super().__init__()

        self.bias_strength = bias_strength  # Probability for items of selected group to be removed
        self.parameter = parameter  # parameter to remove values for
        self.pvalue = parameter_value  # value of parameter to remove values for
        if weight is None:  # optional weight dictionary
            weight = dict()  # example: {label1: {val1: weight, val2: weight}, label2 : {..}
        self.weight = weight  # unfilled weights use value as their weight, if this value is 0, the lowest value above zero is halved and used instead

    def apply(self, data):
        # ensure that a fully functional weight dictionary is present. Fill where empty
        total_weight = dict()
        label_values = data.label_values()
        for key in label_values:
            total_weight[key] = 0
            if key not in self.weight:
                self.weight[key] = dict()
            for value in label_values[key]:
                # fill in missing values
                if value not in self.weight[key]:
                    tmp_value = value
                    # avoid including zero-weight values as these wouldn't receive bias
                    if value == 0:
                        # use half of the lowest non-zero value for these instead
                        tmp_value = min(x for x in label_values[key] if x != 0) / 2.0
                    self.weight[key][value] = tmp_value
                    total_weight[key] += tmp_value
                else:
                    total_weight[key] += self.weight[key][value]
        print(self.weight)
        # drop rows based on weight
        for key in self.weight:
            for value in self.weight[key]:
                # fraction is multiplied by bias_strength to determine the probability for an item to be dropped
                fraction = float(self.weight[key][value]) / float(total_weight[key])
                # potential flaw: as these happen after one another the later values use already modified data
                data.df().drop(
                    data.df().loc[(data.df()[self.parameter] == self.pvalue) & (data.df()[key] == value)].sample(
                        frac=self.bias_strength * fraction).index, inplace=True)
        return data


class MeasurementBiasGenerator:
    def __init__(self, parameter, parameter_value, measurement, weight=None, bias_strength=0.3):
        # Here bias strength indicates the probability in the rows containing the correct biased parameter
        # and value. That the parameter_to_adapt will be increased or decreased by one. Thus the lowest and highest
        # class have only half the effect of this strength as these can only increase or decrease
        self.parameter = parameter
        self.pvalue = parameter_value

        self.measurement = measurement
        self.bias_strength = bias_strength
        if weight is None:
            weight = dict()
        self.weight = weight

        # can be:
        # Higher --> too high measurements
        # Lower --> too low measurements
        # Wider --> Both Lower and Higher
        # Random --> Completely incorrect measurements

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
        data.df().loc[tmp.index.values, self.measurement] += np.round(self.weight["measurement_error"].rvs(len(tmp)))
        # if smaller, set to min value
        data.df().loc[(data.df()[self.measurement] < min(values)), self.measurement] = min(values)
        # if larger, set to max value
        data.df().loc[(data.df()[self.measurement] > max(values)), self.measurement] = max(values)
        # add invalid ratio to selected measurements
        tmp = tmp.sample(frac=self.weight["invalid_ratio"])
        data.df().loc[tmp.index.values, self.measurement] = np.random.randint(min(values), max(values) + 1, len(tmp))
        return data


class OmittedVariableBiasGenerator:
    def __init__(self, parameter_to_omit, parameter_value=None):
        self.parameter_to_omit = parameter_to_omit
        self.pvalue = parameter_value

    def apply(self, data):
        # if no value is provided, remove entire column
        if self.pvalue is None:
            return data.remove_variable(self.parameter_to_omit)
        # if value is provided, remove rows that contain said value
        values = data.df()[self.parameter_to_omit].unique()
        if len(values) == 0:
            return data
        # if it is the final value of this column, remove the column instead of all the rows
        if len(values) == 1 and values[0] == self.pvalue:
            return data.remove_variable(self.parameter_to_omit)
        else:
            data.df().drop(data.df().loc[(data.df()[self.parameter_to_omit] == self.pvalue)].index, inplace=True)
            return data


class CauseEffectBiasGenerator:
    def __init__(self, cpd, data_generator):
        self.name = cpd.variable
        self.ins = cpd.get_evidence()
        self.cpd = cpd
        self.generator = data_generator

    def simulate(self, n=10):
        self.generator.add_node(self.name)
        for i in self.ins:
            self.generator.add_edge(i, self.name)
        self.generator.add_cpds(self.cpd)
        return self.generator.simulate(n)


def sample_model():
    model = BayesianNetwork([('sex', 'income'), ('age', 'income')])

    cpd_sex = TabularCPD('sex', 2, [[0.5], [0.5]])
    cpd_age = TabularCPD('age', 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
    cpd_income = TabularCPD('income', 2, [[0.7, 0.65, 0.6, 0.55, 0.5, 0.7, 0.65, 0.6, 0.55, 0.5],
                                          [0.3, 0.35, 0.4, 0.45, 0.5, 0.3, 0.35, 0.4, 0.45, 0.5]],
                            evidence=["sex", "age"], evidence_card=[2, 5])
    # cpd_h = TabularCPD('H', 2, [[0.2, 0.3, 0.4, 0.5], # h=0 with c=0, p=0;p=1; c=1, p=0;p=1
    #                            [0.8, 0.7, 0.6, 0.5]], # h=1 with c=0, p=0;p=1; c=1, p=0;p=1
    #                   evidence=['C', 'P'], evidence_card=[2, 2])

    model.add_cpds(cpd_sex, cpd_age, cpd_income)
    return model


def test_model(model):
    data = model.simulate(1_000_000)
    if "rich" in data:
        df2 = data.groupby(['sex', 'age'], as_index=False)['income', "rich"].agg(
            {"income": ["mean", "count"], "rich": ["mean"]})
    elif "age" not in data:
        df2 = data.groupby(['sex'], as_index=False)['income'].agg({"mean": "mean", "count": "count"})
    else:
        df2 = data.groupby(['sex', 'age'], as_index=False)['income'].agg({"mean": "mean", "count": "count"})

    # clf = LogisticRegression().fit(data.drop("income", axis=1), data["income"])
    # test_df = model.generator.simulate(50_000)
    # score = clf.score(test_df.drop("income", axis=1), test_df["income"])

    # print(f"Accuracy: {score}")
    print(f"Distribution: \n{df2}")


def test_data(data):
    if "age" not in data.df():
        df2 = data.df().groupby(['sex'], as_index=False)['income'].agg({"mean": "mean", "count": "count"})
    else:
        df2 = data.df().groupby(['sex', 'age'], as_index=False)['income'].agg({"mean": "mean", "count": "count"})

    print(f"Distribution: \n{df2}")


if __name__ == "__main__":
    model = sample_model()
    # test_model(model)

    samplebias = SamplingBiasGenerator(model, ["sex"], ["income"], "sex", 0, 0.7)
    test_model(samplebias)

    measurementbias = MeasurementBiasGenerator(biased_parameter="sex", bp_value=0, parameter_to_adapt="age",
                                               bias_strength=0.3, data_generator=model, bias_type="higher")
    test_model(measurementbias)

    omittedvariablebias = OmittedVariableBiasGenerator("age", model)
    test_model(omittedvariablebias)

    causeeffectbias = CauseEffectBiasGenerator(
        TabularCPD("rich", 2, [[0.8, 0.2],
                               [0.2, 0.8]],
                   evidence=["income"], evidence_card=[2]), model)
    test_model(causeeffectbias)

# bayesian networks library https://pgmpy.org/
# problem with categorical data
# exponential in number of previous variables to next variable,
# this library doesn't work with functions
# field "causality": BN + assumptions, could be interesting
# 360 compatible  https://aif360.mybluemix.net/ https://github.com/Trusted-AI/AIF360
# follow citations to find papers


# mid february halfway meeting
