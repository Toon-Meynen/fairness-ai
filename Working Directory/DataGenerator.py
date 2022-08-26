from Data import Data
import pgmpy.models
import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataGenerator:
    def __init__(self, data_generator, protected_attributes, labels):
        self.data_generator = data_generator
        self._prot_attr = protected_attributes
        self._labels = labels

    # generates n items
    def simulate(self, n=10, seed=None):
        return Data(df=self.data_generator.simulate(n, seed=seed), protected_attributes=self._prot_attr, labels=self._labels)


class BiasGenerator:
    def __init__(self, seed=None):
        self.seed = seed

    def apply(self, data):
        return data


def sample_model():
    bn = BayesianNetwork.BayesianNetwork()
    bn.addEdge("gender", "income")
    bn.addEdge("age", "income")
    bn.addEdge("hard-working", "income")
    bn.addProbability("gender", [0.5, 0.5], 2)
    bn.addProbability("age", scipy.stats.randint(0, 5), 5)
    bn.addProbability("hard-working", [0.8, 0.2], 2)
    bn.addProbability("income", {"gender": scipy.stats.uniform(),
                                 "age": scipy.stats.genhalflogistic(0.773),
                                 "hard-working": scipy.stats.genhalflogistic(0.773),
                                 "income": [0.6, 0.4]}, 2)
    bn._network.check_model()
    return bn


def old_sample_model():
    model = pgmpy.models.BayesianNetwork([('gender', 'income'), ('age', 'income')])

    cpd_sex = TabularCPD('gender', 2, [[0.5], [0.5]])
    cpd_age = TabularCPD('age', 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
    cpd_income = TabularCPD('income', 2, [[0.7, 0.65, 0.6, 0.55, 0.5, 0.7, 0.65, 0.6, 0.55, 0.5],
                                          [0.3, 0.35, 0.4, 0.45, 0.5, 0.3, 0.35, 0.4, 0.45, 0.5]],
                            evidence=["gender", "age"], evidence_card=[2, 5])
    # cpd_h = TabularCPD('H', 2, [[0.2, 0.3, 0.4, 0.5], # h=0 with c=0, p=0;p=1; c=1, p=0;p=1
    #                            [0.8, 0.7, 0.6, 0.5]], # h=1 with c=0, p=0;p=1; c=1, p=0;p=1
    #                   evidence=['C', 'P'], evidence_card=[2, 2])

    model.add_cpds(cpd_sex, cpd_age, cpd_income)
    return model


def police_model():
    model = BayesianNetwork.BayesianNetwork()

    model.addNode("race")
    model.addNode("gender")
    model.addNode("drugs")
    model.addNode("searched")
    model.addNode("drugs-detected")

    model.addEdge("drugs", "drugs-detected")
    model.addEdge("searched", "drugs-detected")

    model.addProbability("race", scipy.stats.randint(0, 2), 2)
    model.addProbability("gender", scipy.stats.randint(0, 2), 2)
    model.addProbability("drugs", [0.9, 0.1], 2)
    model.addProbability("searched", [0.9, 0.1], 2)
    model.addProbability("drugs-detected", {"drugs": [0.1, 0.9],
                                            "searched": [0.3, 0.7]}, 2)

    return model


def test_data(data):
    if "age" not in data.df():
        df2 = data.df().groupby(['gender'], as_index=False)['income'].agg({"mean": "mean", "count": "count"})
    elif "hard-working" not in data.df():
        df2 = data.df().groupby(['gender', 'age'], as_index=False)['income'].agg({"mean": "mean", "count": "count"})
        sns.barplot(x="age", hue="gender", y="mean", data=df2)
    else:
        df2 = data.df().groupby(['gender', 'age', "hard-working"], as_index=False)['income'].agg(
            {"mean": "mean", "count": "count"})
        sns.catplot(x="age", y="mean", hue="gender", col="hard-working", data=df2, kind="bar")
    plt.show()

    if "hyped_param" in data.df():
        plt.plot(range(len(data.df())), data.df()["hyped_param"])
        plt.show()

def test_data2(data):
    df2 = data.df().groupby(['group', 'gender'], as_index=False)['income'].agg(
        {"mean": "mean", "count": "count"})
    print(df2)
    df3 = data.df().groupby(['gender'], as_index=False)['income'].agg(
        {"mean": "mean", "count": "count"})
    print(df3)


def test_police_data(data):
    df2 = data.df().groupby(['gender', 'race', "searched", "drugs"], as_index=False)['drugs-detected'].agg(
        {"mean": "mean", "count": "count"})
    sns.catplot(x="drugs", y="count", hue="race", col="searched", data=df2, kind="bar")
    sns.catplot(x="drugs", y="mean", hue="race", col="searched", data=df2, kind="bar")
    plt.show()


# bayesian networks library https://pgmpy.org/
# problem with categorical data
# exponential in number of previous variables to next variable,
# this library doesn't work with functions
# field "causality": BN + assumptions, could be interesting
# 360 compatible  https://aif360.mybluemix.net/ https://github.com/Trusted-AI/AIF360
# follow citations to find papers


# mid february halfway meeting

