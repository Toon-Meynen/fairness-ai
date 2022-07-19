# classic --> debias --> rebias
import random

from BayesianNetwork import BayesianNetwork
from Data import Data
from aif360.datasets import AdultDataset, BankDataset, CompasDataset, GermanDataset, LawSchoolGPADataset
from pprint import pprint

## EXTRACT AND CONVERT POPULAR DATASET

def fromAif(data):
    extracted_df = data.convert_to_dataframe()
    df = extracted_df[0]
    pa = extracted_df[1]["protected_attribute_names"]
    l  = extracted_df[1]["label_names"]
    return Data(df=df, protected_attributes=pa, labels=l)


## PREPROC METHODS
def disparateImpactRemover(data, sensitive_attribute):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    return DisparateImpactRemover(repair_level=1.0, sensitive_attribute=sensitive_attribute).fit_transform(data)

def lfr(data):
    from aif360.algorithms.preprocessing import LFR
    return LFR(unprivileged_groups=fromAif(data).privilegedGroups(),
               privileged_groups=fromAif(data).unprivilegedGroups(),
               k=5,
               Ax=0.01,
               Ay=1.0,
               Az=50.0,
               print_interval=250,
               verbose=0,
               seed=None).fit_transform(data)

def optimPreproc(data):
    from aif360.algorithms.preprocessing import OptimPreproc
    return OptimPreproc(optimizer="",
                        optim_options={},
                        unprivileged_groups=fromAif(data).unprivilegedGroups(),
                        privileged_groups=fromAif(data).privilegedGroups(),
                        verbose=0,
                        seed=None).fit_transform(data)

def reweighing(data):
    from aif360.algorithms.preprocessing import Reweighing
    return Reweighing(unprivileged_groups=fromAif(data).unprivilegedGroups(),
                      privileged_groups=fromAif(data).privilegedGroups()
                      ).fit_transform(data)


## CUSTOM METHODS
def collectMetrics(data):
    M = data.metrics()
    rval = {}
    #rval["base_rate"] = M.base_rate()
    #rval["consistency"] = M.consistency()
    # rval["difference"] = M.difference() # missing "metric_fun"
    rval["disparate_impact"] = M.disparate_impact()
    #rval["mean_difference"] = M.mean_difference()
    #rval["num_instances"] = M.num_instances()
    #rval["num_negatives"] = M.num_negatives()
    #rval["num_positives"] = M.num_positives()
    # rval["ratio"] = M.ratio() # missing "metric_fun"
    # rval["rich_subgroup"] = M.rich_subgroup() # missing "predictions"
    #rval["smoothed_empirical_differential_fairness"] = M.smoothed_empirical_differential_fairness()
    #rval["statistical_parity_difference"] = M.statistical_parity_difference()
    return rval

def adultDataset():
    protected = "sex"
    data = AdultDataset(protected_attribute_names=[protected],
    privileged_classes=[['Male']], categorical_features=[],
    features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

    return data, protected

def generateDatasets():
    datasets = []

    ### Base set
    bn = BayesianNetwork()
    bn.addNode("Y")
    bn.addNode("A")
    N = 10
    for i in range(N):
        n = "X" + str(i)
        bn.addNode(n)
        bn.addEdge(n, "A")
        bn.addEdge(n, "Y")
        x = 0
        while x < 0.2 or x > 0.8:
            x = random.random()
        bn.addProbability(n, [x, 1-x], 2)

    pa = {}
    py = {}
    for i in range(N):
        n = "X" + str(i)

        x = 0
        while x < 0.2 or x > 0.8:
            x = random.random()
        pa[n] = [x, 1-x]

        x = 0
        while x < 0.2 or x > 0.8:
            x = random.random()
        py[n] = [x, 1-x]

    pa["A"] = [0.7, 0.3]
    py["Y"] = [0.3, 0.7]

    bn.addProbability("A", pa, 2) # 70% has non-biased attribute
    bn.addProbability("Y", py, 2) # 30% gets positive label

    base_data = bn.simulate(100_000)
    print(base_data.A.sum())
    print(base_data.Y.sum())

    ### UNIT SETS





    return datasets


if __name__ == "__main__":
    generateDatasets()


    # for base in [AdultDataset, BankDataset, CompasDataset, GermanDataset, LawSchoolGPADataset]:
    # for debias in [disparateImpactRemover, lfr, optimPreproc, reweighing]:


    #metrics = {}
    #data, protected = adultDataset()
    #debias = disparateImpactRemover

    #print(fromAif(data).privilegedGroups())
    #metrics["base"] = collectMetrics(fromAif(data))
    #data = debias(data, protected)
    #metrics["debiased"] = collectMetrics(fromAif(data))

    #pprint(metrics)
