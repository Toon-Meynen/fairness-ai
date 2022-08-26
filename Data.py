import aif360.metrics
import numpy as np
import aif360.datasets
from sklearn.model_selection import train_test_split
import tqdm
import copy

class Data:
    def __init__(self, df, protected_attributes, labels, weights=None, scores=''):
        self._df = df
        self._y = labels
        self._scores = scores
        self._p = protected_attributes

        self._privileged_groups = dict()
        self._unprivileged_groups = dict()
        for p in self._p:
            self._privileged_groups[p] = [np.max(self._df[p])]
            self._unprivileged_groups[p] = list(set(self._df[p].unique()) - set(self._privileged_groups[p]))

        self._isBinary = self._setBinary()

        if weights is None:
            weights = [1] * len(self._df)
        self._df.loc[:, "__weight__"] = np.asarray(weights)

    # Basic
    def X(self):
        return self.df(weight=False).drop(self._y, axis=1)

    def y(self):
        return self.df(weight=False).loc[:, self._y].values.ravel()

    def weights(self):
        return self.df(weight=True).loc[:, "__weight__"].values.ravel()

    def set_labels(self, new_labels):
        self.df(weight=True).loc[:, self._y] = new_labels.reshape(len(new_labels), 1)
        return self

    def set_scores(self, new_scores, name="scores"):
        self.df(weight=True).loc[:, name] = new_scores
        self._scores = name
        return self

    def df(self, weight=False):
        if weight:
            return self._df
        return self._df.drop('__weight__', axis=1)

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.df(weight=False))

    # aif
    def metrics(self, other=None):
        if other:
            return aif360.metrics.ClassificationMetric(self.aif(), other.aif(),
                                                       privileged_groups=[self._privileged_groups],
                                                       unprivileged_groups=[self._unprivileged_groups])
        else:
            return aif360.metrics.BinaryLabelDatasetMetric(self.aif(),
                                                       privileged_groups=[self._privileged_groups],
                                                       unprivileged_groups=[self._unprivileged_groups])

    def aif(self):
        if self._isBinary:
            rval = aif360.datasets.BinaryLabelDataset(df=self._df,
                                                      label_names=self._y,
                                                      protected_attribute_names=self._p,
                                                      instance_weights_name="__weight__",
                                                      scores_names=self._scores)
            rval.validate_dataset()
            return rval
        else:
            raise NotImplemented("support for non binary label dataset")

    def privilegedGroups(self):
        return self._privileged_groups

    def unprivilegedGroups(self):
        return self._unprivileged_groups

    # misc
    def labels(self):
        return self._y

    def label_values(self):
        rval = dict()
        for y in self._y:
            rval[y] = self.df(weight=False)[y].unique()
        return rval

    def remove_variable(self, variable):
        # remove a column
        self.df(weight=True).drop(variable, axis=1, inplace=True)
        # ensure this column is also removed from all other locations
        if variable in self._y:
            self._y.remove(variable)
        if variable in self._p:
            self._p.remove(variable)
            self._privileged_groups.pop(variable)
            self._unprivileged_groups.pop(variable)
        self._isBinary = self._setBinary()
        return self

    # data frame altering functions
    def drop(self, indices, inplace=False):
        return self._df.drop(indices, inplace=inplace)

    def filter(self, f):
        d = self.copy()
        d._df = self._df.loc[f]
        return d

    # internal functions
    def _setBinary(self):
        for y in self._y:
            if len(self._df[y].unique()) > 2:
                return False
        return True

    def shuffle(self, seed=None):
        """
           Shuffles the data

           :seed: seed to shuffle with, if None, a random seed is used
           :return: /
           """
        from sklearn.utils import shuffle
        self._df = shuffle(self.df(weight=True), random_state=seed)

    def head(self):
        return self.df(weight=False).head()

    def split(self, size, seed=None):
        train, test = train_test_split(self._df, test_size=size, random_state=seed)
        return Data(train, protected_attributes=self._p, labels=self._y), Data(test, protected_attributes=self._p, labels=self._y),

    def train_test_split(self, size, seed=None):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X(), self.y(), test_size=size, random_state=seed)
        return X_train, X_test, Y_train, Y_test

def fromAif(data):
    extracted_df = data.convert_to_dataframe()
    df = extracted_df[0]
    pa = extracted_df[1]["protected_attribute_names"]
    l  = extracted_df[1]["label_names"]
    w  = extracted_df[1]["instance_weights"]
    return Data(df=df, protected_attributes=pa, labels=l, weights=w)

#
#     def display(self):
#         pass
#
#     def delete(self, idx):
#         pass
#
#     def __len__(self):
#         if self.x is None:
#             return 0
#         return len(self.x)
#
#     def __getitem__(self, item):
#         pass
#
#     def __iter__(self):
#         return DataIterator(self)
#
#
# class DataIterator:
#     def __init__(self, data):
#         self._data = data
#         self._index = 0
#
#     def __next__(self):
#         if self._index < len(self._data):
#             result = self._data[self._index]
#             self._index += 1
#             return result
#         raise StopIteration
