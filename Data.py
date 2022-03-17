import aif360.metrics
import numpy as np
import aif360.datasets
import tqdm


class Data:
    def __init__(self, df, protected_attributes, labels):
        self._df = df
        self._y = labels
        self._p = protected_attributes

        self._privileged_groups = dict()
        self._unprivileged_groups = dict()
        for p in self._p:
            self._privileged_groups[p] = [np.max(self._df[p])]
            self._unprivileged_groups[p] = list(set(self._df[p].unique()) - set(self._privileged_groups[p]))

        self._isBinary = self._setBinary()

    # Basic
    def X(self):
        return self._df.drop(self._y, axis=1)

    def y(self):
        return self._df.loc[:, self._y]

    def df(self):
        return self._df

    # aif
    def metrics(self):
        return aif360.metrics.BinaryLabelDatasetMetric(self.aif(),
                                                       privileged_groups=[self._privileged_groups],
                                                       unprivileged_groups=[self._unprivileged_groups])

    def aif(self):
        if self._isBinary:
            rval = aif360.datasets.BinaryLabelDataset(df=self._df,
                                                      label_names=self._y,
                                                      protected_attribute_names=self._p)
            rval.validate_dataset()
            return rval
        else:
            raise NotImplemented("support for non binary label dataset")

    # misc
    def labels(self):
        return self._y

    def label_values(self):
        rval = dict()
        for y in self._y:
            rval[y] = self.df()[y].unique()
        return rval

    def remove_variable(self, variable):
        # remove a column
        self.df().drop(variable, axis=1, inplace=True)
        # ensure this column is also removed from all other locations
        if variable in self._y:
            self._y.remove(variable)
        if variable in self._p:
            self._p.remove(variable)
            self._privileged_groups.pop(variable)
            self._unprivileged_groups.pop(variable)
        self._isBinary = self._setBinary()
        return self

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
        self._df = shuffle(self.df(), random_state=seed)

    def head(self):
        return self.df().head()

#     def train_test_split(self):
#         pass
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
