import pgmpy.models
import scipy.stats
import pgmpy.factors.discrete
import numpy as np
import scipy.special
import pgmpy_test


class BayesianNetwork:
    def __init__(self):
        self._network = pgmpy.models.BayesianNetwork()

    def addEdge(self, a, b):
        self._network.add_edge(a, b)

    def addProbability(self, a, p, size):
        if a in self._network.get_roots():
            try:
                # discrete probability type
                cpd = pgmpy.factors.discrete.TabularCPD(a, size, [[p.pmf(i)] for i in range(size)])
            except AttributeError:
                # list type
                cpd = pgmpy.factors.discrete.TabularCPD(a, size, [[i] for i in p] )
            #TODO: add block for continuous probability type
            self._network.add_cpds(cpd)
        else:
            parents = self._network.get_parents("income")
            # ensure probability dictionary is complete
            if a not in p:
                raise KeyError(f"{a} does not have a probability distribution")
            for i in p:
                if i not in parents and i is not a:
                    raise KeyError(f"{i} is not a parent of {a}")
            for i in parents:
                if i not in p:
                    raise KeyError(f"no statistical function provided for parent: {i}")

            cpd_matrix = []
            for value in range(size):
                # calculate probabilities for each value of outcome
                matrices = []
                for parent in parents:
                    # divide by length to ensure i is within [0, 1]
                    tmp = np.asarray([[p[parent].pdf(i / len(self._network.get_cpds(parent).values))]
                                      for i in range(len(self._network.get_cpds(parent).values))])
                    matrices.append(tmp)

                matrix = None
                # multiply all matrices with each other to get a layer of cpd
                for m in matrices:
                    if matrix is None:
                        matrix = m
                    else:
                        matrix = np.matmul(matrix, m.T).flatten()
                        matrix = matrix.reshape((len(matrix), 1))

                # add weight of the specific outcome value
                matrix = matrix * p[a][value]

                cpd_matrix.extend(matrix.T)

            cpd_matrix = np.asarray(cpd_matrix)
            print(cpd_matrix)
            print()
            # ensure each column sums to 1
            # using softmax
            cpd_matrix = scipy.special.softmax(cpd_matrix, axis=0)
            # using normalization
            #cpd_matrix /= np.asarray([cpd_matrix.T[i].sum() for i in range(len(cpd_matrix.T))])
            print(cpd_matrix)

            cpd = pgmpy.factors.discrete.TabularCPD(a, size, cpd_matrix, evidence=parents,
                                                    evidence_card=[len(self._network.get_cpds(parent).values) for parent
                                                                   in parents])
            self._network.add_cpds(cpd)


if __name__ == "__main__":
    bn = BayesianNetwork()
    bn.addEdge("sex", "income")
    bn.addEdge("age", "income")
    bn.addProbability("sex", [0.5, 0.5], 2)
    bn.addProbability("age", scipy.stats.randint(0, 5), 5)
    bn.addProbability("income", {"sex": scipy.stats.uniform(),
                                 "age": scipy.stats.genhalflogistic(0.773),
                                 "income": [0.6, 0.4]}, 2)
    #print(bn._network.get_cpds("sex"))
    #print(bn._network.get_cpds("age"))
    tmp = bn._network.get_cpds("income")
    model = pgmpy_test.sample_model()
    #print(tmp)
    print()
    bn._network.check_model()

"""
    def sample_model():
        model = BayesianNetwork([('sex', 'income'), ('age', 'income')])

        cpd_sex = TabularCPD('sex', 2, [[0.5], [0.5]])
        cpd_age = TabularCPD('age', 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
        cpd_income = TabularCPD('income', 2, [[0.7, 0.65, 0.6, 0.55, 0.5, 0.7, 0.65, 0.6, 0.55, 0.5],
                                              [0.3, 0.35, 0.4, 0.45, 0.5, 0.3, 0.35, 0.4, 0.45, 0.5]],
                                evidence=["sex", "age"], evidence_card=[2, 5])


        model.add_cpds(cpd_sex, cpd_age, cpd_income)
        return model
"""
