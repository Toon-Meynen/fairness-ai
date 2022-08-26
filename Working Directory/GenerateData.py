import random
from BayesianNetwork import BayesianNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Data import Data
import pickle

def generate_base(size, features):
    bn = BayesianNetwork()
    bn.addNode("Y")
    bn.addNode("A")
    for i in range(features):
        bn.addNode(f"X{i}")

    bn.addProbability("A", [0.5, 0.5], 2)
    bn.addProbability("Y", [0.6, 0.4], 2)

    for i in range(features):
        bn.addEdge("A", f"X{i}")
        bn.addEdge("Y", f"X{i}")

        r = random.uniform(0.2, 0.8)
        r2 = random.uniform(0.2, 0.8)
        bn.addProbability(f"X{i}", {"A": [r, 1-r], "Y":[r2, 1-r2]}, 2)

    return Data(bn.simulate(size), protected_attributes=["A"], labels=["Y"])


if __name__ == "__main__":
    def test(data, name):
        print(f"testing {name}")
        X_train, X_test, Y_train, Y_test = data.train_test_split(0.2)

        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, Y_train)

        predictions = lr.predict(X_test)
        print(accuracy_score(Y_test, predictions))


    data = generate_base(100_000, 50)
    with open("data/synthetic/50_x.p", "wb") as f:
        pickle.dump(data, f)
    test(data, 50)
    data = generate_base(100_000, 20)
    with open("data/synthetic/20_x.p", "wb") as f:
        pickle.dump(data, f)
    test(data, 20)
    data = generate_base(100_000, 10)
    with open("data/synthetic/10_x.p", "wb") as f:
        pickle.dump(data, f)
    test(data, 10)
    data = generate_base(100_000, 5)
    with open("data/synthetic/5_x.p", "wb") as f:
        pickle.dump(data, f)
    test(data, 5)



