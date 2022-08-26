import Bias

if __name__ == "__main__":
    model = Bias.sample_model()
    bias = Bias.DataGenerator(data_generator=model, protected_attributes=["gender"], labels=["income"])

    data = bias.simulate(100_000)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    print()
    bias = Bias.CauseEffectBiasGenerator(new_parameter="ice_cream_consumption", correlated_parameter="income", probability=0.6, consecutive=False)
    data = bias.apply(data)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())
