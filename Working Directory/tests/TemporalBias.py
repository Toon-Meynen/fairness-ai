import Bias

if __name__ == "__main__":
    model = Bias.sample_model()
    generator = Bias.DataGenerator(data_generator=model, protected_attributes=["gender"], labels=["income"])
    data = generator.simulate(10_000)
    print(len(data.df()))
    Bias.test_data(data)
    print(data.metrics().disparate_impact())


    def var_mean(hype):
        from scipy.stats import norm
        return norm.rvs(size=len(hype), loc=hype, scale=2)

    def var_meant(time):
        from scipy.stats import norm
        import numpy as np
        time = np.asarray(time)
        time = 5 * time / max(time)
        return 50 * norm.pdf(x=time, loc=2.5, scale=1)

    bias = Bias.TemporalBiasGenerator(parameter="hyped_param", primary_distribution=var_mean, hype_distribution=var_meant)
    data = bias.apply(data)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    data.shuffle()
    Bias.test_data(data)


    bias = Bias.TemporalBiasGenerator(parameter="hyped_param", primary_distribution=var_mean, hype_distribution=var_meant, use_index_as_time=False)
    data = bias.apply(data)
    Bias.test_data(data)
    print(data.metrics().disparate_impact())

    print(data.df().head())



