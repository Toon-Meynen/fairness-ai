import Bias
import scipy.stats

if __name__ == "__main__":
    base_model = Bias.sample_model()
    generator = Bias.SimpsonsParadoxGenerator(base_model=base_model,
                                              groups=[{"probabilities":[
                                                           ["income", {"gender": [0.4, 0.6],
                                                                       "age": scipy.stats.genhalflogistic(0.773),
                                                                       "hard-working": scipy.stats.genhalflogistic(
                                                                           0.773),
                                                                       "income": [0.6, 0.4]}, 2]]}
                                                  ,
                                                      {"probabilities":[
                                                           ["income", {"gender": [0.6, 0.4],
                                                                       "age": scipy.stats.genhalflogistic(0.4),
                                                                       "hard-working": scipy.stats.genhalflogistic(
                                                                           0.4),
                                                                       "income": [0.6, 0.4]}, 2]]}
                                                      ],
                                              protected_attributes=["gender"], labels=["income"])
    data = generator.simulate(100_000)
    print(len(data.df()))
    Bias.test_data2(data)
    print(data.metrics().disparate_impact())
