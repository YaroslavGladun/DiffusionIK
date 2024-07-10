import math


class SeedEpsilonIKConfig:
    # dataset
    max_seed_dist = 90 * math.pi / 180

    # model
    d_model = 2048

    # sampling dataset for nearest ik
    upper_bound = 0.005
    linspace_size = 1000

    # training
    learning_rate = 1e-3

