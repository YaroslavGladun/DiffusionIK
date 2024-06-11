import math


class SeedEpsilonIKConfig:
    # dataset
    max_seed_dist = 5 * math.pi / 180

    # model
    d_model = 256

    # sampling dataset for nearest ik
    upper_bound = 0.005
    linspace_size = 1000
