import math


class SeedEpsilonIKConfig:
    # dataset
    max_seed_dist = math.pi / 12

    # model
    d_model = 1024

    # sampling dataset for nearest ik
    upper_bound = 0.005
    linspace_size = 1000
