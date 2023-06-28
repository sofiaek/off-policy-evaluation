"""
Create synthetic data.

x - features/covariates
y - outcome
d - decision/treatment
"""

import numpy as np
import pandas as pd

from scipy.special import expit
from functools import partial


class DataGen:
    def __init__(self, rng, case):
        self.rng = rng
        self.case = case

    def get_p_dx_data(self, x):
        p_x = np.empty(len(x))
        if self.case == "case1":
            p_x = expit((x + 1) * 0.5)
        elif self.case == "case2":
            p_x = expit((x + 1) * 1)
        elif self.case == "case3":
            p_x = expit((x + 1) * 2)
        elif self.case == "case4":
            p_x = np.where((x < 1 / 3), 0.05, 0.95)
        elif self.case == "case5":
            p_x = np.where((np.logical_and(x >= 1 / 3, x < 2 * 1 / 3)), 0.05, 0.95)
        return p_x

    def get_p_dx_policy(self, x, dummy, th):
        if th == 10:
            p_new = self.get_p_dx_data(x)
        else:
            p_new = np.where(x <= th, 0, 1)
        return p_new

    def get_data(self, n):
        df = self.__generate_data(n, self.get_p_dx_data)
        return df

    def get_test_data(self, n, th):
        df = self.__generate_data(n, partial(self.get_p_dx_policy, dummy=0, th=th))
        return df

    def __generate_data(self, n, get_p_dx):
        x1 = self.rng.uniform(0, 1, n)
        x2 = self.rng.uniform(0, 1, n)

        p_dx = get_p_dx(x1 * x2)
        d = self.rng.binomial(1, p_dx)

        y = np.where(d == 0, x1 * x2, 1.0 - x1 * x2) + self.rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x1 * x2, "x1": x1, "x2": x2, "d": d, "y": y})
        return df

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data generator")

        parser.add_argument(
            "--n_train",
            nargs="+",
            type=int,
            default=[1000],
            help="number of samples for training (default: %(default)s)",
        )
        parser.add_argument(
            "--n_test",
            type=int,
            default=1000,
            help="number of samples for testing (default: %(default)s)",
        )

        return parent_parser


class DataGenConfounding(DataGen):
    def __init__(self, rng, case):
        super().__init__(rng, case)
        self.rng = rng
        self.case = case

    def get_p_dx_gamma(self, x, u, gamma, case_gamma="hard"):
        p_x = self.get_p_dx_data(x)
        y = -x - u
        th = -0.18
        p = np.empty(len(x))
        if case_gamma == "easy":
            p = np.where(
                y > th,
                1 / (1 + 1 / gamma * (1 / p_x - 1)),
                1 / (1 + gamma * (1 / p_x - 1)),
            )
        if case_gamma == "hard":
            p = np.where(
                y > th,
                1 / (1 + gamma * (1 / p_x - 1)),
                1 / (1 + 1 / gamma * (1 / p_x - 1)),
            )
        if case_gamma == "1":
            p = p_x
        return p

    def get_p_dx_policy(self, x, dummy, th):
        p_new = np.where(x <= th, 0, 1)
        return p_new

    def get_data_gamma(self, n, gamma, case_gamma):
        df = self.__generate_data(
            n, partial(self.get_p_dx_gamma, gamma=gamma, case=case_gamma)
        )
        return df

    def get_data(self, n):
        df = self.__generate_data(n, partial(self.get_p_dx_gamma, gamma=2.0))
        return df

    def get_test_data(self, n, th):
        df = self.__generate_data(n, partial(self.get_p_dx_policy, th=th))
        return df

    def __generate_data(self, n, get_p_dx):
        x1 = self.rng.uniform(0, 1, n)
        x2 = self.rng.uniform(0, 1, n)
        u = self.rng.normal(0, 0.1 * (x1 + x2), n)  #

        p_dx = get_p_dx(x1 * x2, u)
        d = self.rng.binomial(1, p_dx)

        y = np.where(d == 0, x1 * x2, 1 - x1 * x2) + u
        df = pd.DataFrame({"x": x1 * x2, "x1": x1, "x2": x2, "d": d, "y": y})
        return df
