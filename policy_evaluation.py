"""
Main method to evaluate a policy.
"""

import numpy as np


class PolicyEvaluation:
    def __init__(self, loss_max, get_p_dx_data, get_p_dx_policy, x_name="x"):
        self.loss_max = loss_max
        self.func_get_p_dx_data = get_p_dx_data
        self.func_get_p_dx_policy = get_p_dx_policy
        self.x_name = x_name

    @staticmethod
    def get_loss(df):
        loss = df["y"]
        return loss.to_numpy()

    def get_weight(self, d, x):
        p_dx_old = self.func_get_p_dx_data(x)
        p_dx_new = self.func_get_p_dx_policy(x, d)
        weight = np.where(d == 0, (1 - p_dx_new) / (1 - p_dx_old), p_dx_new / p_dx_old)
        return weight

    def get_prop_weight(self, d, x):
        p_dx_old = self.func_get_p_dx_data(x)
        weight = np.where(d == 0, 1 / (1 - p_dx_old), 1 / p_dx_old)
        return weight

    def get_weight_bounds(self, d, x, gamma):
        prop_weight = self.get_prop_weight(d, x)

        weight_low = 1 + 1 / gamma * (prop_weight - 1)
        weight_high = 1 + gamma * (prop_weight - 1)

        p_dx_new = self.func_get_p_dx_policy(x, d)
        weight_low = np.where(
            d == 0, (1 - p_dx_new) * weight_low, p_dx_new * weight_low
        )
        weight_high = np.where(
            d == 0, (1 - p_dx_new) * weight_high, p_dx_new * weight_high
        )

        return weight_low, weight_high

    def get_robust_quantiles(self, quant_list, df, df_beta, gamma):
        loss = self.get_loss(df)
        d = df["d"].to_numpy()
        x = df[self.x_name].to_numpy()

        weight_low, weight_high = self.get_weight_bounds(d, x, gamma)

        loss = loss[weight_low != 0]
        weight_high = weight_high[weight_low != 0]
        weight_low = weight_low[weight_low != 0]

        d_weight = self.func_get_p_dx_policy(
            df_beta[self.x_name].to_numpy(), df_beta["d"].to_numpy()
        )
        x_beta = df_beta[self.x_name].to_numpy()
        weight_low_beta, weight_high_beta = self.get_weight_bounds(
            d_weight, x_beta, gamma
        )

        sort_idx = np.argsort(loss)
        loss = loss[sort_idx]
        weight_low = weight_low[sort_idx]
        weight_high = weight_high[sort_idx]

        weight_cum_low = np.cumsum(weight_low)
        weight_cum_high = np.cumsum(weight_high[::-1])[::-1]

        loss_n1 = np.append(loss, self.loss_max)

        beta_n1, weight_beta_n1 = self.get_weight_cf(weight_high_beta)
        idx_beta = 1
        alpha_n1 = np.ones(len(loss_n1))
        while idx_beta < len(
            beta_n1
        ):  # and beta_n1[idx_beta] <= 1 - np.min(quant_list):
            weight_inf = weight_beta_n1[idx_beta]

            weight_cum = weight_cum_low + np.append(weight_cum_high[1:], 0) + weight_inf
            weight_cum_n1 = np.append(weight_cum, weight_cum_low[-1] + weight_inf)
            weight_cum_low_n1 = np.append(
                weight_cum_low, weight_cum_low[-1] + weight_inf
            )

            F_hat_n1 = weight_cum_low_n1 / weight_cum_n1
            beta = beta_n1[idx_beta]
            alpha_temp = 1 - (1 - beta) * F_hat_n1
            alpha_temp = np.where(beta < alpha_temp, alpha_temp, 1)
            alpha_n1 = np.where(alpha_temp < alpha_n1, alpha_temp, alpha_n1)
            idx_beta += 1
        alpha_n1 = np.where(alpha_n1 == 1, 0, alpha_n1)

        loss_array = self.loss_max * np.ones(len(quant_list))
        for m, quant_alpha in enumerate(quant_list):
            idx = np.argwhere(alpha_n1 < quant_alpha)
            if len(idx) == 0:
                loss_array[m] = self.loss_max
            else:
                loss_array[m] = loss_n1[idx[0]]
        return loss_array, loss_n1, alpha_n1

    def get_quantiles_non_adjusted(self, quant_list, df):
        loss_list = []

        loss = self.get_loss(df)
        weight = self.get_weight(df["d"].to_numpy(), df[self.x_name].to_numpy())

        loss = loss[weight != 0]
        weight = weight[weight != 0]

        sort_idx = np.argsort(loss)
        loss = loss[sort_idx]
        weight = weight[sort_idx]

        weight = weight / np.sum(weight)
        alphas = np.cumsum(weight)

        for quant_alpha in quant_list:
            idx = np.argwhere(alphas > 1 - quant_alpha)[0, 0]
            loss_list += [loss[idx]]

        return loss_list, alphas

    @staticmethod
    def get_weight_cf(weight_all):
        n = len(weight_all)
        weight_n1 = 1 / (n + 1) * np.ones(n + 1)
        alpha_n1 = np.cumsum(weight_n1)
        weight_n1 = np.append(weight_all, 1000)
        weight_n1 = np.sort(weight_n1)
        weight_n1 = weight_n1[::-1]
        return alpha_n1, weight_n1

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Policy evaluation")

        parser.add_argument(
            "--gamma_list",
            type=float,
            default=[1, 2, 3],
            nargs="+",
            help="list of gamma (default: %(default)s)",
        )

        parser.add_argument(
            "--y_max",
            type=float,
            default=100,
            help="y_max for conformal prediction (default: %(default)s)",
        )

        return parent_parser
