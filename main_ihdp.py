"""
Evaluate policy and generate limit curves and figures for miscoverage gaps for the IHDP data set used in
"Off-Policy Evaluation with Out-of-Sample Guarantees"

For the same setup used in the paper use

For coverage: python main_ihdp.py --y_max 100 --gamma_list 1, 1.5, 2 --th_list 0, 1 --coverage --n_mc 1000
For loss: python main_ihdp.py --y_max 100 --gamma_list 1, 1.5, 2 --th_list 0, 1, 1
"""
import argparse
import os

import numpy as np
import pandas as pd

import plot_results
import save_utils

from functools import partial
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from policy_evaluation import PolicyEvaluation
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from main_synthetic_coverage import test_loss


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run simulations to test coverage",
    )

    parser.add_argument(
        "--n_mc",
        type=int,
        default=1000,
        help="number of monte carlo simulations. Ignored if loss (default: %(default)s)",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="proportion of samples for loss curve (default: %(default)s)",
    )

    parser.add_argument(
        "--th_list",
        nargs="+",
        type=float,
        default=[0, 1],
        help="thresholds of evaluated policy (default: %(default)s)",
    )

    # add model specific args
    parser = save_utils.add_specific_args(parser)
    parser = PolicyEvaluation.add_model_specific_args(parser)

    args = parser.parse_args()

    return args


def get_p_dx_policy(x, d, th, use_past):
    if use_past:
        p_dx = d
    else:
        p_dx = th * np.ones(len(x))
    return p_dx


def get_p_dx_data(x):
    test = clf.predict_proba(x)
    return test[:, 1]


def prepare_original_df(df_ihdp, x_name, rng):
    df_ihdp = df_ihdp.rename(columns={0: "d"})

    beta = np.random.choice(5, (1, len(x_name)), p=[0.5, 0.2, 0.15, 0.1, 0.05])

    y_temp = np.random.normal(-np.sum(beta * df_ihdp[x_name], axis=1), 1)

    y = np.where(df_ihdp["d"] == 0, y_temp, y_temp - 4)

    df_ihdp["y"] = y

    y_test = np.random.normal(-np.sum(beta * df_ihdp[x_name], axis=1), 1)
    df_ihdp["y0"] = y_test
    df_ihdp["y1"] = y_test - 4

    scaler = preprocessing.MinMaxScaler()
    # scaler = preprocessing.StandardScaler()

    df_ihdp.loc[:, x_name] = scaler.fit_transform(
        df_ihdp[x_name]
    )  # Train on all/one part (like beta)?

    df_train, df_test = train_test_split(
        df_ihdp, test_size=0.1, random_state=rng.integers(10000)
    )

    return df_train, df_test


def main():
    args = create_parser()

    out_dir = ""
    if args.save:
        out_dir = save_utils.save_logging_and_setup(args)

    quant_arr = np.linspace(0.0001, 0.9999, 51)

    # Read data
    df_ihdp = pd.read_csv("data/ihdp_npci_1.csv", header=None)
    x_name = [i for i in range(5, 30)]

    if args.coverage:
        loop_coverage(df_ihdp, x_name, args, quant_arr, out_dir)
    else:
        loop_ihdp_loss(df_ihdp, x_name, args, quant_arr, out_dir)


def loop_coverage(df_ihdp, x_name, args, quant_arr, out_dir):
    alpha_th0_tot = np.zeros((len(args.gamma_list), np.size(quant_arr)))
    alpha_th1_tot = np.zeros((len(args.gamma_list), np.size(quant_arr)))
    alpha_non_adjusted_tot = [
        np.zeros(np.size(quant_arr)),
        np.zeros(np.size(quant_arr)),
    ]

    alpha_tot = [alpha_th0_tot, alpha_th1_tot]

    for i_mc in range(args.n_mc):
        if i_mc % 50 == 0:
            print("i={}".format(i_mc))

        rng = default_rng(i_mc)
        np.random.seed(i_mc)

        df_train_ihdp, df_test = prepare_original_df(df_ihdp, x_name, rng)
        df_train, df_beta = train_test_split(
            df_train_ihdp, test_size=args.test_size, random_state=rng.integers(10000)
        )

        global clf
        clf = LogisticRegression(random_state=rng.integers(10000)).fit(
            df_beta[x_name].to_numpy(), df_beta["d"]
        )

        for ii, th_i in enumerate(args.th_list):
            robust_alpha = PolicyEvaluation(
                args.y_max,
                get_p_dx_data,
                partial(get_p_dx_policy, th=th_i, use_past=False),
                x_name,
            )

            loss_list_non_adjusted, __ = robust_alpha.get_quantiles_non_adjusted(
                quant_arr.tolist(), df_train_ihdp
            )
            alpha_non_adjusted_tot[ii] += test_loss(
                loss_list_non_adjusted, df_test["y{}".format(ii)].to_numpy()
            )

            for i, gamma_i in enumerate(args.gamma_list):
                loss_list, __, __ = robust_alpha.get_robust_quantiles(
                    quant_arr.tolist(), df_train, df_beta, gamma_i
                )
                alpha_tot[ii][i, :] += test_loss(
                    loss_list, df_test["y{}".format(ii)].to_numpy()
                )

    for i, _ in enumerate(args.th_list):
        alpha_tot[i] = alpha_tot[i] / args.n_mc
        alpha_non_adjusted_tot[i] = alpha_non_adjusted_tot[i] / args.n_mc

    if args.save:
        np.savez(
            os.path.join(out_dir, "IHDP_coverage"),
            alpha_tot=alpha_tot,
            alpha_non_adjusted_tot=alpha_non_adjusted_tot,
            th_list=args.th_list,
            gamma=args.gamma_list,
            quant_arr=quant_arr,
            n_mc=args.n_mc,
            dtype=object,
        )
        plot_results.plot_ihdp_coverage(out_dir, out_dir)


def loop_ihdp_loss(df_ihdp, x_name, args, quant_arr, out_dir):
    use_past = [False, False, True]

    rng = default_rng(1057)
    np.random.seed(1057)

    df_train, df_beta = prepare_original_df(df_ihdp, x_name, rng)

    loss_n1_tot, alpha_n1_tot = loop_loss(
        df_train, df_beta, x_name, args, quant_arr, use_past
    )

    if args.save:
        np.savez(
            os.path.join(out_dir, "IHDP_loss"),
            loss_n1_tot=loss_n1_tot,
            alpha_n1_tot=alpha_n1_tot,
            th_list=args.th_list,
            gamma=args.gamma_list,
            dtype=object,
        )
        plot_results.plot_ihdp_loss(out_dir, out_dir)


def loop_loss(df_train, df_beta, x_name, args, quant_arr, use_past):
    global clf
    clf = LogisticRegression(random_state=0).fit(
        df_beta[x_name].to_numpy(), df_beta["d"]
    )

    loss_n1_tot = []
    alpha_n1_tot = []
    for ii, th_i in enumerate(args.th_list):
        loss_n1_temp = []
        alpha_n1_temp = []
        robust_alpha = PolicyEvaluation(
            args.y_max,
            get_p_dx_data,
            partial(get_p_dx_policy, th=th_i, use_past=use_past[ii]),
            x_name,
        )

        for i, gamma_i in enumerate(args.gamma_list):
            __, loss_n1, alpha_n1 = robust_alpha.get_robust_quantiles(
                quant_arr, df_train, df_beta, gamma_i
            )
            loss_n1_temp += [loss_n1]
            alpha_n1_temp += [alpha_n1]

        loss_n1_tot += [loss_n1_temp]
        alpha_n1_tot += [alpha_n1_temp]

    return loss_n1_tot, alpha_n1_tot


if __name__ == "__main__":
    main()
