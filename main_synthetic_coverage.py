"""
Evaluate policy and generate figures for coverage and miscoverage gaps for the synthetic data in
"Off-Policy Evaluation with Out-of-Sample Guarantees"

For the same setup used in the paper use

Known past policy:
python main_synthetic_coverage.py --y_max 10 --n_train 250, 500, 1000 --n_test 1000 --th_list 0.5 --gamma_list 1 --case_list case1, case2, case3 --coverage_case known --n_mc 1000

First figure
python main_synthetic_coverage.py --y_max 10 --n_train 1000 --n_test 1000 --th_list 0, 0.5, 1 --gamma_list 1 --case_list case2 --coverage_case first_figure --n_mc 1000

Unknown past policy:
python main_synthetic_coverage.py --n_train 250, 500, 1000 --n_test 1000 --th_list 1 --gamma_list 1, 2, 3 --case_list case1 --coverage_case unknown --n_mc 1000

"""
import argparse
import save_utils
import os
import numpy as np

from numpy.random import default_rng
from functools import partial
from sklearn.model_selection import train_test_split

from generate_data import DataGen, DataGenConfounding
from policy_evaluation import PolicyEvaluation
from plot_results import (
    plot_known_past_policy_coverage,
    plot_unknown_past_policy_coverage,
    plot_first_figure_coverage,
)


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--coverage_case",
        default="known",
        choices=["known", "unknown", "first_figure"],
    )

    parser.add_argument(
        "--th_list",
        nargs="+",
        type=float,
        default=[1],
        help="thresholds of evaluated policy (default: %(default)s)",
    )

    parser.add_argument(
        "--n_mc",
        type=int,
        default=1000,
        help="number of monte carlo simulations (default: %(default)s)",
    )

    parser.add_argument(
        "--loss_proportion",
        type=float,
        default=0.5,
        help="proportion of samples for loss curve (default: %(default)s)",
    )

    parser.add_argument(
        "--case_list",
        default=["case1", "case2", "case3"],
        nargs="+",
        help="case for past policy  (default: %(default)s)",
    )

    # add model specific args
    parser = save_utils.add_specific_args(parser)
    parser = PolicyEvaluation.add_model_specific_args(parser)
    parser = DataGen.add_model_specific_args(parser)

    args = parser.parse_args()

    return args


def test_loss(loss_list, loss_test):
    n = np.zeros(len(loss_list))
    loss_arr = np.array(loss_list)
    for loss_i in loss_test:
        n[loss_i > loss_arr] += 1

    alpha_test = n / len(loss_test)
    return np.array(alpha_test)


def main():
    args = create_parser()

    out_dir = ""
    if args.save:
        out_dir = save_utils.save_logging_and_setup(args)

    quant_arr = np.linspace(0.05, 0.95, 19)
    th = args.th_list[0]
    gamma = args.gamma_list[0]

    alpha_est_tot = []
    alpha_est_cdf_tot = []

    loop_list = []
    if args.coverage_case == "known":
        loop_list = args.case_list
        plot_fcn = plot_known_past_policy_coverage
    elif args.coverage_case == "unknown":
        loop_list = args.gamma_list
        plot_fcn = plot_unknown_past_policy_coverage
    elif args.coverage_case == "first_figure":
        loop_list = args.th_list
        plot_fcn = plot_first_figure_coverage

    for ii, n_i in enumerate(args.n_train):
        alpha_est = [np.zeros(len(quant_arr)) for __ in loop_list]
        alpha_est_cdf = [np.zeros(len(quant_arr)) for __ in loop_list]

        for i_mc in range(args.n_mc):
            if i_mc % 100 == 0:
                print("i={}".format(i_mc))

            for i, loop_i in enumerate(loop_list):
                rng = default_rng(i_mc)

                if args.coverage_case == "known":
                    data_gen = DataGen(rng, loop_i)
                elif args.coverage_case == "unknown":
                    data_gen = DataGenConfounding(rng, args.case_list[0])
                    gamma = loop_i
                elif args.coverage_case == "first_figure":
                    data_gen = DataGen(rng, args.case_list[0])
                    th = loop_i

                df = data_gen.get_data(n_i)
                df_test = data_gen.get_test_data(args.n_test, th)
                df_train, df_beta = train_test_split(
                    df,
                    train_size=args.loss_proportion,
                    random_state=rng.integers(10000),
                )

                robust_alpha = PolicyEvaluation(
                    args.y_max,
                    data_gen.get_p_dx_data,
                    partial(data_gen.get_p_dx_policy, th=th),
                )

                loss_list, __, __ = robust_alpha.get_robust_quantiles(
                    quant_arr.tolist(), df_train, df_beta, gamma
                )
                alpha_est[i] += test_loss(loss_list, df_test["y"].to_numpy())

                loss_list_cdf, p_cum_cdf = robust_alpha.get_quantiles_non_adjusted(
                    quant_arr.tolist(), df
                )
                alpha_est_cdf[i] += test_loss(loss_list_cdf, df_test["y"].to_numpy())

        alpha_est = [i / args.n_mc for i in alpha_est]
        alpha_est_cdf = [i / args.n_mc for i in alpha_est_cdf]

        alpha_est_tot += [alpha_est]
        alpha_est_cdf_tot += [alpha_est_cdf]

    # %%
    if args.save:
        np.savez(
            os.path.join(out_dir, "coverage_{}".format(args.coverage_case)),
            alpha_est_tot=alpha_est_tot,
            alpha_est_cdf_tot=alpha_est_cdf_tot,
            gamma=args.gamma_list,
            n=args.n_train,
            case_list=args.case_list,
            quant_arr=quant_arr,
        )
        plot_fcn(out_dir, out_dir)


if __name__ == "__main__":
    main()
