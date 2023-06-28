"""
Evaluate policy and generate limit curves for the synthetic data in
"Off-Policy Evaluation with Out-of-Sample Guarantees"

Known past policy:
python main_synthetic_loss.py --y_max 2.5 --n_train 1000 --gamma_list 1 --th_list 0, 0.5, 1, 10

Unknown past policy:
python main_synthetic_loss.py --n_train 1000 --gamma_list 1, 2, 3 --y_max 10 --unknown_past_policy
"""
import argparse
import save_utils
import os
import numpy as np

from numpy.random import default_rng
from functools import partial
from generate_data import DataGen, DataGenConfounding
from policy_evaluation import PolicyEvaluation
from sklearn.model_selection import train_test_split
from plot_results import (
    plot_unknown_past_policy_loss,
    plot_known_past_policy_loss,
    plot_first_figure_losses,
)


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=1057, help="random seed (default: %(default)s)"
    )

    parser.add_argument(
        "--unknown_past_policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="unknown past policy",
    )

    parser.add_argument(
        "--th_list",
        nargs="+",
        type=float,
        default=[1],
        help="thresholds of evaluated policy (default: %(default)s)",
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


def main():
    args = create_parser()

    out_dir = ""
    if args.save:
        out_dir = save_utils.save_logging_and_setup(args)

    # Setup
    quant_arr = np.linspace(0.0001, 0.9999, 20)
    ####

    loss_n1_tot = []
    alpha_n1_tot = []
    for case in args.case_list:
        rng = default_rng(args.seed)
        if args.unknown_past_policy:
            data_gen = DataGenConfounding(rng, case)
        else:
            data_gen = DataGen(rng, case)

        df = data_gen.get_data(args.n_train)
        df_train, df_beta = train_test_split(
            df, train_size=args.loss_proportion, random_state=rng.integers(10000)
        )

        loss_n1_temp = []
        alpha_n1_temp = []
        for ii, th in enumerate(args.th_list):
            for i, gamma_i in enumerate(args.gamma_list):
                robust_alpha = PolicyEvaluation(
                    args.y_max,
                    data_gen.get_p_dx_data,
                    partial(data_gen.get_p_dx_policy, th=th),
                )

                __, loss_n1, alpha_n1 = robust_alpha.get_robust_quantiles(
                    quant_arr.tolist(), df_train, df_beta, gamma_i
                )
                loss_n1_temp += [loss_n1]
                alpha_n1_temp += [alpha_n1]

        loss_n1_tot += [loss_n1_temp]
        alpha_n1_tot += [alpha_n1_temp]

    if args.save:
        name = "loss_unknown" if args.unknown_past_policy else "loss_known"
        np.savez(
            os.path.join(out_dir, name),
            loss_n1_tot=loss_n1_tot,
            alpha_n1_tot=alpha_n1_tot,
            gamma=args.gamma_list,
            th_list=args.th_list,
            n=args.n_train,
            cases=args.case_list,
            quant_arr=quant_arr,
            dtype=object,
        )
        if args.unknown_past_policy:
            plot_unknown_past_policy_loss(out_dir, out_dir)
        else:
            plot_known_past_policy_loss(out_dir, out_dir)
            plot_first_figure_losses(out_dir, out_dir)


if __name__ == "__main__":
    main()
