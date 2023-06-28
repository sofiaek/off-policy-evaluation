"""
Evaluate policy and generate limit curves and figures for miscoverage gaps for the nhanes fish data set used in
"Off-Policy Evaluation with Out-of-Sample Guarantees"

* Action A is fish.level = 'low' or 'high'
* Covariates X are gender, age, income, income missing, race, education, 
has smoked (at least 100 cigs), cigarettes smoked last month,
* Outcome Y is log2 of 'o.LBXTHG' (blood mercury, total in ug/L)

For the setup used in the paper:
python main_fish_loss.py --y_max 100 --gamma_list 1, 2, 3
"""

import pyreadr

import numpy as np
import pandas as pd
import argparse

import plot_results
import save_utils
import os

from numpy.random import default_rng
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from policy_evaluation import PolicyEvaluation
from main_ihdp import loop_loss


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="proportion of samples for loss curve (default: %(default)s)",
    )

    parser.add_argument(
        "--case_list",
        default=["All", "Women"],
        nargs="+",
        choices=["All", "Women"],
        help="cases (default: %(default)s)",
    )

    parser.add_argument(
        "--th_list",
        nargs="+",
        type=float,
        default=[0, 1, 1],
        help="thresholds of evaluated policy (default: %(default)s)",
    )

    # add model specific args
    parser = save_utils.add_specific_args(parser)
    parser = PolicyEvaluation.add_model_specific_args(parser)

    args = parser.parse_args()

    return args


def prepare_original_df(df_fish, args, x_name, rng):
    a = df_fish["fish.level"].replace(["low", "high"], [0, 1])
    y = df_fish["o.LBXTHG"]

    print(np.mean(df_fish[df_fish["fish.level"] == "high"]["o.LBXTHG"]))

    df_x = df_fish[x_name].copy()
    df_x = pd.concat([df_x, pd.get_dummies(df_x.race, prefix="race")], axis=1)
    df_x = df_x.drop("race", axis=1)

    df = pd.DataFrame({"d": a, "y": y})
    df = pd.concat([df_x, df], axis=1)

    x_name_cont = ["age", "income"]
    x_name_discrete = [
        "gender",
        "income.missing",
        "education",
        "smoking.ever",
        "smoking.now",
    ]

    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = preprocessing.StandardScaler()

    df.loc[:, x_name_cont] = standard_scaler.fit_transform(df[x_name_cont])
    df.loc[:, x_name_discrete] = min_max_scaler.fit_transform(df[x_name_discrete])

    df_train, df_beta = train_test_split(
        df, test_size=args.test_size, random_state=rng.integers(10000)
    )

    return df_train, df_beta, df_x.columns.to_list()


def prepare_original_df_women(df_fish, args, x_name, rng):
    df_women = df_fish[df_fish["gender"] == 2]  # Female
    return prepare_original_df(df_women, args, x_name, rng)


def main():
    args = create_parser()

    out_dir = ""
    if args.save:
        out_dir = save_utils.save_logging_and_setup(args)

    use_past = [False, False, True]
    quant_list = np.linspace(0.0001, 0.9999, 51).tolist()

    # Read data
    result = pyreadr.read_r("data/nhanes.fish.rda")
    df_fish = result["nhanes.fish"]

    x_name_orig = [
        "gender",
        "age",
        "income",
        "income.missing",
        "race",
        "education",
        "smoking.ever",
        "smoking.now",
    ]

    rng = default_rng(1057)
    name = ""
    df_train, df_beta, x_name = [], [], []
    for case in args.case_list:
        if case == "All":
            df_train, df_beta, x_name = prepare_original_df(
                df_fish, args, x_name_orig, rng
            )
            name = "loss_fish"
        if case == "Women":
            df_train, df_beta, x_name = prepare_original_df_women(
                df_fish, args, x_name_orig, rng
            )
            name = "loss_fish_women"

        loss_n1_tot, alpha_n1_tot = loop_loss(
            df_train, df_beta, x_name, args, quant_list, use_past
        )

        if args.save:
            np.savez(
                os.path.join(out_dir, name),
                loss_n1_tot=loss_n1_tot,
                alpha_n1_tot=alpha_n1_tot,
                th_list=args.th_list,
                gamma=args.gamma_list,
                quant_list=quant_list,
                dtype=object,
            )

    if args.save:
        plot_results.plot_fish_data(out_dir, out_dir)


if __name__ == "__main__":
    main()
