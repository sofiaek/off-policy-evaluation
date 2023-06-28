"""
Helper file to save and log results.

Note that the logging is not used.
"""

import argparse
import os
import datetime
import json
import subprocess
import logging
import sys


def save_logging_and_setup(args):
    # create output directory
    out_dir = set_out_dir(args)

    # save experiment details
    save_logging(out_dir, args)
    save_hyperparams(out_dir, args)
    save_git_hash(out_dir)

    return out_dir


def set_out_dir(args):
    folder = os.path.join(
        os.getcwd(),
        "out",
        args.exp_folder_name,
        str(datetime.datetime.now())
        .replace(":", "_")
        .replace(" ", "__")
        .replace(".", "__"),
    )
    # Create output folder if needed
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    return folder


def get_full_path(folder, name, prefix=""):
    return os.path.join(folder, (prefix + "_" + name) if prefix else name)


def save_hyperparams(folder, args, prefix=""):
    with open(get_full_path(folder, "config.json", prefix), "w") as handle:
        json.dump(vars(args), handle, indent="\t")


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_diff() -> str:
    diff = (
        subprocess.check_output(["git", "diff", "HEAD"])
        .decode(sys.stdout.encoding)
        .strip()
    )
    if diff:
        message = "It is uncommited changes in the code."
        logging.warning(message)
        print("WARNING: " + message)
    return diff


def save_git_hash(folder, prefix=""):
    with open(
        get_full_path(folder, "git.txt", prefix), "w", encoding="utf-8"
    ) as handle:
        handle.write(get_git_revision_hash() + "\n")
        handle.write(get_git_diff())


def save_logging(folder, args, prefix=""):
    handlers = [
        logging.FileHandler(get_full_path(folder, "debug.log", prefix)),
    ]
    if args.print_logs:
        handlers += [logging.StreamHandler()]
    if args.debug:
        logging.basicConfig(handlers=handlers, encoding="utf-8", level=logging.DEBUG)
    else:
        logging.basicConfig(handlers=handlers, encoding="utf-8", level=logging.INFO)

    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)


def save_path_to_data(folder, path_data, prefix=""):
    with open(
        get_full_path(folder, "data_path.txt", prefix), "w", encoding="utf-8"
    ) as handle:
        handle.write(path_data)


def add_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("Save utils")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save results and logging from experiment",
    )

    parser.add_argument(
        "--exp_folder_name",
        default="exp",
        help="name of experiment folder (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="logging in debug mode",
    )
    parser.add_argument(
        "--print_logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also print logging",
    )
    return parent_parser
