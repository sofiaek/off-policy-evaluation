"""
To generate the plots in "Off-policy evaluation with out-of-sample guarantees".
"""

import os
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sns.set(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("white")
font_size = 20
font_size_legend = 15

colors = ["darkmagenta", "darkgreen", "darkblue"]
linestyles = ["solid", "dashed", "dotted"]
markers = ["s", "*", "."]
names = ["Proposed", "Benchmark"]


def center_subtitles(legend):
    """Centers legend labels with alpha=0"""
    vpackers = legend.findobj(matplotlib.offsetbox.VPacker)
    for vpack in vpackers[:-1]:  # Last vpack will be the title box
        vpack.align = "left"
        for hpack in vpack.get_children():
            draw_area, text_area = hpack.get_children()
            for collection in draw_area.get_children():
                alpha = collection.get_alpha()
                # sizes = collection.get_sizes()
                if alpha == 0:  # or all(sizes == 0):
                    draw_area.set_visible(False)
    return legend


# %% First figure: Losses
def plot_first_figure_losses(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "loss_known.npz"), allow_pickle=True)
    loss_est_tot = npzfile["loss_n1_tot"]
    alpha_n1_tot = npzfile["alpha_n1_tot"]

    plt.figure(figsize=(7, 5))

    plt.step(loss_est_tot[0][0], 1 - alpha_n1_tot[0][0], color=colors[0], where="post")
    plt.step(loss_est_tot[0][1], 1 - alpha_n1_tot[0][1], color=colors[1], where="post")
    plt.step(loss_est_tot[0][2], 1 - alpha_n1_tot[0][2], color=colors[2], where="post")

    plt.xlabel(r"$\ell_{\alpha}$")
    plt.ylabel(r"$1-\alpha$")
    h = []

    colors_temp = [colors[1], colors[2], colors[0]]
    h += [
        plt.plot([], [], label=case_i, color=c)[0]
        for case_i, c, m in zip(
            [r"$\pi$", r"$\pi_1$", r"$\pi_0$"], colors_temp, markers
        )
    ]
    plt.legend(handles=h, loc="lower right", fontsize=font_size_legend)
    plt.grid()
    plt.xlim([-0.35, 1.5])
    plt.ylim([0, 1])

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if save_fig:
        plt.savefig(os.path.join(out_dir, "limit_curve_first.pdf"), bbox_inches="tight")


# %% First figure: Coverage
def plot_first_figure_coverage(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "coverage_first_figure.npz"))
    alpha_est_tot = npzfile["alpha_est_tot"]
    quant_arr = npzfile["quant_arr"]

    plt.figure(figsize=(7, 5))
    sns.lineplot(x=[0, 1], y=[0, 1], color="k", ls="solid", zorder=0)

    plt.scatter(
        1 - quant_arr,
        1 - alpha_est_tot[0][1],
        marker=markers[0],
        color=colors[0],
        zorder=4,
    )
    plt.scatter(
        1 - quant_arr,
        1 - alpha_est_tot[0][2],
        marker=markers[1],
        color=colors[1],
        zorder=6,
    )
    plt.scatter(
        1 - quant_arr,
        1 - alpha_est_tot[0][0],
        marker=markers[2],
        color=colors[2],
        zorder=8,
    )

    h = []
    h += [
        plt.scatter([], [], label=case_i, marker=m, color=c)
        for case_i, c, m in zip([r"$\pi$", r"$\pi_1$", r"$\pi_0$"], colors, markers)
    ]
    leg = plt.legend(handles=h, loc="lower right", fontsize=font_size_legend)

    plt.xlabel(r"Target coverage")
    plt.ylabel(r"Actual coverage")

    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    if save_fig:
        plt.savefig(os.path.join(out_dir, "coverage_first.pdf"), bbox_inches="tight")


# %% Known past policy: Compare policy
def plot_known_past_policy_loss(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "loss_known.npz"), allow_pickle=True)

    tests = npzfile["cases"]
    th_list = npzfile["th_list"]
    loss_est_tot = npzfile["loss_n1_tot"]
    alpha_n1_tot = npzfile["alpha_n1_tot"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    label_list = [r"$\tau={}$".format(x) for x in th_list]

    for i, test_i in enumerate(tests):
        for y, alpha, c in zip(loss_est_tot[i], alpha_n1_tot[i], colors):
            axes[i].step(y, 1 - alpha, color=c, linestyle=linestyles[0], where="post")

        axes[i].step(
            loss_est_tot[i][-1],
            1 - alpha_n1_tot[i][-1],
            "k",
            linestyle=linestyles[1],
            where="post",
        )

        if i == 1:
            axes[i].set_xlabel(r"$\ell_{\alpha}$", fontsize=font_size)
        if i == 0:
            axes[i].set_ylabel(r"$1-\alpha$", fontsize=font_size)

        axes[i].set_ylim([0.0, 1.0])
        axes[i].set_xlim([-0.5, 1.6])
        axes[i].set_title(r"$p_{}(A|X)$".format(i + 1))
        axes[i].grid()

        if i == 2:
            h = []
            h += [plt.plot([], [], alpha=0, label=r"$p_{\pi}(A|X)$")[0]]
            h += [
                axes[i].plot([], [], label=case_i, color=c)[0]
                for case_i, c in zip(label_list, colors)
            ]
            h += [axes[i].plot([], [], label=r"$p_{\pi} = p$", color="k", ls="--")[0]]
            leg = plt.legend(fontsize=font_size_legend, loc=(1.03, 0))
            center_subtitles(leg)

        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    if save_fig:
        plt.savefig(os.path.join(out_dir, "limit_curve_known.pdf"), bbox_inches="tight")


# %% Known past policy: Coverage
def plot_known_past_policy_coverage(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "coverage_known.npz"))
    alpha_est_tot = npzfile["alpha_est_tot"]
    alpha_est_cdf_tot = npzfile["alpha_est_cdf_tot"]
    quant_arr = npzfile["quant_arr"]
    n = npzfile["n"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ii, n_i in enumerate(n):
        if n_i != 2000:
            sns.lineplot(ax=axes[ii], x=[0, 1], y=[0, 0], color="k", ls=linestyles[2])
            for y, c in zip(alpha_est_tot[ii], colors):
                sns.lineplot(
                    ax=axes[ii], x=quant_arr, y=quant_arr - y, color=c, ls=linestyles[0]
                )

            for y, c in zip(alpha_est_cdf_tot[ii], colors):
                sns.lineplot(
                    ax=axes[ii],
                    x=quant_arr,
                    y=quant_arr - y,
                    color=c,
                    linestyle=linestyles[1],
                )

            axes[ii].set_title(r"$n = {}$".format(n_i))

            if n_i == n[1]:
                axes[ii].set_xlabel(r"Target $\alpha$", fontsize=font_size)
            if n_i == n[0]:
                axes[ii].set_ylabel(r"Miscoverage gap", fontsize=font_size)

            if n_i == n[-1]:
                h = []
                h += [plt.plot([], [], alpha=0, label=r"Past policy")[0]]
                h += [
                    axes[ii].scatter([], [], label=case_i, color=c)
                    for case_i, c in zip(
                        [r"$p_1(A|X)$", r"$p_2(A|X)$", r"$p_3(A|X)$"], colors
                    )
                ]
                h += [plt.plot([], [], alpha=0, label=r"Type")[0]]
                h += [
                    axes[ii].plot([], [], label=name_i, color="k", ls=ls_i)[0]
                    for ls_i, name_i in zip(linestyles, names)
                ]
                leg = plt.legend(handles=h, loc=(1.03, 0), fontsize=font_size_legend)

                center_subtitles(leg)

            axes[ii].set_xlim([0, 1])
            axes[ii].set_ylim([-0.03, 0.15])

            axes[ii].grid()

    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "miscoverage_gap_known.pdf"), bbox_inches="tight"
        )


# %%Unknown past policy: Effect of gamma
def plot_unknown_past_policy_loss(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "loss_unknown.npz"), allow_pickle=True)
    tests = npzfile["cases"]
    gamma = npzfile["gamma"]
    th_list = npzfile["th_list"]
    loss_est_tot = npzfile["loss_n1_tot"]
    alpha_n1_tot = npzfile["alpha_n1_tot"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for i, test_i in enumerate(tests):
        for y, alpha, c in zip(loss_est_tot[i], alpha_n1_tot[i], colors):
            axes[i].step(y, 1 - alpha, linestyle=linestyles[0], color=c, where="post")

        if i == 1:
            axes[i].set_xlabel(r"$\ell_{\alpha}$", fontsize=font_size)
        if i == 0:
            axes[i].set_ylabel(r"$1-\alpha$", fontsize=font_size)

        axes[i].set_ylim([0.0, 1.0])
        axes[i].set_xlim([-0.5, 1.3])
        axes[i].set_title(r"$p_{}(A|X)$".format(i + 1))
        axes[i].grid()
        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        if np.all(test_i == tests[-1]):
            h = []
            h += [plt.plot([], [], alpha=0, label=r"Gamma $\Gamma$")[0]]
            h += [
                axes[i].scatter([], [], label=gamma_i, color=c_i)
                for c_i, gamma_i in zip(colors, gamma)
            ]
            leg = plt.legend(handles=h, loc=(1.03, 0), fontsize=font_size_legend)

            center_subtitles(leg)

    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "limit_curve_unknown.pdf"), bbox_inches="tight"
        )


# %% Unknown past policy: Coverage
def plot_unknown_past_policy_coverage(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "coverage_unknown.npz"))
    alpha_est_tot = npzfile["alpha_est_tot"]
    alpha_est_cdf_tot = npzfile["alpha_est_cdf_tot"]
    quant_arr = npzfile["quant_arr"]
    gamma = npzfile["gamma"]

    n = npzfile["n"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ii, n_i in enumerate(n):
        sns.lineplot(ax=axes[ii], x=[0, 1], y=[0, 0], color="k", ls=linestyles[2])
        for y, c in zip(alpha_est_tot[ii], colors):
            sns.lineplot(
                ax=axes[ii], x=quant_arr, y=quant_arr - y, color=c, ls=linestyles[0]
            )

        sns.lineplot(
            ax=axes[ii],
            x=quant_arr,
            y=quant_arr - alpha_est_cdf_tot[ii][0],
            color="k",
            linestyle=linestyles[1],
        )

        axes[ii].set_title(r"$n = {}$".format(n_i))

        if n_i == n[1]:
            axes[ii].set_xlabel(r"Target $\alpha$", fontsize=font_size)
        if n_i == n[0]:
            axes[ii].set_ylabel(r"Miscoverage gap", fontsize=font_size)

        if n_i == n[-1]:
            h = []
            h += [plt.plot([], [], alpha=0, label=r"Gamma $\Gamma$")[0]]
            h += [
                axes[ii].scatter([], [], label=gamma_i, color=c)
                for gamma_i, c in zip(gamma, colors)
            ]
            h += [plt.plot([], [], alpha=0, label="Type")[0]]
            h += [
                axes[ii].plot([], [], label=name_i, color="k", ls=ls_i)[0]
                for ls_i, name_i in zip(linestyles, names)
            ]
            leg = plt.legend(handles=h, loc=(1.03, 0), fontsize=font_size_legend)

            center_subtitles(leg)

        axes[ii].set_xlim([0, 1])
        axes[ii].set_ylim([-0.25, 0.25])
        axes[ii].grid()

    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "miscoverage_gap_unknown.pdf"), bbox_inches="tight"
        )


# %% Real data
def plot_fish_data(out_dir, load_dir, save_fig=True):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    files = ["loss_fish.npz", "loss_fish_women.npz"]
    title = ["All", "Women"]
    name_policy = [r"$\pi_0$", r"$\pi_1$", r"$p_\pi = p$"]

    for i in range(2):
        npzfile = np.load(os.path.join(load_dir, files[i]), allow_pickle=True)
        loss_est_tot = npzfile["loss_n1_tot"]
        alpha_n1_tot = npzfile["alpha_n1_tot"]
        gamma = npzfile["gamma"]

        for y, alpha, c in zip(loss_est_tot[0], alpha_n1_tot[0], colors):
            axes[i].step(y, 1 - alpha, linestyle=linestyles[0], color=c, where="post")

        for y, alpha, c in zip(loss_est_tot[1], alpha_n1_tot[1], colors):
            axes[i].step(y, 1 - alpha, linestyle=linestyles[1], color=c, where="post")

        axes[i].step(
            loss_est_tot[2][0],
            1 - alpha_n1_tot[2][0],
            linestyle=linestyles[2],
            color="k",
            where="post",
        )

        axes[i].set_ylim([-0.01, 1.01])
        axes[i].set_xlim([0, 15])
        axes[i].grid()
        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        axes[i].set_title(title[i])
        axes[i].set_xlabel(r"$\ell_{\alpha} \; $[$\mu$g/L]", fontsize=font_size)
        if i == 0:
            axes[0].set_ylabel(r"$1-\alpha$", fontsize=font_size)

        if i == 1:
            h = []
            h += [
                axes[1].plot([], [], label=name_i, color="k", ls=ls_i)[0]
                for ls_i, name_i in zip(linestyles, name_policy)
            ]
            h += [plt.plot([], [], alpha=0, label=r"Gamma $\Gamma$")[0]]
            h += [
                axes[1].scatter([], [], label=gamma_i, color=c_i)
                for c_i, gamma_i in zip(colors, gamma)
            ]
            leg = plt.legend(fontsize=font_size_legend, loc=(1.03, 0))
            center_subtitles(leg)

    if save_fig:
        plt.savefig(os.path.join(out_dir, "limit_curve_fish.pdf"), bbox_inches="tight")


# %% Appendix ###################################
# IHDP data: loss
def plot_ihdp_loss(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "IHDP_loss.npz"), allow_pickle=True)
    loss_n1_tot = npzfile["loss_n1_tot"]
    alpha_n1_tot = npzfile["alpha_n1_tot"]
    gamma = npzfile["gamma"]

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))

    for y, alpha, c in zip(loss_n1_tot[0], alpha_n1_tot[0], colors):
        axes.step(y, 1 - alpha, linestyle=linestyles[0], color=c, where="post")

    for y, alpha, c in zip(loss_n1_tot[1], alpha_n1_tot[1], colors):
        axes.step(y, 1 - alpha, linestyle=linestyles[1], color=c, where="post")

    axes.step(
        loss_n1_tot[2][0],
        1 - alpha_n1_tot[2][0],
        linestyle=linestyles[2],
        color="k",
        where="post",
    )

    axes.set_ylim([-0.01, 1.01])
    axes.set_xlim([-25, 15])

    axes.grid()
    axes.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    names = [r"$\pi_0$", r"$\pi_1$", r"$p_\pi = p$"]

    axes.set_xlabel(r"$\ell_{\alpha}$", fontsize=font_size)
    axes.set_ylabel(r"$1-\alpha$", fontsize=font_size)

    h = []
    h += [
        axes.plot([], [], label=name_i, color="k", ls=ls_i)[0]
        for ls_i, name_i in zip(linestyles, names)
    ]
    h += [plt.plot([], [], alpha=0, label=r"Gamma $\Gamma$")[0]]
    h += [
        axes.scatter([], [], label=gamma_i, color=c_i)
        for c_i, gamma_i in zip(colors, gamma)
    ]
    leg = plt.legend(fontsize=font_size_legend, loc=(1.03, 0))
    center_subtitles(leg)

    if save_fig:
        plt.savefig(os.path.join(out_dir, "limit_curve_ihdp.pdf"), bbox_inches="tight")


# %% IHDP data: coverage
def plot_ihdp_coverage(out_dir, load_dir, save_fig=True):
    npzfile = np.load(os.path.join(load_dir, "IHDP_coverage.npz"), allow_pickle=True)
    alpha_tot = npzfile["alpha_tot"]
    alpha_non_adjusted_tot = npzfile["alpha_non_adjusted_tot"]
    quant_arr = npzfile["quant_arr"]
    gamma = npzfile["gamma"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(2):
        sns.lineplot(ax=axes[i], x=[0, 1], y=[0, 0], color="k", ls=linestyles[2])
        for alpha, c in zip(alpha_tot[i], colors):
            sns.lineplot(
                ax=axes[i], x=quant_arr, y=quant_arr - alpha, color=c, ls=linestyles[0]
            )

        sns.lineplot(
            ax=axes[i],
            x=quant_arr,
            y=quant_arr - alpha_non_adjusted_tot[i],
            color="k",
            ls=linestyles[1],
        )

        axes[i].set_title(r"$\pi_{}$".format(i))
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([-0.1, 0.3])
        axes[i].grid()
        axes[i].set_xlabel(r"Target $\alpha$", fontsize=font_size)
        if i == 0:
            axes[i].set_ylabel(r"Miscoverage gap", fontsize=font_size)

        if i == 1:
            h = []
            h += [plt.plot([], [], alpha=0, label=r"Gamma $\Gamma$")[0]]
            h += [
                axes[i].scatter([], [], label=gamma_i, color=c_i)
                for c_i, gamma_i in zip(colors, gamma)
            ]

            h += [plt.plot([], [], alpha=0, label="Type")[0]]
            h += [
                axes[i].plot([], [], label=name_i, color="k", ls=ls_i)[0]
                for ls_i, name_i in zip(linestyles, names)
            ]

            leg = plt.legend(fontsize=font_size_legend, loc=(1.03, 0))
            center_subtitles(leg)
    if save_fig:
        plt.savefig(
            os.path.join(out_dir, "miscoverage_gap_ihdp.pdf"), bbox_inches="tight"
        )
