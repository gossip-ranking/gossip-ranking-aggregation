import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")


def plot_results(
    path, results, list_names, colors, markers, config, metric="score", biases=None
):

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    fontsize = 12
    horizon = config.get("horizon", 0)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10, 
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 10,
        }
    )
    for res, name, color, marker in zip(results, list_names, colors, markers):

        mean_err_scores = res[0]
        std_err_scores = res[1]
        (line,) = ax.plot(
            range(horizon),
            mean_err_scores,
            label=f"{name}",
            marker=marker,
            color=color,
            markevery=range(horizon // 10, horizon, horizon // 10),
            markersize=6,
        )
        line.set_rasterized(True)
        poly = ax.fill_between(
            range(horizon),
            mean_err_scores - std_err_scores,
            mean_err_scores + std_err_scores,
            color=color,
            alpha=0.3,
        )
        poly.set_rasterized(True)
    if metric == "score":
        plt.title("MSE of Scores vs. Iterations", fontsize=fontsize + 2)
    elif metric == "consensus":
        plt.title(
            "Kendall-$\\tau$ Error vs. Iterations",
            fontsize=fontsize + 2,
        )
    plt.xlim(1, horizon)
    plt.ylim(0, None)

    if horizon < 100:
        ticks = np.linspace(0, horizon + 1, 6, dtype=int)
        ax.set_xticklabels(ticks)
    elif horizon <= 1000:
        num_ticks = int(horizon // 100)
        ticks = np.linspace(0, horizon, num_ticks + 1, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([0] + [f"{t/100:.0f}e2" for t in ticks[1:]])
    elif horizon <= 10000:
        num_ticks = int(horizon // 1000)
        ticks = np.linspace(0, horizon, num_ticks + 1, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([0] + [f"{t/1e3:.0f}e3" for t in ticks[1:]])
    else:
        num_ticks = int(horizon // 10000)
        ticks = np.linspace(0, horizon, num_ticks + 1, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([0] + [f"{t/1e4:.0f}e4" for t in ticks[1:]])

    maxi = 1.0
    if metric == "consensus":
        eps = config.get("eps", 0.0)
        maxi = 1.0 + eps * 10
        
        styles = ["--", ":", "-."]
        if biases is not None:
            for bias, name, color, style in zip(biases, list_names, colors, styles):
                plt.axhline(
                    y=bias,
                    color=color,
                    linestyle=style,
                    linewidth=3,
                    markersize=6,
                    label=f"{name} Consensus",
                )
    ticks = np.linspace(0.0, maxi, 5)
    ax.set_yticks(ticks)
    plt.legend()

    plt.savefig(
        f"fig/{path}_{metric}.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def plot_multigraph_results(results, dataset_name: str, legend: str = "score"):
    """
    Plot results for multiple methods and graph types.

    Args:
        results: Dict of results
        legend: Which subplot gets the legend ("score", "consensus", or "both")
    """

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 10,
        }
    )

    markers = {"Borda": "o", "Copeland": "s", "Footrule": "^"}
    linestyles = {"Borda": "-", "Copeland": "--", "Footrule": ":"}

    colors = sns.color_palette("colorblind")

    graph_type_colors = {
        "Complete": colors[0],
        "Watts-Strogatz": colors[1],
        "Geometric": colors[4],
        "2D Grid": colors[2],
        "Cycle": colors[3],
    }

    fig_consensus, ax_consensus = plt.subplots(figsize=(4.5, 4))
    fig_score_bf, ax_score_bf = plt.subplots(figsize=(4.5, 4))
    fig_score_c, ax_score_c = plt.subplots(figsize=(4.5, 4))

    for _, (method_name, method_results) in enumerate(results.items()):
        marker = markers.get(method_name, "o")
        linestyle = linestyles.get(method_name, "-")

        for graph_type, result_data in method_results.items():
            color = graph_type_colors[graph_type]

            mean_err_scores = result_data[0][0]
            mean_err_consensus = result_data[1][0]
            std_err_consensus = result_data[1][1]

            iterations = len(mean_err_scores)

            # Plot score errors on separate plots
            if method_name == "Copeland":
                ax_score_c.plot(
                    mean_err_scores,
                    marker=marker,
                    markevery=range(iterations // 10, iterations, iterations // 10),
                    markersize=6,
                    linestyle=linestyle,
                    color=color,
                )
            else:
                ax_score_bf.plot(
                    mean_err_scores,
                    marker=marker,
                    markevery=range(iterations // 10, iterations, iterations // 10),
                    markersize=6,
                    linestyle=linestyle,
                    color=color,
                )
            # Plot consensus errors
            line = ax_consensus.plot(
                mean_err_consensus,
                marker=marker,
                markevery=range(iterations // 10, iterations, iterations // 10),
                markersize=6,
                linestyle=linestyle,
                color=color,
            )[0]
            line.set_rasterized(True)
            poly = ax_consensus.fill_between(
                range(iterations),
                mean_err_consensus - 2 * std_err_consensus,
                mean_err_consensus + 2 * std_err_consensus,
                alpha=0.4,
                color=color,
            )
            poly.set_rasterized(True)

    ax_score_bf.set_title("MSE of Scores vs. Iterations (Borda)")
    if legend in ["score", "both"]:
        from matplotlib.lines import Line2D

        plotted_graph_types = set()
        for method_name in ["Borda", "Footrule"]:
            if method_name in results:
                plotted_graph_types.update(results[method_name].keys())

        color_handles = [
            Line2D([0], [0], color=graph_type_colors[gt], lw=2, label=gt)
            for gt in sorted(graph_type_colors.keys())
            if gt in plotted_graph_types
        ]

        method_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                marker=markers[method],
                linestyle=linestyles[method],
                markersize=6,
                label=method,
            )
            for method in ["Borda", "Footrule"]
            if method in results
        ]

        first_legend = ax_score_bf.legend(
            handles=color_handles, loc="upper right", title="Graph Type"
        )
        ax_score_bf.add_artist(first_legend)
        ax_score_bf.legend(handles=method_handles, loc="upper left", title="Method")
    ax_score_bf.set_xlim(1, iterations)
    ax_score_bf.set_yscale("log")
    ax_score_bf.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    fig_score_bf.savefig(
        f"fig/{dataset_name}_borda_footrule_score.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )

    ax_score_c.set_title("MSE of Scores vs. Iterations (Copeland)")
    if legend in ["score", "both"]:
        plotted_graph_types_c = set()
        if "Copeland" in results:
            plotted_graph_types_c.update(results["Copeland"].keys())

        color_handles_c = [
            Line2D([0], [0], color=graph_type_colors[gt], lw=2, label=gt)
            for gt in sorted(graph_type_colors.keys())
            if gt in plotted_graph_types_c
        ]
        ax_score_c.legend(
            handles=color_handles_c, loc="upper right", title="Graph Type"
        )
    ax_score_c.set_xlim(1, iterations)
    ax_score_c.set_yscale("log")
    ax_score_c.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    fig_score_c.savefig(
        f"fig/{dataset_name}_copeland_score.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )

    ax_consensus.set_title("Kendall-$\\tau$ Error vs. Iterations")
    ax_consensus.set_xlim(1, iterations)
    ax_consensus.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))

    if legend in ["consensus", "both"]:
        from matplotlib.lines import Line2D

        color_handles = [
            Line2D([0], [0], color=graph_type_colors[gt], lw=2, label=gt)
            for gt in sorted(graph_type_colors.keys())
            if gt in results[list(results.keys())[0]]
        ]

        method_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                marker=markers[method],
                linestyle=linestyles[method],
                markersize=6,
                label=method,
            )
            for method in results.keys()
        ]

        first_legend = ax_consensus.legend(
            handles=color_handles, loc="upper right", title="Graph Type"
        )
        ax_consensus.add_artist(first_legend)
        ax_consensus.legend(handles=method_handles, loc="upper left", title="Method")
    plt.tight_layout()

    method_names = "_vs_".join([m.lower().replace(" ", "_") for m in results.keys()])
    fig_consensus.savefig(
        f"fig/{dataset_name}_{method_names}_consensus.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
