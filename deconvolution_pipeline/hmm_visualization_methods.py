import itertools
import numpy as np
import pandas as pd
import logomaker
import pomegranate
from logomaker.src import colors
from logomaker.src.Glyph import Glyph
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
from scipy.stats import norm
import pyvis
import seaborn as sns
import colorcet as cc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap
from training_parameters import *
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


def _sanitize_filename(filename: str) -> str:
    """
    Replace characters that are unsafe or treated as path separators
    in filenames (/, \, *, :, ?, ", <, >, |) with underscores.
    """
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        filename = filename.replace(ch, "_")
    return filename


def save_figure_to_svg(fig, dir, filename):
    os.makedirs(dir, exist_ok=True)
    fig.savefig(os.path.join(dir, filename), format="svg", bbox_inches='tight', transparent=False)


def save_figure_to_png(fig, dir, filename, dpi=200):
    os.makedirs(dir, exist_ok=True)
    fig.savefig(os.path.join(dir, filename), format="png", dpi=dpi, bbox_inches='tight', transparent=False)


def save_figure(fig, dir, filename_without_ext, save_svg=True, save_png=True, dpi=200):
    os.makedirs(dir, exist_ok=True)
    if save_svg:
        save_figure_to_svg(fig, dir, f"{filename_without_ext}.svg")
    if save_png:
        save_figure_to_png(fig, dir, f"{filename_without_ext}.png", dpi=dpi)


# Функция берёт probability distribution по аминокислотам и превращает его в матрицу информационного содержания для одной позиции motif/logo.
def probs_to_info_matrix(distribution):
    amino_acids = list(distribution.parameters[0].keys())
    dummy_df = pd.DataFrame(columns=sorted(amino_acids), index=[0], dtype=float)
    for i, (acid, prob) in enumerate(distribution.parameters[0].items()):
        dummy_df.loc[0, acid] = prob
    dummy_df.index.name = "pos"
    info_matrix = logomaker.transform_matrix(dummy_df, from_type='probability', to_type='information')
    return info_matrix


def plot_aa_distr(model, distr, ax, horizontal, matrix_type, set_lim=False):
    amino_acids = list(model.keymap[0].keys())
    amino_acids = list(sorted(amino_acids))
    color_dict = colors.get_color_dict(None, amino_acids)
    if horizontal:
        glyph_list = []
        for i, (acid, prob) in enumerate(distr.items()):
            # print(i, acid, prob)
            floor = 0
            ceiling = floor + abs(prob)
            this_color = color_dict[acid]
            flip = False
            glyph = Glyph(i, acid,
                          ax=ax,
                          floor=floor,
                          width=1,
                          ceiling=ceiling,
                          color=this_color,
                          flip=flip,
                          zorder=0,
                          # font_name='Arial Rounded MT Bold',
                          alpha=1.0,
                          vpad=0.0)
            glyph_list.append(glyph)
            xmin = min([g.p - .5 * g.width for g in glyph_list])
            xmax = max([g.p + .5 * g.width for g in glyph_list])
            ax.set_xlim([xmin, xmax])
            # set ylims
            ymin = min([g.floor for g in glyph_list])
            # ymax = max([g.ceiling for g in glyph_list])
            ymax = 1
            ax.set_ylim([ymin, ymax])
            plt.xticks(np.arange(0, 20, 1.0), labels=sorted(amino_acids))

    else:
        dummy_df = pd.DataFrame(columns=sorted(amino_acids), index=[0], dtype=float)
        for i, (acid, prob) in enumerate(distr.items()):
            dummy_df.loc[0, acid] = prob
        dummy_df.index.name = "pos"
        # return dummy_df
        if matrix_type == 'information':
            info_matrix = logomaker.transform_matrix(dummy_df, from_type='probability', to_type='information')
        else:
            info_matrix = dummy_df
        ww_logo = logomaker.Logo(info_matrix, ax=ax
                                 # font_name='Arial Rounded MT Bold',
                                 # stack_order='small_on_top'
                                 )

        max_info = np.max(info_matrix.values)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        if max_info < 0.1:
            ax.ticklabel_format(useOffset=False)
        if set_lim:
            ax.set_ylim([0, 1.7])


def plot_pos_distr(distr, ax):
    sns.barplot(x=list(distr.keys()), y=list(distr.values()), ax=ax)


def plot_distributions_for_states(model, TARGET_PATH_TO_RESULTS, horizontal=True, discrete=True, initial_params=None,
                                  matrix_type='information'):
    PATH_TO_SAVE_PICTURES = TARGET_PATH_TO_RESULTS + "/" + model.name + "/"
    if not os.path.exists(PATH_TO_SAVE_PICTURES):
        os.makedirs(PATH_TO_SAVE_PICTURES)
    for state in model.states:
        if not (state == model.start or state == model.end):
            if discrete:
                if isinstance(state.distribution, pomegranate.IndependentComponentsDistribution):  # posiotion component case
                    list_of_dists = state.distribution.parameters[0]
                    fig, axes = plt.subplots(ncols=1, nrows=len(list_of_dists), figsize=(8, 14), height_ratios=[13, 10])
                    aa_dist = list_of_dists[0].parameters[0]
                    aa_ax = axes[0]
                    plot_aa_distr(model=model, distr=aa_dist, ax=aa_ax, horizontal=horizontal, matrix_type=matrix_type, set_lim=True)
                    pos_dist = list_of_dists[1].parameters[0]
                    pos_ax = axes[1]
                    plot_pos_distr(pos_dist, ax=pos_ax)
                    save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}.svg")
                    plt.close(fig)
                else:  # simple discrete distribution case

                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 9))
                    distr = state.distribution.parameters[0]
                    plot_aa_distr(model, distr, ax, horizontal, matrix_type)

                    save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}.svg")
                    plt.close(fig)



            elif type(state.distribution) is pomegranate.IndependentComponentsDistribution:
                list_of_dists = state.distribution.parameters[0]

                for i, distr in enumerate(list_of_dists):  # just list of normals

                    distr_params = distr.parameters
                    mean_value = distr_params[0]
                    std_value = distr_params[1]

                    mean_for_picture = mean_value
                    std_for_picture = std_value
                    if initial_params is not None:
                        assert type(initial_params) == list
                        initial_distr = initial_params[i]
                        initial_mean = initial_distr['mean']
                        initial_std = initial_distr['std']
                        mean_for_picture = initial_mean
                        std_for_picture = max(initial_std, std_for_picture) + abs(mean_value - initial_mean)
                    x = np.linspace(mean_for_picture - 4 * std_for_picture, mean_for_picture + 4 * std_for_picture, 1000)
                    y1 = norm.pdf(x, loc=mean_value, scale=std_value)
                    axes[i].plot(x, y1, 'b', lw=2, label=f'pdf')
                    if initial_params is not None:
                        y2 = norm.pdf(x, loc=initial_mean, scale=initial_std)
                        axes[i].plot(x, y2, 'r', lw=2, label=f'pdf_initial')
                    axes[i].text(0.0, 0.75, f'{np.round(mean_value, 3)}', fontsize=35, transform=axes[i].transAxes)
                    axes[i].text(0.7, 0.75, f'{np.round(std_value, 3)}', fontsize=35, transform=axes[i].transAxes)
                save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}")
                plt.close(fig)
            else:
                # normal distr here
                distr_params = state.distribution.parameters
                mean_value = distr_params[0]
                std_value = distr_params[1]
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 3))
                x = np.linspace(mean_value - 4 * std_value, mean_value + 4 * std_value, 1000)
                y = norm.pdf(x, loc=mean_value, scale=std_value)
                ax.plot(x, y, 'r', lw=2, label='pdf')
                save_figure_to_svg(fig, dir=PATH_TO_SAVE_PICTURES, filename=f"{state.name}")
                plt.close(fig)


def plot_motifs_for_paths(model, TARGET_PATH_TO_RESULTS, paths_df, top=None):
    PATH_TO_SAVE_PICTURES = TARGET_PATH_TO_RESULTS + "/" + model.name + "/paths/"
    if not os.path.exists(PATH_TO_SAVE_PICTURES):
        os.makedirs(PATH_TO_SAVE_PICTURES)
    result_df = pd.DataFrame()
    state_name_to_index = dict()
    for i, state in enumerate(model.states):
        state_name_to_index[state.name] = i
    states = model.states
    if top:
        paths_df = paths_df.sort_values("prob", ascending=False)[:top]
    for index, row in paths_df.iterrows():
        probs_url = f"path_{index}_probs.png"
        info_url = f"path_{index}_info.png"
        result_dict = {
            "id": index,
            "info_picture": '<img src="' + info_url + '" width="320" >',
            "probs_picture": '<img src="' + probs_url + '" width="320" >',
            "motif_url": info_url,
            "prob": row['prob']
        }

        df = pd.DataFrame()
        for position, state_name in enumerate(row['path'], start=-1):
            if state_name not in [model.start.name, model.end.name]:
                params = states[state_name_to_index[state_name]].distribution.parameters[0]
                df = df.append(params, ignore_index=True)
                result_dict[f'pos_{position}'] = state_name

        info_matrix = logomaker.transform_matrix(df, from_type='probability', to_type='information')
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
        info_logo = logomaker.Logo(info_matrix, ax=ax)
        save_figure_to_png(fig, dir=PATH_TO_SAVE_PICTURES, filename=info_url)
        plt.close(fig)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
        probs_logo = logomaker.Logo(df, ax=ax)
        save_figure_to_png(fig, dir=PATH_TO_SAVE_PICTURES, filename=probs_url)
        plt.close(fig)

        result_df = result_df.append(result_dict, ignore_index=True)
    result_df.to_html(f'{PATH_TO_SAVE_PICTURES}/paths.html', escape=False)

    return result_df


def make_levels(model, precision=4):
    source_gr = model.graph
    name_to_level = defaultdict(int)
    name_to_size = dict()
    passed_nodes = set()
    nodes_stack = []
    nodes_stack.append(model.start)
    name_to_level[model.start.name] = 1

    while len(nodes_stack) > 0:
        cur_node = nodes_stack.pop()
        passed_nodes.add(cur_node)
        for node_to in source_gr.successors(cur_node):
            data = source_gr.get_edge_data(cur_node, node_to, default=None)
            log_probability = data['probability']
            p = np.exp(log_probability)
            p = round(p, precision)
            if node_to not in passed_nodes:
                if p != 0:
                    cur_level = name_to_level[cur_node.name] + 1
                    if not node_to.name in name_to_level:
                        name_to_level[node_to.name] = cur_level
                        # print(cur_node.name, "->", node_to.name, p,  cur_level )
                    nodes_stack.append(node_to)

    # repeat for all subgraphs
    not_seen_states = []
    for state in model.states:
        if state not in passed_nodes:
            not_seen_states.append(state)
    state_to_in_count = dict()
    for state in not_seen_states:
        predecessors = source_gr.predecessors(state)
        selected_predecessors = []
        for node_from in predecessors:
            data = source_gr.get_edge_data(node_from, state, default=None)
            log_probability = data['probability']
            p = np.exp(log_probability)
            p = round(p, precision)
            if p != 0:
                selected_predecessors.append(node_from)
        state_to_in_count[state.name] = len(selected_predecessors)
    min_in_count_states = [state for state in not_seen_states if
                           state_to_in_count[state.name] == min(state_to_in_count.values())]
    for state in min_in_count_states:
        if state not in passed_nodes:
            print("new starting point")
            nodes_stack.append(state)
            name_to_level[state.name] = 1
            while len(nodes_stack) > 0:
                cur_node = nodes_stack.pop()
                passed_nodes.add(cur_node)
                for node_to in source_gr.successors(cur_node):
                    data = source_gr.get_edge_data(cur_node, node_to, default=None)
                    log_probability = data['probability']
                    p = np.exp(log_probability)
                    p = round(p, precision)
                    if node_to not in passed_nodes:
                        if p != 0:
                            cur_level = name_to_level[cur_node.name] + 1
                            if not node_to.name in name_to_level:
                                name_to_level[node_to.name] = cur_level
                                # print(cur_node.name, "->", node_to.name, p,  cur_level )
                            nodes_stack.append(node_to)
    # Add rest with

    return name_to_level


def make_pyviz_graph(model, TARGET_PATH_TO_RESULTS, precision=4, prefefined_hierarchical_layout=True):
    graph, name_to_level = convert_graph_to_good_format(model, logo_pictures_path=f"../{_sanitize_filename(model.name)}/", precision=precision)

    nt = Network('500px', '1600', directed=True)
    if pyvis._version.__version__ > '0.1.9':
        # Bug in pyvis
        nt.from_nx(graph, show_edge_weights=True)
    else:
        nt.from_nx(graph)
        # nt.show_buttons()
        # nt.show_buttons(filter_=['physics'])
    if prefefined_hierarchical_layout:
        nt.set_options(
            """
            var options = {
                "configure": {
                "enabled": true
            },
                "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR"
                }
            },
            "physics": {
              "hierarchicalRepulsion": {
                "centralGravity": 0
              },
              "minVelocity": 0.75,
              "solver": "hierarchicalRepulsion"
        }
    }
    """)
    else:
        nt.show_buttons()

    path_to_save_html = f"{TARGET_PATH_TO_RESULTS}/result_html_files"
    if not os.path.exists(path_to_save_html):
        os.makedirs(path_to_save_html)

    nt.save_graph(f"{path_to_save_html}/{model.name}-{precision}.html")
    return nt


def convert_graph_to_good_format(model, logo_pictures_path, use_logos=True, precision=4, edge_value="probability"):
    assert edge_value in ["probability", "log_probability"]
    name_to_level = make_levels(model, precision)  # print(name_to_level)
    source_gr = model.graph
    gr = nx.DiGraph()

    # calculate information for sizes
    per_state_information = dict()
    per_level_information = defaultdict(float)
    for state in model.states:
        if state not in [model.start, model.end]:
            if type(state.distribution) == pomegranate.DiscreteDistribution \
                    or type(state.distribution) == pomegranate.DiscreteDistributionAnchor \
                    or type(state.distribution) == pomegranate.DiscreteDistributionCycle:
                info_matrix = probs_to_info_matrix(state.distribution)
                per_state_information[state.name] = sum(info_matrix.loc[0])
                per_level_information[name_to_level[state.name]] += per_state_information[state.name]
            elif type(state.distribution) is pomegranate.NormalDistribution:
                per_state_information[state.name] = state.distribution.parameters[1]
                per_level_information[name_to_level[state.name]] += per_state_information[state.name]
            elif type(state.distribution) is pomegranate.IndependentComponentsDistribution:
                per_state_information[state.name] = 1
                per_level_information[name_to_level[state.name]] += per_state_information[state.name]

    # Add all states from model's graph (logo pictures should be prepared)
    for state in model.states:
        if state == model.start or state == model.end or not use_logos:  # start and end states without picture
            gr.add_node(state.name,
                        level=name_to_level[state.name],
                        size=35)
        else:
            # info_portion = per_state_information[state.name] / per_level_information[name_to_level[state.name]]
            size_percentage = per_state_information[state.name] / max(per_state_information.values())
            size_percentage = max(size_percentage, 3 / 35)
            # print(f"State", state.name,"Size", size_percentage)
            gr.add_node(state.name,
                        level=name_to_level[state.name],
                        size=size_percentage * 35 * 1.8,
                        shape='image',
                        # image=f"file:///{logo_pictures_path}{state.name}.svg")
                        image=f"{logo_pictures_path}{state.name}.svg")

    # Add all non-zero transitions
    for edge in source_gr.edges:
        data = source_gr.get_edge_data(edge[0], edge[1], default=None)
        log_probability = data['probability']
        p = np.exp(log_probability)
        p = round(p, precision)
        if p != 0:
            if edge_value == "probability":
                gr.add_edge(edge[0].name, edge[1].name, label=p)
            else:
                gr.add_edge(edge[0].name, edge[1].name, label=log_probability)

    return gr, name_to_level


def make_logos_for_train_sequences(target_length, per_allele_per_kfold_per_length_training_data,
                                   TARGET_PATH_TO_RESULTS):
    per_allele_per_split_prob_matrix = dict()
    for allele_name in per_allele_per_kfold_per_length_training_data.keys():
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
        per_allele_per_split_prob_matrix[allele_name] = dict()
        for split_num, per_length_data in per_allele_per_kfold_per_length_training_data[allele_name].items():
            model_name = f"{target_length}-mer_model-{target_allele_name}-{split_num}"
            binders_array = per_length_data[target_length]
            matrix = logomaker.alignment_to_matrix(binders_array)
            info_matrix = logomaker.transform_matrix(matrix, from_type='counts', to_type='information')
            prob_matrix = logomaker.transform_matrix(matrix, from_type='counts', to_type='probability')
            ww_logo = logomaker.Logo(info_matrix,
                                     # font_name='Arial Rounded MT Bold',
                                     )
            ww_logo.ax.set_title(f"Allele {allele_name}, length {target_length}, split {split_num}")
            PATH_TO_SAVE_PICTURES = TARGET_PATH_TO_RESULTS + f"{model_name}/"
            save_figure_to_svg(fig=ww_logo.fig, dir=PATH_TO_SAVE_PICTURES, filename=f"logo_for_sequences-{target_length}.svg")


def plot_all_scores_as_lines(per_allele_data_df, per_name_models):
    for allele_name in per_allele_data_df.keys():
        df = per_allele_data_df[allele_name].reset_index(drop=True)
        palette = sns.color_palette(cc.glasbey, n_colors=len(per_name_models))
        palette_lines = sns.color_palette(cc.linear_blue_95_50_c20, n_colors=len(per_name_models))

        target_df = df.melt(id_vars=['peptide'], value_vars=[f'model_{i}' for i in range(len(per_name_models))],
                            value_name='score', var_name='model', ignore_index=False)
        #  target_df = target_df.reset_index(drop=True)
        ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=palette)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        for i in range(len(per_name_models)):
            sns.lineplot(x=df.index, y=df[f'model_{i}'].median(), ax=ax, linestyle='dashed', linewidth=2,
                         color=palette_lines[i])
        return target_df



# HEATMAP_CMAP = "vlag"


def _compute_cluster_metrics(annotated_df: pd.DataFrame):
    rows = []
    for cluster_id, grp in annotated_df.groupby("peptide_cluster"):
        counts = grp["allele"].value_counts()
        dominant = counts.index[0]
        purity = counts.iloc[0] / len(grp)
        frac_dict = {f"frac_{a}": counts.get(a, 0) / len(grp)
                     for a in annotated_df["allele"].unique()}
        row = {
            "cluster": cluster_id,
            "n_peptides": len(grp),
            "dominant_allele": dominant,
            "purity": purity,
            **frac_dict,
        }
        rows.append(row)

    purity_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

    true_labels = annotated_df["allele"].astype("category").cat.codes.values
    pred_labels = annotated_df["peptide_cluster"].values
    ari = adjusted_rand_score(true_labels, pred_labels)

    return purity_df, ari


def _save_cluster_metrics_plot(purity_df, ari, allele_names, save_path):
    fig, (ax_stack, ax_purity) = plt.subplots(1, 2, figsize=(10, 4))

    frac_cols = [c for c in purity_df.columns if c.startswith("frac_")]
    allele_short = [c.replace("frac_", "") for c in frac_cols]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x = np.arange(len(purity_df))
    bottoms = np.zeros(len(purity_df))

    for j, (col, allele) in enumerate(zip(frac_cols, allele_short)):
        ax_stack.bar(x, purity_df[col].values, bottom=bottoms,
                     color=palette[j % len(palette)], label=allele, width=0.6)
        bottoms += purity_df[col].values

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(purity_df["cluster"].astype(str), fontsize=8)
    ax_stack.set_xlabel("Cluster", fontsize=8)
    ax_stack.set_ylabel("Allele fraction", fontsize=8)
    ax_stack.set_title("Allele composition per cluster", fontsize=9)
    ax_stack.set_ylim(0, 1.05)
    ax_stack.legend(fontsize=6, loc="upper right")
    for xi, n in zip(x, purity_df["n_peptides"]):
        ax_stack.text(xi, 1.01, str(n), ha="center", va="bottom", fontsize=6)

    colors = [palette[j % len(palette)] for j in range(len(purity_df))]
    ax_purity.bar(x, purity_df["purity"].values, color=colors, width=0.6)
    ax_purity.axhline(1 / len(allele_names), color="gray", linestyle="--",
                      linewidth=1, label=f"random ({1 / len(allele_names):.2f})")
    ax_purity.set_xticks(x)
    ax_purity.set_xticklabels(purity_df["cluster"].astype(str), fontsize=8)
    ax_purity.set_xlabel("Cluster", fontsize=8)
    ax_purity.set_ylabel("Purity (dominant allele)", fontsize=8)
    ax_purity.set_title(f"Allele purity per cluster\nARI = {ari:.3f}", fontsize=9)
    ax_purity.set_ylim(0, 1.1)
    ax_purity.legend(fontsize=7)
    for xi, (_, row) in zip(x, purity_df.iterrows()):
        short = (row["dominant_allele"].split("_")[-1]
                 if "_" in row["dominant_allele"]
                 else row["dominant_allele"][:8])
        ax_purity.text(xi, row["purity"] / 2, short,
                       ha="center", va="center", fontsize=6,
                       color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)



def plot_split_diagnostics(
        per_allele_data_df,
        ann_df,
        n_clusters=None,
        layer_counter=None,
        save_dir=None,
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    ann_map = ann_df[0]

    ann_frames = []
    for real_allele, ann_part_df in ann_map.items():
        ann_part_df = ann_part_df.copy()
        if "source_allele" not in ann_part_df.columns and "allele" not in ann_part_df.columns:
            ann_part_df["allele"] = real_allele
        ann_frames.append(ann_part_df)

    ann_all = pd.concat(ann_frames, ignore_index=True)

    if "split" in ann_all.columns:
        ann_all = ann_all[ann_all["split"] == 0].copy()
        print("this method supports 1 split for now")

    if "source_allele" in ann_all.columns:
        allele_col = "source_allele"
    elif "allele" in ann_all.columns:
        allele_col = "allele"
    else:
        raise KeyError("Neither 'source_allele' nor 'allele' found in annotation dataframe")

    ann_all = ann_all[["peptide", allele_col]].drop_duplicates(subset=["peptide"])
    ann_all = ann_all.rename(columns={allele_col: "allele"})
    ann_all["peptide"] = ann_all["peptide"].astype(str)

    results = {}

    for allele_name in per_allele_data_df.keys():
        print(f"allele_name {allele_name}")

        df = per_allele_data_df[allele_name].copy()

        if "peptide" not in df.columns:
            df["peptide"] = df.index.astype(str)
        else:
            df["peptide"] = df["peptide"].astype(str)

        annotated_df = df.join(
            ann_all.set_index("peptide"),
            on="peptide",
            how="left",
        )
        annotated_df["allele"] = annotated_df["allele"].fillna("Unknown")

        model_cols = [col for col in df.columns if col.startswith("model")]
        if len(model_cols) == 0:
            raise ValueError(f"No model columns found for allele {allele_name}")

        target_df = df[model_cols].copy()
        target_df = target_df.apply(pd.to_numeric, errors="coerce")
        target_df = target_df.replace([np.inf, -np.inf], np.nan)
        target_df = target_df.fillna(target_df.median(numeric_only=True))
        target_df = target_df.fillna(0)
        target_df.index = df["peptide"].values

        X = target_df.values

        # --- KMeans ---
        if n_clusters is not None:
            best_k = n_clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            best_score = silhouette_score(X, labels)
            print(f"Using provided n_clusters={best_k}, silhouette={best_score:.4f}")
        else:
            silhouette_scores = {}
            k_range = range(2, min(16, len(X)))
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels)
                silhouette_scores[k] = score
                print(f"  k={k}: silhouette={score:.4f}")
            best_k = max(silhouette_scores, key=silhouette_scores.get)
            best_score = silhouette_scores[best_k]
            print(f"Best k={best_k} with silhouette={best_score:.4f}")
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

        peptide_clusters = labels + 1
        annotated_df = annotated_df.copy()
        annotated_df["peptide_cluster"] = peptide_clusters

        purity_df, ari = _compute_cluster_metrics(annotated_df)
        print(f"\n=== Cluster metrics for {allele_name} ===")
        print(f"ARI (clusters vs alleles): {ari:.4f}")
        print(purity_df[["cluster", "n_peptides", "dominant_allele", "purity"]].to_string(index=False))
        print()

        sort_df = annotated_df[["peptide", "allele", "peptide_cluster"]].copy()
        sort_df["_row_pos"] = np.arange(len(sort_df))
        sort_df = sort_df.sort_values(
            ["peptide_cluster", "allele", "peptide"],
            ascending=[True, True, True],
        )
        row_order = sort_df["_row_pos"].values
        target_df_sorted = target_df.iloc[row_order].reset_index(drop=True)
        annotated_df_sorted = annotated_df.iloc[row_order].reset_index(drop=True)

        unique_alleles = sorted(annotated_df_sorted["allele"].unique())
        allele_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#FD9000"]
        allele_color_map = dict(zip(unique_alleles, allele_palette))

        unique_clusters = sorted(annotated_df_sorted["peptide_cluster"].unique())
        cluster_palette = ["#4b598c", "#9467bd", "#7a8c4b", "#4b7c8c"]
        cluster_color_map = dict(zip(unique_clusters, cluster_palette))

        row_colors = pd.DataFrame(
            {
                "allele": annotated_df_sorted["allele"].map(allele_color_map).values,
                "kmeans_cluster": annotated_df_sorted["peptide_cluster"].map(cluster_color_map).values,
            },
            index=pd.RangeIndex(len(target_df_sorted)),
        )

        fig = plt.figure(figsize=(24, 10))
        gs = GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[5, 1.5, 1.5],
            wspace=0.35,
            left=0.03,
            right=0.97,
            top=0.92,
            bottom=0.12,
        )

        # ---------- Heatmap ----------
        ax_hm_host = fig.add_subplot(gs[0, 0])
        ax_hm_host.set_visible(False)   # placeholder для bbox


        annotated_df["allele"] = annotated_df["allele"].fillna("Unknown")

        g = sns.clustermap(
            target_df_sorted,
            row_cluster=False,
            col_cluster=False,
            yticklabels=False,
            xticklabels=1,
            row_colors=row_colors,
            cbar_pos=(0.02, 0.8, 0.02, 0.15),  # colorbar (left, bottom, width, height)
            cmap="magma",
            figsize=(14, 10),
        )
        ax_hm = g.ax_heatmap
        g.fig.set_size_inches(20, 10)
        g.fig.subplots_adjust(right=0.5)

        ax = g.fig.add_axes([0.55, 0.05, 0.42, 0.92])

        fit = umap.UMAP()
        u = fit.fit_transform(target_df.values)

        to_plot = pd.DataFrame({
            "x": u[:, 0],
            "y": u[:, 1],
            "allele": annotated_df["allele"].values,
            "peptide": annotated_df["peptide"].values,
        })
        allele_legend_handles = [Patch(facecolor=allele_color_map[a], label=a) for a in unique_alleles]

        legend_allele = g.fig.legend(
            handles=allele_legend_handles,
            title="Allele",
            loc="lower right",
            bbox_to_anchor=(0.11, 0.7),
            fontsize=11,
            title_fontsize=9,
            frameon=True,
        )
        g.fig.add_artist(legend_allele)  # чтобы обе легенды отобразились

        ax_hm.set_title(
            f"Summary of {len(target_df_sorted.columns)} models | "
            f"KMeans k={best_k} (best silhouette={best_score:.3f})",
            fontsize=10, pad=6,
        )

        # ---------- PCA ----------
        ax_pca = fig.add_subplot(gs[0, 1])
        pca2 = PCA(n_components=2, random_state=42)
        X_pca2 = pca2.fit_transform(X)
        pca_plot_df = pd.DataFrame({
            "x": X_pca2[:, 0],
            "y": X_pca2[:, 1],
            "allele": annotated_df["allele"].values,
        })
        sns.kdeplot(data=pca_plot_df, x="x", y="y", hue="allele",
                    palette=allele_color_map, ax=ax_pca,
                    common_norm=False, warn_singular=False)
        ax_pca.set_title(f"PCA · {len(model_cols)} models", fontsize=10)
        ax_pca.get_legend().remove()

        # ---------- UMAP ----------
        ax_umap = fig.add_subplot(gs[0, 2])
        n_neighbors = min(15, max(2, len(X) - 1))
        n_components_pca = min(50, X.shape[0], X.shape[1])
        pca_full = PCA(n_components=n_components_pca, random_state=42)
        X_pca_full = pca_full.fit_transform(X)
        fit = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
        u = fit.fit_transform(X_pca_full)
        to_plot = pd.DataFrame({
            "x": u[:, 0],
            "y": u[:, 1],
            "allele": annotated_df["allele"].values,
            "peptide": annotated_df["peptide"].values,
            "peptide_cluster": annotated_df["peptide_cluster"].values,
        })
        sns.kdeplot(data=to_plot, x="x", y="y", hue="allele",
                    palette=allele_color_map, ax=ax_umap,
                    common_norm=False, warn_singular=False)
        ax_umap.set_title(
            f"UMAP by allele | k={best_k} KMeans clusters", fontsize=10)

        allele_handles = [Patch(facecolor=allele_color_map[a], label=a)
                          for a in unique_alleles]
        ax_umap.legend(handles=allele_handles, title="Allele",
                       bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

        if save_dir is not None:
            prefix = f"layer_{layer_counter}_" if layer_counter is not None else ""

            heatmap_path = os.path.join(
                save_dir, f"{prefix}{allele_name}_kmeans_sorted_heatmap.png")
            g.fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            print("Saved:", heatmap_path)

            # PCA + UMAP
            fig_path = os.path.join(
                save_dir, f"{prefix}{allele_name}_pca_umap.png")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_path)


            # CSV с отсортированной матрицей признаков
            target_path = os.path.join(
                save_dir, f"{prefix}{allele_name}_features_matrix_kmeans_sorted.csv")
            target_to_save = target_df_sorted.copy()
            target_to_save.insert(0, "peptide", annotated_df_sorted["peptide"].values)
            target_to_save.insert(1, "peptide_cluster", annotated_df_sorted["peptide_cluster"].values)
            target_to_save.insert(2, "allele", annotated_df_sorted["allele"].values)
            target_to_save.to_csv(target_path, index=False)

            # UMAP-координаты
            umap_path = os.path.join(
                save_dir, f"{prefix}{allele_name}_umap_points.csv")
            to_plot.to_csv(umap_path, index=False)

            # Метрики кластеров
            metrics_path = os.path.join(
                save_dir, f"{prefix}{allele_name}_cluster_metrics.csv")
            purity_df["ari"] = ari
            purity_df["silhouette"] = best_score
            purity_df.to_csv(metrics_path, index=False)
            print("Saved:", metrics_path)

            # Barplot метрик кластеров
            barplot_path = os.path.join(
                save_dir, f"{prefix}{allele_name}_cluster_metrics_barplot.png")
            _save_cluster_metrics_plot(purity_df, ari, unique_alleles, barplot_path)

        results[allele_name] = {
            "umap": to_plot,
            "figure": fig,
            "sorted_matrix": target_df_sorted,
            "sorted_annotation": annotated_df_sorted,
            "purity_df": purity_df,
            "ari": ari,
        }

        plt.close(fig)
        plt.close(g.fig)

    first = next(iter(results.values()))
    return first["sorted_annotation"], first["figure"]

def save_all_visualization_results(per_name_models,
                                   per_name_histories,
                                   experiment_params: ExperimentParams,
                                   subfolder_to_safe_result: str, subset=None, predefined_hirerarchical_layout=True):
    if subset:
        per_name_models = {key: value for key, value in itertools.islice(per_name_models.items(), 0, subset)}
        per_name_histories = {key: value for key, value in itertools.islice(per_name_histories.items(), 0, subset)}
    path_to_save_models = f"{experiment_params.experiment_result_data_path}/{subfolder_to_safe_result}/"
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params
    # learning curve
    fig, ax = plt.subplots(1, 1)
    for i in range(1):
        sns.lineplot(
            per_name_histories[
                list(per_name_histories.keys())[i]
            ].log_probabilities, ax=ax)
    plt.close(fig)
    print('\nModelGraph')
    for i, model in enumerate(per_name_models.values()):
        print(i, end=' ')

        out_path = Path(path_to_save_models) / model.name / "state_graph_cropped.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            model.plot(file=f, crop_zero=True)
    print('\nDistributions')
    for i, model in enumerate(per_name_models.values()):
        print(i, end=' ')
        plot_distributions_for_states(model, path_to_save_models, horizontal=False,
                                      discrete=model_training_params.aa_labels_training,
                                      initial_params=model_training_params.initial_params)

    print('\nPyViz')
    per_name_pyviz = dict()
    for name, model in per_name_models.items():
        per_name_pyviz[
            name] = make_pyviz_graph(model, path_to_save_models, precision=3, prefefined_hierarchical_layout=predefined_hirerarchical_layout)


def plot_info_matrices_to_file(
    *info_matrices,           # любое число матриц веток
    folder_to_save=None,
    allele_name=None,
    main_title=None,
    filename=None,
):
    if folder_to_save:
        os.makedirs(folder_to_save, exist_ok=True)

    branches = [(i + 1, m) for i, m in enumerate(info_matrices) if m is not None]

    nrows = len(branches)
    fig, axes = plt.subplots(ncols=1, nrows=nrows, figsize=(5, 3 * nrows), squeeze=False)

    for ax, (branch_num, matrix) in zip(axes[:, 0], branches):
        ww_logo = logomaker.Logo(matrix, font_name='Arial', flip_below=True, ax=ax)
        ww_logo.ax.set_title(f"branch {branch_num}")

    fig.suptitle(main_title)
    plt.tight_layout()

    save_figure_to_svg(fig, dir=folder_to_save, filename=filename)
    plt.close(fig)