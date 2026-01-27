import os  # работа с ОС (пути, переменные окружения и т.п.); в этом коде напрямую не используется
from pathlib import Path  # удобная работа с путями как с объектами
from protlearn.features import aaindex1  # получение AAindex1 свойств аминокислот
import numpy as np  # матричные операции и численные вычисления
import pandas as pd  # работа с таблицами (DataFrame)
from sklearn.cluster import FeatureAgglomeration  # агломеративная кластеризация признаков (сжатие фич)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # построение графиков
import umap  # UMAP для снижения размерности (в этом коде не используется)
from sklearn.decomposition import PCA  # PCA (в этом коде не используется)
from scipy.stats import gaussian_kde  # KDE-оценка плотности (в этом коде не используется)
from scipy.cluster.hierarchy import linkage, dendrogram  # иерархическая кластеризация и дендрограмма
from scipy.spatial.distance import squareform  # перевод матрицы расстояний в “сжатый” формат
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PROJECT_ROOT = Path("/Users/annaklimova/Desktop/Washu_project")  # корень проекта (абсолютный путь)
OUTPUT_DIR = PROJECT_ROOT / "output"  # папка для выходных файлов внутри проекта
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # создать output (и родителей), если не существует

CSV_PATH = PROJECT_ROOT / "filtered_df.csv"  # путь к входному CSV с данными


def open_iedb_data(csv_path: Path) -> pd.DataFrame:  # функция чтения IEDB-таблицы из CSV
    return pd.read_csv(csv_path)  # читаем CSV и возвращаем DataFrame


def get_data_for_allele(
    df: pd.DataFrame,
    allele: str = "HLA-DPA1*02:02/DPB1*05:01"
) -> pd.DataFrame:
    return df.loc[
        (df["Allele"] == allele) &
        (df["Binder"] == 1)
    ]

def get_peptide_splits(df: pd.DataFrame, core_window: int = 9) -> dict:  # нарезка пептидов в окна длины core_window
    peptide_splits_dict = {}  # словарь: исходный пептид -> список окон (каждое окно как список символов)
    for pep in df["Peptide"]:  # идём по всем пептидам в столбце Peptide
        if len(pep) > core_window:  # если пептид длиннее окна (например 15 > 9)
            windows = [pep[p:p + core_window] for p in range(0, len(pep) - core_window + 1)]  # все скользящие 9-меры
            peptide_splits_dict[pep] = [list(w) for w in windows]  # сохраняем список окон, каждое окно -> список аминокислот
        else:  # если пептид длиной <= 9
            windows = [pep]  # единственное “окно” — сам пептид (короче 9)
            peptide_splits_dict[pep] = [list(w) for w in windows]  # сохраняем как список букв

    return peptide_splits_dict  # возвращаем словарь нарезок

def make_amino_pattern_table(peptide_splits_dict, core_window=9):  # делаем таблицу позиций 1..9 для каждого окна
    rows = []  # сюда будем собирать строки будущего DataFrame

    for peptide, windows in peptide_splits_dict.items():  # перебираем: пептид -> список окон
        for w in windows:  # перебираем каждое окно (список аминокислот)
            if len(w) != core_window:  # пропускаем окна, которые не ровно 9 (например короткие пептиды)
                continue  # перейти к следующему окну

            row = {"Peptide": peptide}  # начинаем строку с исходного пептида (идентификатор)
            for i, aa in enumerate(w, start=1):  # нумеруем позиции с 1 до 9
                row[i] = aa  # столбцы 1..9 содержат букву аминокислоты на позиции i

            rows.append(row)  # добавляем строку в общий список

    df = pd.DataFrame(rows)  # создаём DataFrame из накопленных строк
    df = df[["Peptide"] + list(range(1, core_window + 1))]  # упорядочиваем колонки: Peptide, 1..9
    return df  # возвращаем таблицу 9-меров (по позициям)


def corr_clustermap(
        df,
        title=None,
        outpath=None,
        method="spearman",
        linkage_method="average",
        figsize=(14, 14),
        dpi=400,
        label_fontsize=7,
        label_step=7,
):
    # 1. Вычисление корреляции
    corr = df.corr(method=method)
    M = np.abs(corr.values)

    # 2. Иерархическая кластеризация
    dist = 1.0 - M
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method=linkage_method)

    # 3. Определение порядка
    dendro = dendrogram(Z, no_plot=False)
    order = dendro["leaves"]

    M = M[np.ix_(order, order)]
    labels = corr.columns.to_list()
    labels = [labels[i] for i in order]
    n = len(labels)

    # --- NEW: индексы, где реально рисуем подписи
    step = max(int(label_step), 1)
    tick_idx = np.arange(0, n, step)

    # 4. Создание фигуры
    fig = plt.figure(figsize=figsize, facecolor='white')

    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[0.18, 1.0],
        height_ratios=[0.18, 1.0],
        wspace=0.005, hspace=0.005
    )

    ax_dendro_left = fig.add_subplot(gs[1, 0])
    ax_dendro_top = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, 1])

    # 5. Дендрограммы
    dendrogram(
        Z, ax=ax_dendro_top, orientation="top",
        no_labels=True, color_threshold=0, above_threshold_color="black"
    )
    dendrogram(
        Z, ax=ax_dendro_left, orientation="left",
        no_labels=True, color_threshold=0, above_threshold_color="black"
    )

    for ax in (ax_dendro_top, ax_dendro_left):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # 6. Хитмап
    im = ax_heat.imshow(
        M, vmin=0.0, vmax=1.0,
        cmap="magma",
        interpolation="nearest",
        aspect="auto"
    )

    # --- CHANGED: показываем только tick_idx
    ax_heat.set_xticks(tick_idx)
    ax_heat.set_xticklabels([labels[i] for i in tick_idx], rotation=90, fontsize=label_fontsize)
    ax_heat.set_yticks([])
    ax_heat.tick_params(axis='x', length=2, pad=2)

    # Y-метки справа
    ax_right = ax_heat.twinx()
    ax_right.set_ylim(ax_heat.get_ylim())
    ax_right.set_yticks(tick_idx)
    ax_right.set_yticklabels([labels[i] for i in tick_idx], fontsize=label_fontsize)
    ax_right.tick_params(axis="y", length=0, pad=2)
    for sp in ax_right.spines.values():
        sp.set_visible(False)

    # 7. Colorbar
    ax_cbar_area = fig.add_subplot(gs[0, 0])
    ax_cbar_area.set_axis_off()

    cax = inset_axes(
        ax_cbar_area,
        width="30%",
        height="70%",
        loc="center",
        borderpad=0
    )
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=label_fontsize + 1, length=3, direction='out')
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(0.8)

    # 8. Заголовок
    fig.suptitle(title, fontsize=28, y=0.97)

    if outpath:
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {outpath}")

    plt.close()



def main():  # главная функция пайплайна
    train_df = open_iedb_data(CSV_PATH)  # читаем исходные данные из CSV

    allele_train_df = get_data_for_allele(train_df, "HLA-DPA1*02:02/DPB1*05:01")  # фильтруем по конкретному аллелю
    peptide_splits_dict = get_peptide_splits(allele_train_df, core_window=9)  # режем пептиды на 9-мерные окна

    print("Peptides:", len(peptide_splits_dict))  # печатаем число уникальных пептидов (ключей словаря)

    for i, (pep, windows) in enumerate(peptide_splits_dict.items()):  # проходим по пептидам и их окнам с индексом
        print("Peptide:", pep)  # печать пептида
        print("Num windows:", len(windows))  # сколько 9-меров получилось из пептида
        print("First windows:", windows[:5])  # первые 5 окон для примера
        print("-" * 30)  # разделитель в консоли
        if i == 30:  # только для первого пептида
            break  # выходим из цикла, чтобы не засорять вывод

# ----------------------------------------------------------------------------------------------------------------------

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")  # список 20 стандартных аминокислот в однобуквенном коде
    prop_values, prop_names = aaindex1(amino_acids, standardize="zscore")  # матрица свойств + названия свойств (z-score)

    aa_prop_index = (  # DataFrame: строки=AA, колонки=свойства (AAindex)
        pd.DataFrame(prop_values, columns=prop_names, index=amino_acids)  # создаём таблицу 20 x N
        .dropna(axis="columns")  # выкидываем свойства, где есть NaN
    )

# ----------------------------------------------------------------------------------------------------------------------
#     Шаг 1: Кластеризация свойств(AAindex)

    X = aa_prop_index.values  # превращаем DataFrame в numpy массив (20, n_features)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)  # (закомментировано) стандартизация, если нужна

    n_clusters = 73  # число кластеров признаков

    agg = FeatureAgglomeration(  # создаём модель агломерации признаков
        n_clusters=n_clusters,  # целевое число кластеров
        linkage="ward",  # Ward linkage (минимизация внутрикластерной дисперсии)
    )

    X_reduced = agg.fit_transform(X)  # обучаем агломерацию и получаем reduced-признаки (20, 73)
    labels = agg.labels_  # метка кластера для каждого исходного свойства (длина = n_features)
    cluster_names = [f"Cluster_{i:03d}" for i in range(n_clusters)]  # имена колонок кластеров: Cluster_000..Cluster_072

    aa_prop_reduced = pd.DataFrame(  # DataFrame reduced свойств, рассчитанных sklearn (среднее/аггломерация)
        X_reduced,  # матрица 20 x 73
        index=aa_prop_index.index,  # строки = аминокислоты
        columns=cluster_names  # колонки = кластеры
    )

    feature_to_cluster = pd.Series(labels, index=aa_prop_index.columns, name="cluster_id")  # серия: свойство -> cluster_id
    clusters = {}  # словарь: cluster_id -> список свойств
    for fid, cid in zip(aa_prop_index.columns, labels):  # перебираем (feature_id, cluster_id)
        clusters.setdefault(cid, []).append(fid)  # добавляем feature в список соответствующего кластера

    # Среднее по колонкам каждого кластера
    aa_prop_reduced_manual = pd.DataFrame(index=aa_prop_index.index)  # вручную создаём DataFrame reduced свойств (пустой)

    for cid, feats in sorted(clusters.items()):  # идём по кластерам по порядку id
        aa_prop_reduced_manual[f"Cluster_{cid:03d}"] = aa_prop_index[feats].mean(axis=1)  # усредняем свойства внутри кластера

    print(aa_prop_reduced_manual)  # печать таблицы reduced (может быть большой)
    cluster_sizes = feature_to_cluster.value_counts().sort_values(ascending=False)  # размеры кластеров (сколько фич в каждом)
    print(cluster_sizes.head(15))  # топ-15 самых больших кластеров

    # ---------------------------------------------------------------------------------------------------------------
    amino_pattern_table = make_amino_pattern_table(  # строим таблицу 9-меров: Peptide + позиции 1..9
        peptide_splits_dict,  # словарь с окнами
        core_window=9  # длина окна
    )


    # --- Шаг 2: векторизация 9-меров в таблицу 657 признаков (9 * 73)

    pos_cols = list(range(1, 10))  # позиции 1..9 в amino_pattern_table
    prop_mat = aa_prop_reduced_manual  # (20 x 73), index = AA

    # (N x 9) letters
    letters = amino_pattern_table[pos_cols].to_numpy()

    # буквы -> индексы строк в prop_mat
    aa_to_row = {aa: i for i, aa in enumerate(prop_mat.index)}
    idx = np.vectorize(aa_to_row.get)(letters)  # (N, 9)

    # свойства для каждой буквы в каждой позиции: (N, 9, 73)
    prop_values = prop_mat.to_numpy()  # (20, 73)
    encoded = prop_values[idx]  # (N, 9, 73)

    # расплющиваем в (N, 657)
    X_peptides = encoded.reshape(encoded.shape[0], -1)

    # делаем имена колонок: 0_new_prop_0 ... 8_new_prop_72
    n_props = prop_mat.shape[1]  # 73
    col_names = [
        f"{pos}_new_prop_{k}"
        for pos in range(9)  # pos = 0..8
        for k in range(n_props)  # k = 0..72
    ]

    # итоговая таблица (N x 657)
    transformed_peptides_table = pd.DataFrame(X_peptides, columns=col_names)

    transformed_peptides_table.insert(0, "Peptide", amino_pattern_table["Peptide"].values)

    print(transformed_peptides_table.shape)  # будет (N, 658) если с Peptide, иначе (N, 657)
    print(transformed_peptides_table.iloc[:30, :5])


    # corr_clustermap(  # строим и сохраняем clustermap корреляций исходных AAindex свойств
    #     aa_prop_index,  # исходные свойства (до кластеризации)
    #     # title="Absolute Correlation of properties",  # заголовок
    #     outpath=OUTPUT_DIR / "abs_corr_clustermap_before.png",  # путь сохранения “до”
    #     method="spearman",  # корреляция Спирмена
    #     linkage_method="ward",  # average linkage для дендрограммы
    #     # show_labels=True  # показывать подписи
    # )
    #
    # corr_clustermap(  # строим и сохраняем clustermap корреляций после сжатия (73 кластера)
    #     aa_prop_reduced_manual,  # reduced свойства (после кластеризации/усреднения)
    #     # title="Absolute Correlation of properties",  # заголовок
    #     outpath=OUTPUT_DIR / "abs_corr_clustermap_after.png",  # путь сохранения “после”
    #     method="spearman",  # корреляция Спирмена
    #     linkage_method="ward",  # average linkage
    #     label_step=1
    #     # show_labels=True  # показывать подписи
    # )


if __name__ == "__main__":
    main()
