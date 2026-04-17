# Скрипт генерирует "пептиды" (строки из аминокислот),
# в которых есть "ядро" (core) с мотивами (anchors) и фланки (случайные аминокислоты).
# Также добавляется "noise"  часть core заменяется на случайные (шумовые).

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt  # графики
import logomaker
from pomegranate import DiscreteDistribution
from itertools import combinations_with_replacement  # комбинации с повторениями

# ====== Настройки "чтобы вывод был понятный" ======
np.set_printoptions(suppress=True)  # чтобы numpy не печатал 1e-05 в странном виде
pd.set_option("display.width", 160)  # чтобы DataFrame печатался шире
pd.set_option("display.max_columns", 50)  # показывать больше колонок в консоли

# ====== Глобальные настройки генерации ======
CLASS_TO_GENERATE = "II"  #or "I"
# длины пептидов, которые будем генерить (зависит от класса)
LENGTHS_TO_GENERATE = [12, 13, 14, 15, 16, 17, 18, 19] if CLASS_TO_GENERATE == "II" else [8, 9, 10, 11, 12, 13, 14]

# список аминокислот
amino_acids_list = list("ACDEFGHIKLMNPQRSTVWY")

# genm к данным
DATA_PATH_BASE = "/Users/annaklimova/Desktop/Washu_project/data/simple_model_enrichment/simulated_data/16_02"

scenario = "five_different_motifs"  # можно менять

# Папка для результата: base + scenario
RESULT_DATA_PATH = os.path.join(DATA_PATH_BASE, scenario)
Path(RESULT_DATA_PATH).mkdir(parents=True, exist_ok=True)

# Папка для logo-графиков
LOGO_PATH = os.path.join(RESULT_DATA_PATH, "logos")
Path(LOGO_PATH).mkdir(parents=True, exist_ok=True)

print(f"Scenario: {scenario}")
print(f"Result folder: {RESULT_DATA_PATH}")
print(f"Class: {CLASS_TO_GENERATE}")
print(f"Lengths to generate: {LENGTHS_TO_GENERATE}")
print()

# ====== Сценарии мотивов ======
# motifs - это список. Каждый элемент списка = один "мотив".
# Мотив - это dict: позиция -> набор допустимых аминокислот в этой позиции.
motifs = []

if scenario == "two_distinct_motifs":
    # ====== Two distinct motifs ======

    anchors1 = {}  # первый мотив
    anchors1[0] = {"L", "V", "P"}
    anchors1[3] = {"S", "G", "T", "I"}
    anchors1[5] = {"H", "R", "K", "Q"}
    anchors1[8] = {"L", "V", "W"}
    motifs.append(anchors1)  # добавили мотив в список

    anchors2 = {}  # второй мотив
    anchors2[0] = {"M", "N", "P"}
    anchors2[3] = {"L", "M", "W", "K"}
    anchors2[5] = {"D", "E", "Y", "F"}
    anchors2[8] = {"K", "T"}
    motifs.append(anchors2)

elif scenario == "two_more_similar_motifs":
    anchors1 = {}
    anchors1[0] = {"L", "F", "P"}
    anchors1[3] = {"S", "G", "T", "I"}
    anchors1[5] = {"H", "R", "K", "Q"}
    anchors1[8] = {"L", "V", "W"}
    motifs.append(anchors1)

    anchors2 = {}
    anchors2[0] = {"L", "V", "F", "P"}
    anchors2[3] = {"L", "M", "W", "K"}
    anchors2[5] = {"D", "E", "Y", "F"}
    anchors2[8] = {"L", "V"}
    motifs.append(anchors2)

elif scenario == "two_motifs_with_different_positions":
    anchors1 = {}
    anchors1[0] = {"L", "F", "P"}
    anchors1[2] = {"S", "G", "T", "I"}
    anchors1[5] = {"H", "R", "K", "Q"}
    anchors1[8] = {"L", "V", "W"}
    motifs.append(anchors1)

    anchors2 = {}
    anchors2[0] = {"L", "V", "F", "P"}
    anchors2[3] = {"L", "M", "W", "K"}
    anchors2[5] = {"D", "E", "Y", "F"}
    anchors2[8] = {"L", "V"}
    motifs.append(anchors2)

elif scenario == "three_different_motifs":
    anchors1 = {}
    anchors1[0] = {"L", "V", "P"}
    anchors1[3] = {"S", "G", "T", "I"}
    anchors1[5] = {"H", "R", "K", "Q"}
    anchors1[8] = {"L", "V", "W"}
    motifs.append(anchors1)

    anchors2 = {}
    anchors2[0] = {"M", "N", "P"}
    anchors2[3] = {"L", "M", "W", "K"}
    anchors2[5] = {"D", "E", "Y", "F"}
    anchors2[8] = {"K", "T"}
    motifs.append(anchors2)

    anchors3 = {}
    anchors3[0] = {"V", "W", "F"}
    anchors3[3] = {"H", "K", "R"}
    anchors3[5] = {"G", "S", "T"}
    anchors3[8] = {"N", "Q"}
    motifs.append(anchors3)

elif scenario == "five_different_motifs":
    # ====== Five distinct motifs ======

    anchors1 = {}
    anchors1[0] = {"L", "V", "P"}
    anchors1[3] = {"S", "G", "T", "I"}
    anchors1[5] = {"H", "R", "K", "Q"}
    anchors1[8] = {"L", "V", "W"}
    motifs.append(anchors1)

    anchors2 = {}
    anchors2[0] = {"M", "N", "P"}
    anchors2[3] = {"L", "M", "W", "K"}
    anchors2[5] = {"D", "E", "Y", "F"}
    anchors2[8] = {"K", "T"}
    motifs.append(anchors2)

    anchors3 = {}
    anchors3[0] = {"V", "W", "F"}
    anchors3[3] = {"H", "K", "R"}
    anchors3[5] = {"G", "S", "T"}
    anchors3[8] = {"N", "Q"}
    motifs.append(anchors3)

    anchors4 = {}
    anchors4[0] = {"K", "R", "H"}
    anchors4[3] = {"L", "V", "F"}
    anchors4[5] = {"F", "Q", "H"}
    anchors4[8] = {"S", "T", "N"}
    motifs.append(anchors4)

    anchors5 = {}
    anchors5[0] = {"D", "E"}
    anchors5[3] = {"Q", "M", "I"}
    anchors5[5] = {"L", "V", "W"}
    anchors5[8] = {"K", "R"}
    motifs.append(anchors5)

elif scenario == "random_cores":
    # В этом сценарии "якоря" разрешают любые аминокислоты
    anchors = {}
    anchors[0] = set(amino_acids_list)
    anchors[8] = set(amino_acids_list)
    motifs.append(anchors)

    anchors = {}
    anchors[0] = set(amino_acids_list)
    anchors[8] = set(amino_acids_list)
    motifs.append(anchors)

else:
    raise ValueError(f"Unknown scenario: {scenario}")

print(f"Motifs count: {len(motifs)}")
for i, m in enumerate(motifs):
    # какие позиции заданы
    print(f"  Motif {i}: positions={sorted(m.keys())}")
print()

# ====== Длины core (ядра) ======
# core_lengths: для каждого мотива считаем длину ядра = max позиция + 1
core_lengths = [max(motif.keys()) + 1 for motif in motifs]
print(f"Core lengths per motif: {core_lengths}")
print()

# ====== Примеры распределений длины пептидов ======
center_dist_examples = [
    {12: 357, 13: 812, 14: 1251, 15: 5369, 16: 1449, 17: 991, 18: 661},
    {12: 148, 13: 458, 14: 656, 15: 3117, 16: 900, 17: 585, 18: 327},
]

# ====== Строим length_distributions: распределение длины для каждого мотива ======
length_distributions = []

for i in range(len(motifs)):  # по каждому мотиву
    d_length_motif = DiscreteDistribution()  # распределение pomegranate: значение -> вероятность

    # Берём пример распределения:
    # - для первых 2 мотивов берём example[0] и example[1]
    # - дальше случайно выбираем один из примеров
    if i >= 2:
        source_dist = center_dist_examples[np.random.randint(0, len(center_dist_examples))]
    else:
        source_dist = center_dist_examples[i]

    fit_data = []  # сюда набьём много длины, чтобы по ним "подогнать" распределение

    for target_length, example_cnt in source_dist.items():
        # Небольшая случайность: +-20% к количеству примеров, чтобы разные мотивы отличались
        jitter = np.random.randint(int(-example_cnt * 0.2), int(example_cnt * 0.2))
        real_cnt = max(1, example_cnt + jitter)  # чтобы не получилось 0 или минус
        fit_data += [target_length] * real_cnt  # добавляем длину много раз

    # Обучаем распределение на "данных"
    d_length_motif.fit(fit_data)
    length_distributions.append(d_length_motif)

print("Built length distributions for motifs.")
print()

# ====== Распределение стартовой позиции core: чаще в середине ======
def get_start_dist_middle(target_length: int = 15, motif_len: int = 9) -> DiscreteDistribution:
    """
    Генерим распределение стартовой позиции core внутри пептида.
    Идея: core чаще начинается в середине, чем на краях.

    target_length — длина пептида
    motif_len — длина core
    """
    # возможные старты: от 0 до (target_length - motif_len)
    x = np.arange(0, target_length - (motif_len - 1))

    # xU/xL нужны для интеграла нормального распределения по "коробкам"
    xU, xL = x + 0.5, x - 0.5

    mean_x = np.mean(x)  # центр диапазона (примерно середина)

    # считаем вероятность каждой позиции через разность CDF (как площадь под кривой)
    probs = ss.norm.cdf(xU, np.floor(mean_x), scale=1.8) - ss.norm.cdf(xL, np.floor(mean_x), scale=1.8)
    probs = probs / probs.sum()  # нормируем, чтобы сумма была 1

    # превращаем в словарь: позиция -> вероятность
    d_dict = {key: prob for key, prob in zip(x, probs)}

    # создаём дискретное распределение
    return DiscreteDistribution(d_dict)

# ====== Стартовые распределения для каждого мотива и каждой длины ======
# start_distributions[motif_index][target_length] = DiscreteDistribution(...)
start_distributions = [
    {target_length: get_start_dist_middle(target_length, core_lengths[i]) for target_length in LENGTHS_TO_GENERATE}
    for i in range(len(motifs))
]

print("Built start distributions.")
print()

# ====== Генерация комбинаций долей мотивов (ratio_combination) ======
# шаг 5%: 0.05, 0.10, ... 0.95
RATIOS_TO_GENERATE = list(np.round(np.linspace(0, 1, 20, endpoint=False), 2))[1:]

# все комбинации с повторениями длиной = число мотивов
total_combinations = list(combinations_with_replacement(RATIOS_TO_GENERATE, len(motifs)))

# берём только те, что суммируются ровно в 1 (например 0.2+0.3+0.5)
sum_to_one_combinations = [comb for comb in total_combinations if sum(comb) == 1]

print(f"Ratios candidates: {len(RATIOS_TO_GENERATE)}")
print(f"Ratio combinations total: {len(total_combinations)}")
print(f"Ratio combinations sum-to-1: {len(sum_to_one_combinations)}")
if sum_to_one_combinations:
    print(f"Example ratio: {sum_to_one_combinations[0]}")
print()

# ====== Noise уровни ======
# шум: доля пептидов, где core будет случайный
NOIZE_LEVELS_TO_GENERATE = [0.25]

print(f"Noise levels: {NOIZE_LEVELS_TO_GENERATE}")
print()

# ====== Функция: построить распределение аминокислот на позиции ======
def get_new_distribution(motifs, motif_num: int = -1, position: int = 0) -> DiscreteDistribution:
    """
    Возвращаем DiscreteDistribution по аминокислотам.
    Если motif_num >= 0 и position в anchors мотива -> делаем распределение только по "разрешённым" AA.
    Иначе -> случайное распределение по всем 20 AA.
    """
    # базово: случайные веса для каждой аминокислоты
    rng = np.random.default_rng()
    d_dict = {aa: w for aa, w in zip(amino_acids_list, rng.uniform(0, 1, size=len(amino_acids_list)))}

    # если выбрали конкретный мотив
    if motif_num >= 0:
        anchors = motifs[motif_num]  # dict позиция -> набор AA

        # если конкретная позиция — якорная
        if position in anchors:
            # тогда в этой позиции разрешаем только аминокислоты из anchors[position]
            d_dict = {aa: 0.0 for aa in amino_acids_list}
            for aa in anchors[position]:
                d_dict[aa] = rng.uniform(0, 1)

    # нормировка: сумма вероятностей должна быть 1
    total = sum(d_dict.values())
    d_dict = {aa: (w / total if total > 0 else 1.0 / len(amino_acids_list)) for aa, w in d_dict.items()}

    # превращаем словарь в распределение pomegranate
    return DiscreteDistribution(d_dict)

# ====== Функция: сгенерировать cores (ядра) ======
def generate_cores(motifs, n: int, ratio_combination, noize_level: float):
    """
    Генерим массив cores длины n:
    - выбираем, какой мотив для каждого пептида (по ratio_combination)
    - генерим core посимвольно по распределениям
    - часть cores заменяем на "noise core" (случайные аминокислоты), доля = noize_level
    """
    print("---- generate_cores() ----")
    print(f"n={n}, ratio_combination={ratio_combination}, noize_level={noize_level}")

    # распределение выбора мотива: motif_index -> probability
    motifs_selection_distribution = DiscreteDistribution({i: ratio for i, ratio in enumerate(ratio_combination)})

    # распределение "noise или нет": 1 = шум, 0 = норм
    noize_selection_distribution = DiscreteDistribution({1: noize_level, 0: 1 - noize_level})

    # для каждого мотива строим распределения AA по каждой позиции core
    motifs_aa_distributions = [
        {pos: get_new_distribution(motifs, motif_num=i, position=pos) for pos in range(core_lengths[i])}
        for i in range(len(motifs))
    ]

    # сюда сложим готовые core строки
    cores = np.empty(shape=(n,), dtype=object)

    # labels: какой мотив у каждого пептида
    peptide_labels = np.array(motifs_selection_distribution.sample(n))

    # noize_labels: шум или нет
    noize_labels = np.array(noize_selection_distribution.sample(n))

    # Генерим cores по мотивам
    for current_motif, motif_dist_by_pos in enumerate(motifs_aa_distributions):
        core_len = core_lengths[current_motif]  # длина core

        # индексы тех пептидов, которые относятся к этому мотиву
        idx_motif = (peptide_labels == current_motif)

        curr_motif_cnt = int(idx_motif.sum())  # сколько таких пептидов
        idx_noize_inside_motif = (noize_labels == 1) & idx_motif
        curr_noize_cnt = int(idx_noize_inside_motif.sum())  # сколько из них станет шумом

        print(f"Motif {current_motif}: total={curr_motif_cnt}, will_noise_replace={curr_noize_cnt}, core_len={core_len}")

        # если нет пептидов этого мотива — пропускаем
        if curr_motif_cnt == 0:
            continue

        # генерим матрицу (curr_motif_cnt x core_len)
        current_cores = np.empty((curr_motif_cnt, core_len), dtype=str)

        # для каждой позиции core — семплим нужное число аминокислот
        for pos in range(core_len):
            current_cores[:, pos] = motif_dist_by_pos[pos].sample(curr_motif_cnt)

        # превращаем строки типа ['A','C','D'...] в одну строку "ACD..."
        cores_for_motif = list(np.apply_along_axis(lambda row: "".join(row), axis=1, arr=current_cores))

        # кладём их в общий массив cores на нужные места
        cores[idx_motif] = cores_for_motif

        # печать примера
        print(f"Example core for motif {current_motif}: {cores_for_motif[0]}")

        # Теперь генерим "noise cores" и заменяем часть core
        if curr_noize_cnt > 0:
            noise_cores = np.empty((curr_noize_cnt, core_len), dtype=str)

            for pos in range(core_len):
                # шумовая позиция: распределение по всем аминокислотам (без anchors)
                noise_dist = get_new_distribution(motifs, motif_num=-1, position=pos)
                noise_cores[:, pos] = noise_dist.sample(curr_noize_cnt)

            noise_cores_as_str = list(np.apply_along_axis(lambda row: "".join(row), axis=1, arr=noise_cores))

            # заменяем только там, где noise_labels == 1 и motif совпадает
            cores[idx_noize_inside_motif] = noise_cores_as_str

            print(f"Example NOISE core for motif {current_motif}: {noise_cores_as_str[0]}")

    print("---- generate_cores() DONE ----")
    print()
    return cores, peptide_labels, noize_labels

# ====== Функция: сгенерировать полные пептиды (core + flanks) ======
def generate_peptides(motifs, cores, peptide_labels, noize_labels):
    """
    Для каждого core:
    1) выбираем длину пептида (по length_distributions для мотива)
    2) выбираем start позицию core (по start_distributions)
    3) добавляем слева/справа случайные фланки
    4) собираем DataFrame
    """
    print("---- generate_peptides() ----")

    # ====== 1) длины пептидов ======
    peptide_lengths = np.zeros(len(cores), dtype=int)

    for current_motif in range(len(motifs)):
        length_dist = length_distributions[current_motif]  # распределение длины
        idx_motif = (peptide_labels == current_motif)  # где этот мотив
        cnt = int(idx_motif.sum())  # сколько таких

        if cnt == 0:
            continue

        # семплим длины
        peptide_lengths[idx_motif] = np.array(length_dist.sample(cnt), dtype=int)

    print("Peptide lengths filled.")
    print("Example lengths:", peptide_lengths[:10])
    print()

    # ====== 2) старт core внутри пептида ======
    core_starts = np.zeros(len(cores), dtype=int)

    for curr_motif in range(len(motifs)):
        per_length_start_dist = start_distributions[curr_motif]  # словарь: длина -> распределение стартов

        for target_length in LENGTHS_TO_GENERATE:
            idx_subset = (peptide_lengths == target_length) & (peptide_labels == curr_motif)
            cnt_subset = int(idx_subset.sum())
            if cnt_subset == 0:
                continue

            # семплим стартовые позиции
            core_starts[idx_subset] = np.array(per_length_start_dist[target_length].sample(cnt_subset), dtype=int)

    print("Core starts filled.")
    print("Example core_starts:", core_starts[:10])
    print()

    # ====== 3) фланки (случайные AA) ======
    flank_dist = get_new_distribution(motifs, motif_num=-1, position=0)  # просто случайное распределение

    result_peptides = []
    result_alleles = np.array([f"Dummy_allele_{m}" for m in peptide_labels])

    for core_start, core, peptide_length in zip(core_starts, cores, peptide_lengths):
        core_len = len(core)

        # слева фланк длиной core_start
        left_flank = "".join(flank_dist.sample(core_start))

        # справа сколько осталось
        right_len = peptide_length - core_len - len(left_flank)

        # справа фланк
        right_flank = "".join(flank_dist.sample(right_len))

        pep = left_flank + core + right_flank

        # проверка “на всякий случай”
        assert len(pep) == peptide_length, (core_start, len(left_flank), len(right_flank), peptide_length)

        result_peptides.append(pep)

    print("Peptides assembled.")
    print("Example peptide:", result_peptides[0])
    print()

    # ====== 4) DataFrame ======
    df = pd.DataFrame({
        "peptide": result_peptides,
        "core": cores,
        "core_start": core_starts,
        "length": peptide_lengths,
        "allele": result_alleles,
        "noize": noize_labels,
        "binder": 1
    })

    print("DataFrame created.")
    print(df.head(5))
    print("---- generate_peptides() DONE ----")
    print()
    return df

# ====== Визуализация logo ======
def make_logo_for_data(binders_array, name, ax=None):
    """
    Делает logo (частоты/информация по символам).
    binders_array — список строк одинаковой длины (например core)
    """
    sns.set_theme(style="white")

    # превращаем выравнивание в матрицу частот
    matrix = logomaker.alignment_to_matrix(binders_array)

    # преобразуем counts -> information (обычно для sequence logo)
    info_matrix = logomaker.transform_matrix(matrix, from_type="counts", to_type="information")

    # строим logo
    ww_logo = logomaker.Logo(info_matrix, font_name="Arial Rounded MT Bold", flip_below=True, ax=ax)

    # заголовок
    ww_logo.ax.set_title(f"{name}, n={len(binders_array)}")

def make_logo_restricted(df, peptide_len, core_start=-1):
    """
    Рисуем logo для пептидов фиксированной длины,
    опционально ещё и фиксированного core_start.
    """
    target_df = df[df.length == peptide_len]
    title = f"peptide_len={peptide_len}"

    if core_start >= 0:
        target_df = target_df[target_df.core_start == core_start]
        title += f", core_start={core_start}"

    if len(target_df) == 0:
        print(f"[WARN] No data for logo: {title}")
        return

    make_logo_for_data(target_df.peptide.tolist(), title)

# ====== MAIN: чтобы PyCharm нормально запускал файл ======
def main():
    # Сохраним, какие длины мотивов есть (для имени файла)
    lengths_generated = tuple(sorted(set(max(motif.keys()) for motif in motifs)))

    results = {}  # сюда сложим DataFrame для каждого ratio

    # Главный цикл генерации
    for noize_level in NOIZE_LEVELS_TO_GENERATE:
        for ratio_combination in sum_to_one_combinations:
            # Генерим cores
            cores, labels, noize_labels = generate_cores(
                motifs=motifs,
                n=12000,
                ratio_combination=ratio_combination,
                noize_level=noize_level
            )

            # Генерим полные пептиды
            df = generate_peptides(motifs, cores, labels, noize_labels)

            # Путь для CSV
            out_path = os.path.join(
                RESULT_DATA_PATH,
                f"sim_data_motifs_{len(motifs)}_ratio_{ratio_combination}_lengths_{lengths_generated}_noize_{noize_level}.csv"
            )

            # Сохраняем
            df.to_csv(out_path, sep=";", index=False)

            # Запоминаем в results
            results[ratio_combination] = df

            print(f"[SAVED] {out_path}")
            print(f"Allele counts:\n{df.allele.value_counts()}")
            print(f"Noise counts:\n{df.noize.value_counts()}")
            print("-" * 80)

    # Печатаем, какие ratio были
    print("Generated ratio keys:", list(results.keys())[:10], "..." if len(results) > 10 else "")
    print()

    # Возьмём последний df (как у тебя в исходнике)
    # ВНИМАНИЕ: если sum_to_one_combinations пустой — df не существует, но у нас обычно не пустой.
    df_last = df

    # ====== Примеры визуализаций: сохраняем графики в LOGO_PATH ======
    plt.figure()
    make_logo_restricted(df_last, peptide_len=15, core_start=0)
    plt.savefig(os.path.join(LOGO_PATH, "logo_peptidelen_15_corestart_0.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    make_logo_for_data(df_last.core.tolist(), "combined core")
    plt.savefig(os.path.join(LOGO_PATH, "logo_core_combined.png"), bbox_inches="tight")
    plt.close()

    # logo только для noise core
    noise_df = df_last[df_last.noize == 1]
    if len(noise_df) > 0:
        plt.figure()
        make_logo_for_data(noise_df.core.tolist(), "combined core (noise only)")
        plt.savefig(os.path.join(LOGO_PATH, "logo_core_combined_noise_only.png"), bbox_inches="tight")
        plt.close()

    # logo по каждому мотиву
    alleles_unique = np.unique(df_last.allele)
    for i, allele in enumerate(alleles_unique):
        plt.figure()
        make_logo_for_data(df_last[df_last.allele == allele].core.tolist(), f"motif{i} (all)")
        plt.savefig(os.path.join(LOGO_PATH, f"logo_motif_{i}_all.png"), bbox_inches="tight")
        plt.close()

    for i, allele in enumerate(alleles_unique):
        plt.figure()
        make_logo_for_data(df_last[(df_last.allele == allele) & (df_last.noize == 0)].core.tolist(), f"motif{i} (no noise)")
        plt.savefig(os.path.join(LOGO_PATH, f"logo_motif_{i}_no_noise.png"), bbox_inches="tight")
        plt.close()

    # ====== Диаграммы распределений ======
    target_length = 16
    motif = 0

    d_start = start_distributions[motif][target_length]
    samples = np.array(d_start.sample(10000), dtype=int)
    bins = list(d_start.parameters[0].keys())
    plt.figure()
    sns.histplot(x=samples, discrete=True).set(title=f"Start distribution (motif={motif}, peptide_len={target_length})")
    plt.xticks(bins)
    plt.savefig(os.path.join(LOGO_PATH, f"start_distribution_motif_{motif}_len_{target_length}.png"), bbox_inches="tight")
    plt.close()

    samples_len = np.array(length_distributions[motif].sample(10000), dtype=int)
    bins_len = list(length_distributions[motif].parameters[0].keys())
    plt.figure()
    sns.histplot(x=samples_len, discrete=True).set(title=f"Length distribution (motif={motif})")
    plt.xticks(bins_len)
    plt.savefig(os.path.join(LOGO_PATH, f"length_distribution_motif_{motif}.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
