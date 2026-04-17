"""
Скрипт для анализа сохраненных данных о скорах пептидов и моделях.

Использование:
    python analyze_saved_scores.py --layer 0 --allele <allele_name>
    python analyze_saved_scores.py --layer 0 --allele <allele_name> --find-clusters
"""

import argparse
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
import numpy as np


def load_scores_data(results_path, layer_num, allele_name):
    """Загружает данные о скорах для указанного слоя и аллеля."""
    layer_dir = f"{results_path}/root/layer_{layer_num}/"
    
    # Загрузить скоры
    scores_csv = f"{layer_dir}/{allele_name}_scores.csv"
    scores_pkl = f"{layer_dir}/{allele_name}_scores.pkl"
    
    if os.path.exists(scores_pkl):
        df = pd.read_pickle(scores_pkl)
    elif os.path.exists(scores_csv):
        df = pd.read_csv(scores_csv)
    else:
        raise FileNotFoundError(f"Не найдены файлы со скорами в {layer_dir}")
    
    # Загрузить информацию о моделях
    models_info_path = f"{layer_dir}/models_info.json"
    if os.path.exists(models_info_path):
        with open(models_info_path, 'r') as f:
            models_info = json.load(f)
    else:
        models_info = None
        print(f"Предупреждение: не найдена информация о моделях в {models_info_path}")
    
    # Загрузить UMAP координаты
    umap_csv = f"{layer_dir}/{allele_name}_umap_coordinates.csv"
    umap_pkl = f"{layer_dir}/{allele_name}_umap_coordinates.pkl"
    
    umap_df = None
    if os.path.exists(umap_pkl):
        umap_df = pd.read_pickle(umap_pkl)
    elif os.path.exists(umap_csv):
        umap_df = pd.read_csv(umap_csv)
    
    return df, models_info, umap_df


def print_models_info(models_info):
    """Выводит информацию о моделях."""
    if models_info is None:
        print("Информация о моделях недоступна")
        return
    
    print("\n=== ИНФОРМАЦИЯ О МОДЕЛЯХ ===")
    for model_key, info in models_info.items():
        print(f"{model_key}: {info['model_name']}")


def analyze_scores(df, models_info=None):
    """Анализирует скоры пептидов."""
    print("\n=== АНАЛИЗ СКОРОВ ===")
    print(f"Всего пептидов: {len(df)}")
    
    # Найти колонки со скорами
    score_cols = [col for col in df.columns if col.startswith('model_')]
    print(f"Количество моделей: {len(score_cols)}")
    
    # Статистика по скорам
    print("\nСтатистика по скорам:")
    print(df[score_cols].describe())
    
    # Топ пептиды по лучшей модели
    if 'model_0' in df.columns:
        print(f"\nТоп-10 пептидов по model_0:")
        top_peptides = df.nlargest(10, 'model_0')[['peptide', 'model_0']]
        print(top_peptides.to_string(index=False))
    
    # Корреляции между моделями
    if len(score_cols) > 1:
        print("\nКорреляции между моделями:")
        corr_matrix = df[score_cols].corr()
        print(corr_matrix.round(3))
    
    return score_cols


def find_clusters_kmeans(umap_df, n_clusters=5):
    """Находит кластеры используя KMeans на UMAP координатах."""
    if umap_df is None:
        print("UMAP координаты недоступны для кластеризации")
        return None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(umap_df[['x', 'y']])
    umap_df['cluster'] = clusters
    
    print(f"\n=== КЛАСТЕРИЗАЦИЯ KMEANS (n_clusters={n_clusters}) ===")
    print(f"Размеры кластеров:")
    print(umap_df['cluster'].value_counts().sort_index())
    
    return umap_df


def find_clusters_dbscan(umap_df, eps=0.5, min_samples=10):
    """Находит кластеры используя DBSCAN на UMAP координатах."""
    if umap_df is None:
        print("UMAP координаты недоступны для кластеризации")
        return None
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(umap_df[['x', 'y']])
    umap_df['cluster'] = clusters
    
    print(f"\n=== КЛАСТЕРИЗАЦИЯ DBSCAN (eps={eps}, min_samples={min_samples}) ===")
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"Найдено кластеров: {n_clusters}")
    print(f"Шумовых точек: {n_noise}")
    print(f"Размеры кластеров:")
    print(umap_df['cluster'].value_counts().sort_index())
    
    return umap_df


def visualize_clusters(umap_df, save_path=None):
    """Визуализирует кластеры на UMAP координатах."""
    if umap_df is None or 'cluster' not in umap_df.columns:
        print("Нет данных для визуализации кластеров")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(umap_df['x'], umap_df['y'], 
                        c=umap_df['cluster'], 
                        cmap='tab10', 
                        s=20, 
                        alpha=0.6)
    ax.set_xlabel('UMAP координата X')
    ax.set_ylabel('UMAP координата Y')
    ax.set_title('Кластеры пептидов на UMAP координатах')
    plt.colorbar(scatter, ax=ax, label='Кластер')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nВизуализация сохранена в: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_clustered_data(df, umap_df, models_info, output_dir, allele_name):
    """Сохраняет данные с информацией о кластерах."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Объединить данные
    if umap_df is not None and 'cluster' in umap_df.columns:
        # Объединить по пептидам
        merged_df = df.merge(
            umap_df[['peptide', 'x', 'y', 'cluster']], 
            on='peptide', 
            how='left'
        )
        
        # Сохранить
        merged_df.to_csv(f"{output_dir}/{allele_name}_with_clusters.csv", index=False)
        merged_df.to_pickle(f"{output_dir}/{allele_name}_with_clusters.pkl")
        
        # Сохранить по кластерам
        for cluster_id in sorted(merged_df['cluster'].unique()):
            if pd.notna(cluster_id):
                cluster_data = merged_df[merged_df['cluster'] == cluster_id]
                cluster_file = f"{output_dir}/{allele_name}_cluster_{int(cluster_id)}.csv"
                cluster_data.to_csv(cluster_file, index=False)
                print(f"Кластер {int(cluster_id)}: {len(cluster_data)} пептидов -> {cluster_file}")
        
        print(f"\nДанные с кластерами сохранены в: {output_dir}")
    else:
        print("Нет информации о кластерах для сохранения")


def main():
    parser = argparse.ArgumentParser(description='Анализ сохраненных данных о скорах')
    parser.add_argument('--results-path', type=str, required=True,
                       help='Путь к директории с результатами эксперимента')
    parser.add_argument('--layer', type=int, default=0,
                       help='Номер слоя для анализа (по умолчанию 0)')
    parser.add_argument('--allele', type=str, required=True,
                       help='Имя аллеля для анализа')
    parser.add_argument('--find-clusters', action='store_true',
                       help='Найти кластеры пептидов')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Количество кластеров для KMeans (по умолчанию 5)')
    parser.add_argument('--dbscan-eps', type=float, default=0.5,
                       help='eps параметр для DBSCAN (по умолчанию 0.5)')
    parser.add_argument('--dbscan-min-samples', type=int, default=10,
                       help='min_samples параметр для DBSCAN (по умолчанию 10)')
    parser.add_argument('--method', type=str, choices=['kmeans', 'dbscan', 'both'], 
                       default='kmeans',
                       help='Метод кластеризации (по умолчанию kmeans)')
    parser.add_argument('--save-clusters', action='store_true',
                       help='Сохранить данные с кластерами')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Директория для сохранения результатов (по умолчанию та же что и входная)')
    
    args = parser.parse_args()
    
    # Загрузить данные
    print(f"Загрузка данных для layer {args.layer}, allele {args.allele}...")
    df, models_info, umap_df = load_scores_data(
        args.results_path, 
        args.layer, 
        args.allele
    )
    
    # Вывести информацию о моделях
    print_models_info(models_info)
    
    # Анализ скоров
    score_cols = analyze_scores(df, models_info)
    
    # Найти кластеры если нужно
    if args.find_clusters:
        output_dir = args.output_dir or f"{args.results_path}/root/layer_{args.layer}/"
        
        if args.method in ['kmeans', 'both']:
            umap_df = find_clusters_kmeans(umap_df, n_clusters=args.n_clusters)
            if umap_df is not None:
                visualize_clusters(
                    umap_df, 
                    save_path=f"{output_dir}/{args.allele}_kmeans_clusters.png"
                )
        
        if args.method in ['dbscan', 'both']:
            umap_df = find_clusters_dbscan(
                umap_df, 
                eps=args.dbscan_eps, 
                min_samples=args.dbscan_min_samples
            )
            if umap_df is not None:
                visualize_clusters(
                    umap_df, 
                    save_path=f"{output_dir}/{args.allele}_dbscan_clusters.png"
                )
        
        # Сохранить данные с кластерами
        if args.save_clusters and umap_df is not None:
            save_clustered_data(df, umap_df, models_info, output_dir, args.allele)
    
    print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")


if __name__ == '__main__':
    main()
