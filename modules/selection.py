import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import umap
from mace.calculators import MACECalculator
from typing import List, Dict
from .utils import smiles_to_ase_atoms

def generate_relevant_dataset(
    smiles: str,
    all_configs: List, # List[Configuration]
    ref_fingerprints: np.ndarray,
    selection_params: Dict,
    fp_params: Dict,
    query_atoms_list: List, #List[ase.Atoms]
) -> (List, np.ndarray): # Возвращаем и конфиги, и фингерпринты
    """
    Основная функция, выполняющая весь пайплайн отбора данных.
    """
    # ... (Шаг 1 и 2 без изменений) ...
    # --- Шаг 1 ---
    
    
    logging.info("Расчет фингерпринтов для query-структуры...")
    calc = MACECalculator(model_paths=fp_params['mace_model_path'], device=fp_params['device'])
    query_fp_raw_list = [calc.get_descriptors(atoms) for atoms in query_atoms_list]
    query_fp_raw = np.vstack(query_fp_raw_list)

    # --- Шаг 2: Подготовка и нормализация данных ---
    logging.info("Подготовка и нормализация данных...")
    ref_fp_only = ref_fingerprints[:, :-1]
    ref_config_indices = ref_fingerprints[:, -1].astype(int)

    scaler = StandardScaler().fit(ref_fp_only)
    ref_fp_scaled = scaler.transform(ref_fp_only)
    query_fp_scaled = scaler.transform(query_fp_raw)
    
    # --- Шаг 3: Внутриконфигурационная кластеризация (как в ноутбуке) ---
    logging.info("Этап 1: Внутриконфигурационная кластеризация...")
    n_configs = len(all_configs)
    n_clusters_per_config = selection_params['intra_config_clusters']
    fp_centroids_list = []
    
    for i in tqdm(range(n_configs), desc="Кластеризация конфигураций"):
        mask = (ref_config_indices == i)
        if not np.any(mask): continue
        
        config_fps_scaled = ref_fp_scaled[mask]
        
        # Убедимся, что атомов достаточно для кластеризации
        n_atoms = config_fps_scaled.shape[0]
        n_c = min(n_atoms, n_clusters_per_config)
        
        if n_c < 1: continue

        clusterizer = AgglomerativeClustering(n_clusters=n_c).fit(config_fps_scaled)
        
        for label in range(n_c):
            centroid = np.mean(config_fps_scaled[clusterizer.labels_ == label], axis=0)
            fp_centroids_list.append(centroid)

    fp_centroids = np.array(fp_centroids_list)
    
    # --- Шаг 4 (UMAP и финальная кластеризация) ---
    logging.info("Этап 2: Понижение размерности с помощью UMAP...")
    reducer = umap.UMAP(**selection_params['umap_params']).fit(fp_centroids)
    embedding_cl = reducer.transform(fp_centroids)
    
    logging.info("Этап 3: Финальная кластеризация центроидов...")
    final_clusterizer = AgglomerativeClustering(**selection_params['clustering_params']).fit(embedding_cl)
    labels = final_clusterizer.labels_
    
    # +++ НОВОЕ: ЛОГИРОВАНИЕ СТАТИСТИКИ ПО КЛАСТЕРАМ +++
    unique_labels, counts = np.unique(labels, return_counts=True)
    logging.info(f"Найдено {len(unique_labels)} уникальных кластеров в референсном датасете.")
    for label, count in zip(unique_labels, counts):
        logging.info(f"  - Кластер {label}: {count} центроидов конфигураций")
    
    # --- Шаг 5: Поиск релевантных кластеров ---
    logging.info("Этап 4: Определение релевантных кластеров...")
    clf = NearestCentroid().fit(embedding_cl, labels)
    
    # Кластеризуем query-фингерпринты, чтобы получить query-центроиды
    query_clusterizer = AgglomerativeClustering(n_clusters=min(query_fp_scaled.shape[0], n_clusters_per_config)).fit(query_fp_scaled)
    query_centroids_scaled = np.array([np.mean(query_fp_scaled[query_clusterizer.labels_ == l], axis=0) for l in range(query_clusterizer.n_clusters_)])
    
    query_embedding = reducer.transform(query_centroids_scaled)
    target_labels = np.unique(clf.predict(query_embedding))
    logging.info(f"Целевые кластеры, релевантные для SMILES: {target_labels.tolist()}")
    
    # --- Шаг 6: Отбор конфигураций ---
    logging.info("Этап 5: Отбор релевантных конфигураций...")
    # Находим, каким исходным центроидам соответствуют целевые кластеры
    relevant_centroid_indices = np.where(np.isin(labels, target_labels))[0]
    
    # Теперь нужно сопоставить эти индексы с исходными конфигурациями. 
    # Это самая сложная часть, если индексы были потеряны. 
    # Перестроим логику, чтобы сохранить связь.

    # Обновленная логика для шагов 3-6 для сохранения связей
    # (Более простой и надежный подход)
    
    # Индексируем центроиды по их исходным конфигурациям
    centroid_to_config_map = []
    current_centroid_idx = 0
    for i in range(n_configs):
        mask = (ref_config_indices == i)
        if not np.any(mask): continue
        n_atoms = np.sum(mask)
        n_c = min(n_atoms, n_clusters_per_config)
        if n_c < 1: continue
        for _ in range(n_c):
            centroid_to_config_map.append(i)
        
    relevant_config_ids = set()
    for centroid_idx in relevant_centroid_indices:
        config_id = centroid_to_config_map[centroid_idx]
        relevant_config_ids.add(config_id)

    logging.info(f"Найдено {len(relevant_config_ids)} уникальных релевантных конфигураций до финальной выборки.")
    
    # --- Шаг 7: Финальная выборка ---
    num_to_select = selection_params['num_output_configs']
    if len(relevant_config_ids) < num_to_select:
        logging.warning(f"Найдено меньше конфигураций ({len(relevant_config_ids)}), чем запрошено ({num_to_select}). Будут использованы все найденные.")
        final_indices = list(relevant_config_ids)
    else:
        # Простое случайное сэмплирование для начала
        final_indices = np.random.choice(list(relevant_config_ids), num_to_select, replace=False)

    logging.info(f"Отобрано {len(final_indices)} конфигураций для итогового датасета.")
    
    final_configurations = [all_configs[i] for i in final_indices]
    return final_configurations, query_fp_raw, query_atoms_list