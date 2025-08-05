import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import umap
from mace.calculators import MACECalculator
from typing import List, Dict
from collections import defaultdict

from .utils import smiles_to_ase_atoms

def generate_relevant_dataset(
    smiles: str,
    source_datasets_info: List[Dict],
    thinned_configs_metadata: List[Dict],
    ref_fingerprints: np.ndarray,
    selection_params: Dict,
    fp_params: Dict,
    query_atoms_list: List
) -> (List[Dict], np.ndarray):
    """
    Основная функция, выполняющая весь пайплайн отбора данных с детерминированной выборкой.
    """
    # --- Шаги 1-4: Генерация FP, нормализация, кластеризация, UMAP ---
    # Эта часть остается без изменений, она подготавливает все необходимые данные.
    
    logging.info("Расчет фингерпринтов для query-структур...")
    calc = MACECalculator(model_paths=fp_params['mace_model_path'], device=fp_params['device'])
    query_fp_raw_list = [calc.get_descriptors(atoms) for atoms in query_atoms_list]
    query_fp_raw = np.vstack(query_fp_raw_list)

    logging.info("Подготовка и нормализация данных...")
    ref_fp_only = ref_fingerprints[:, :-1]
    ref_thinned_config_indices = ref_fingerprints[:, -1].astype(int)
    scaler = StandardScaler().fit(ref_fp_only)
    ref_fp_scaled = scaler.transform(ref_fp_only)
    query_fp_scaled = scaler.transform(query_fp_raw)

    logging.info("Этап 1: Внутриконфигурационная кластеризация...")
    n_thinned_configs = len(thinned_configs_metadata)
    n_clusters_per_config = selection_params['intra_config_clusters']
    fp_centroids_list = []
    centroid_to_thinned_config_map = []
    for i in tqdm(range(n_thinned_configs), desc="Кластеризация конфигураций"):
        mask = (ref_thinned_config_indices == i)
        if not np.any(mask): continue
        config_fps_scaled = ref_fp_scaled[mask]
        n_atoms = config_fps_scaled.shape[0]
        n_c = min(n_atoms, n_clusters_per_config)
        if n_c < 1: continue
        clusterizer = AgglomerativeClustering(n_clusters=n_c).fit(config_fps_scaled)
        for label in range(n_c):
            centroid = np.mean(config_fps_scaled[clusterizer.labels_ == label], axis=0)
            fp_centroids_list.append(centroid)
            centroid_to_thinned_config_map.append(i)

    fp_centroids = np.array(fp_centroids_list)
    
    logging.info("Этап 2: Понижение размерности с помощью UMAP...")
    reducer = umap.UMAP(**selection_params['umap_params']).fit(fp_centroids)
    embedding_cl = reducer.transform(fp_centroids)
    
    logging.info("Этап 3: Финальная кластеризация центроидов...")
    final_clusterizer = AgglomerativeClustering(**selection_params['clustering_params']).fit(embedding_cl)
    labels = final_clusterizer.labels_
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    logging.info(f"Найдено {len(unique_labels)} уникальных кластеров в референсном датасете.")
    
    # --- Шаг 5: Поиск релевантных кластеров и расчет метрик релевантности ---
    logging.info("Этап 4: Определение релевантных кластеров и расчет метрик...")
    clf = NearestCentroid().fit(embedding_cl, labels)
    
    query_clusterizer = AgglomerativeClustering(n_clusters=min(query_fp_scaled.shape[0], n_clusters_per_config)).fit(query_fp_scaled)
    query_centroids_scaled = np.array([np.mean(query_fp_scaled[query_clusterizer.labels_ == l], axis=0) for l in range(query_clusterizer.n_clusters_)])
    query_embedding = reducer.transform(query_centroids_scaled)
    
    # Группируем query-центроиды по предсказанным кластерам
    predicted_labels = clf.predict(query_embedding)
    target_labels = np.unique(predicted_labels)
    
    query_centroids_by_cluster = defaultdict(list)
    for i, label in enumerate(predicted_labels):
        query_centroids_by_cluster[label].append(query_embedding[i])

    # --- НОВАЯ ЛОГИКА ---
    # --- Шаг 6: Детерминированный отбор конфигураций ---
    logging.info("Этап 5: Детерминированный отбор релевантных конфигураций...")
    
    # 1. Для каждого релевантного кластера находим все его конфигурации и считаем их релевантность
    candidates_by_cluster = defaultdict(list)
    for i, label in enumerate(labels):
        if label in target_labels:
            db_centroid_emb = embedding_cl[i]
            # Считаем среднее расстояние до релевантных query-центроидов
            distance = np.mean([np.linalg.norm(db_centroid_emb - qc_emb) for qc_emb in query_centroids_by_cluster[label]])
            
            thinned_idx = centroid_to_thinned_config_map[i]
            candidates_by_cluster[label].append({
                "thinned_idx": thinned_idx,
                "distance": distance
            })
            
    # 2. Сортируем кандидатов внутри каждого кластера по релевантности (возрастанию расстояния)
    for label in candidates_by_cluster:
        # Убираем дубликаты по thinned_idx, оставляя запись с минимальным расстоянием
        unique_candidates = {}
        for cand in candidates_by_cluster[label]:
            idx = cand["thinned_idx"]
            if idx not in unique_candidates or cand["distance"] < unique_candidates[idx]["distance"]:
                unique_candidates[idx] = cand
        
        sorted_unique = sorted(unique_candidates.values(), key=lambda x: x["distance"])
        candidates_by_cluster[label] = sorted_unique

    # 3. Пропорциональный отбор "по кругу"
    num_to_select = selection_params['num_output_configs']
    final_selection_metadata = []
    
    # Оценочное количество конфигураций на блок
    avg_sampling_rate = np.mean([meta['sampling_rate'] for meta in thinned_configs_metadata]) if thinned_configs_metadata else 1
    
    # Итераторы для каждого кластера
    iterators = {label: iter(candidates) for label, candidates in candidates_by_cluster.items()}
    
    total_configs_selected = 0
    while total_configs_selected < num_to_select:
        added_in_round = False
        for label in target_labels:
            try:
                candidate = next(iterators[label])
                # Добавляем метаданные выбранного блока
                final_selection_metadata.append({**candidate, "label": label})
                
                # Прибавляем "вес" этого блока
                thinned_idx = candidate["thinned_idx"]
                total_configs_selected += thinned_configs_metadata[thinned_idx]['sampling_rate']
                added_in_round = True
                
                if total_configs_selected >= num_to_select:
                    break
            except StopIteration:
                # В этом кластере кандидаты закончились
                continue
        
        if not added_in_round:
            logging.warning("Все релевантные кандидаты исчерпаны.")
            break

    # 4. Логирование новой метрики
    log_messages = []
    selected_by_cluster_for_log = defaultdict(list)
    for item in final_selection_metadata:
        selected_by_cluster_for_log[item['label']].append(item['distance'])
        
    for label in target_labels:
        distances = selected_by_cluster_for_log.get(label)
        if distances:
            avg_dist = np.mean(distances)
            count = len(distances)
            log_messages.append(
                f"  - Кластер {label}: отобрано {count} блоков. Ср. расстояние до query-центроидов = {avg_dist:.4f}"
            )

    logging.info("Статистика по отобранным кластерам:\n" + "\n".join(log_messages))
    
    # 5. Расширение выборки до полного набора
    final_config_identifiers = []
    for item in final_selection_metadata:
        meta = thinned_configs_metadata[item['thinned_idx']]
        start_idx = meta['original_start_idx']
        sampling_rate = meta['sampling_rate']
        source_path = meta['original_path']
        
        total_in_file = [info['count'] for info in source_datasets_info if info['path'] == source_path][0]
        
        for i in range(sampling_rate):
            current_idx = start_idx + i
            if current_idx < total_in_file:
                final_config_identifiers.append({
                    'source_path': source_path,
                    'index_in_file': current_idx
                })

    logging.info(f"Сгенерировано {len(final_config_identifiers)} идентификаторов релевантных конфигураций.")
    
    # Обрезаем до нужного размера
    if len(final_config_identifiers) > num_to_select:
        final_config_identifiers = final_config_identifiers[:num_to_select]
        logging.info(f"Список идентификаторов обрезан до требуемого размера: {len(final_config_identifiers)}.")
        
    return final_config_identifiers, query_fp_raw


