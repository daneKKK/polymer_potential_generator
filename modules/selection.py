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
    source_datasets_info: List[Dict], # Новая структура с информацией о датасетах
    thinned_configs_metadata: List[Dict],
    ref_fingerprints: np.ndarray,
    selection_params: Dict,
    fp_params: Dict,
    query_atoms_list: List # Configuration
) -> (List[Dict], np.ndarray, List): # Возвращаем список словарей-идентификаторов
    """
    Основная функция, выполняющая весь пайплайн отбора данных.
    """
    # --- Шаг 1: Получение фингерпринтов для query-структуры из SMILES ---
    #logging.info(f"Генерация query-структур (мономер и кольца) для SMILES: {smiles}")
    #query_atoms_list = smiles_to_ase_atoms(smiles)
    if not query_atoms_list:
        raise RuntimeError("Не удалось сгенерировать ни одной query-структуры из SMILES.")

    logging.info("Расчет фингерпринтов для query-структур...")
    calc = MACECalculator(model_paths=fp_params['mace_model_path'], device=fp_params['device'])
    
    query_fp_raw_list = []
    for atoms in query_atoms_list:
        logging.info(f"  -> Обработка структуры '{atoms.info.get('name', 'N/A')}' с {len(atoms)} атомами.")
        query_fp_raw_list.append(calc.get_descriptors(atoms))
    
    query_fp_raw = np.vstack(query_fp_raw_list)
    
    # --- Шаг 2: Подготовка и нормализация данных ---
    # ... (эта часть остается почти такой же) ...
    ref_fp_only = ref_fingerprints[:, :-1]
    # Теперь этот индекс - это индекс в прореженной выборке
    ref_thinned_config_indices = ref_fingerprints[:, -1].astype(int)

    scaler = StandardScaler().fit(ref_fp_only)
    ref_fp_scaled = scaler.transform(ref_fp_only)
    query_fp_scaled = scaler.transform(query_fp_raw)
    
    # --- Шаг 3: Внутриконфигурационная кластеризация ---
    logging.info("Этап 1: Внутриконфигурационная кластеризация...")
    n_thinned_configs = len(thinned_configs_metadata)
    n_clusters_per_config = selection_params['intra_config_clusters']
    fp_centroids_list = []
    centroid_to_thinned_config_map = [] # Карта: индекс центроида -> индекс в thinned_configs_metadata

    for i in tqdm(range(n_thinned_configs), desc="Кластеризация конфигураций"):
        mask = (ref_thinned_config_indices == i)
        if not np.any(mask): continue
        
        # ... (логика кластеризации одной конфигурации остается той же) ...
        config_fps_scaled = ref_fp_scaled[mask]
        n_atoms = config_fps_scaled.shape[0]
        n_c = min(n_atoms, n_clusters_per_config)
        if n_c < 1: continue

        clusterizer = AgglomerativeClustering(n_clusters=n_c).fit(config_fps_scaled)
        
        for label in range(n_c):
            centroid = np.mean(config_fps_scaled[clusterizer.labels_ == label], axis=0)
            fp_centroids_list.append(centroid)
            # Запоминаем, что этот центроид относится к i-й конфигурации в прореженной выборке
            centroid_to_thinned_config_map.append(i)

    fp_centroids = np.array(fp_centroids_list)
    
    # --- Шаг 4 (UMAP и финальная кластеризация) ---
    logging.info("Этап 2: Понижение размерности с помощью UMAP...")
    reducer = umap.UMAP(**selection_params['umap_params']).fit(fp_centroids)
    embedding_cl = reducer.transform(fp_centroids)
    
    logging.info("Этап 3: Финальная кластеризация центроидов...")
    final_clusterizer = AgglomerativeClustering(**selection_params['clustering_params']).fit(embedding_cl)
    labels = final_clusterizer.labels_
    
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
   
    # Сначала найдем для каждого query-центроида его ближайший референсный кластер и расстояние до его центра
    predicted_labels_per_query = clf.predict(query_embedding)
    cluster_distances = {} # Словарь для хранения минимального расстояния до каждого целевого кластера
    
    for i, query_point_emb in enumerate(query_embedding):
        label = predicted_labels_per_query[i]
        ref_centroid_emb = clf.centroids_[label]
        distance = np.linalg.norm(query_point_emb - ref_centroid_emb)
        
        # Сохраняем минимальное расстояние для каждого кластера
        if label not in cluster_distances or distance < cluster_distances[label]:
            cluster_distances[label] = distance

    log_messages = []
    for label in target_labels:
        # Находим все точки (центроиды конфигураций) в этом кластере
        points_in_cluster = fp_centroids[labels == label]
        
        # Считаем дисперсию (размер) кластера. Используем не embedding, а исходные fp_centroids
        if points_in_cluster.shape[0] > 1:
            # Средняя дисперсия по всем измерениям
            dispersion = np.mean(np.var(points_in_cluster, axis=0))
            # "Радиус" или "размер" кластера
            radius = np.sqrt(dispersion)
        else:
            radius = 1e-6 # Избегаем деления на ноль для кластеров из одной точки
        
        min_dist_to_cluster = cluster_distances.get(label, 0.0)
        relevance_metric = min_dist_to_cluster / radius

        log_messages.append(
            f"  - Кластер {label}: {counts[list(unique_labels).index(label)]} конфиг., метрика близости = {relevance_metric:.3f}"
        )
    
    logging.info("Целевые кластеры, релевантные для SMILES:\n" + "\n".join(log_messages))
    
    # --- Шаг 6: Отбор конфигураций ---
    logging.info("Этап 5: Отбор релевантных конфигураций...")
    
    # Находим индексы релевантных центроидов
    relevant_centroid_indices = np.where(np.isin(labels, target_labels))[0]
    
    # По карте находим индексы релевантных ПРОРЕЖЕННЫХ конфигураций
    relevant_thinned_config_indices = set()
    for centroid_idx in relevant_centroid_indices:
        thinned_config_idx = centroid_to_thinned_config_map[centroid_idx]
        relevant_thinned_config_indices.add(thinned_config_idx)

    logging.info(f"Найдено {len(relevant_thinned_config_indices)} релевантных блоков конфигураций до финальной выборки.")
    
    # --- Шаг 7: Финальная выборка и РАСШИРЕНИЕ ---
    num_to_select = selection_params['num_output_configs']
    
    # Выбираем ИНДЕКСЫ из прореженной выборки
    if len(relevant_thinned_config_indices) * np.mean([meta['sampling_rate'] for meta in thinned_configs_metadata]) < num_to_select:
        logging.warning(f"Найдено меньше конфигураций, чем запрошено. Будут использованы все найденные.")
        final_thinned_indices = list(relevant_thinned_config_indices)
    else:
        # Случайная выборка БЛОКОВ
        final_thinned_indices = np.random.choice(
            list(relevant_thinned_config_indices), 
            # Приблизительно выбираем нужное число блоков
            int(num_to_select / np.mean([meta['sampling_rate'] for meta in thinned_configs_metadata])) + 1, 
            replace=False
        )

    final_config_identifiers = []
    for thinned_idx in final_thinned_indices:
        meta = thinned_configs_metadata[thinned_idx]
        start_idx = meta['original_start_idx']
        sampling_rate = meta['sampling_rate']
        source_path = meta['original_path']
        
        # Находим, сколько всего конфигураций в этом файле, чтобы не выйти за границу
        total_in_file = 0
        for info in source_datasets_info:
            if info['path'] == source_path:
                total_in_file = info['count']
                break
        
        # Добавляем "адреса" всех конфигураций в блоке
        for i in range(sampling_rate):
            current_idx = start_idx + i
            if current_idx < total_in_file:
                final_config_identifiers.append({
                    'source_path': source_path,
                    'index_in_file': current_idx
                })

    logging.info(f"Сгенерировано {len(final_config_identifiers)} идентификаторов релевантных конфигураций.")
    
    # Обрезаем до нужного размера, если выбрали слишком много
    if len(final_config_identifiers) > num_to_select:
        final_config_identifiers = final_config_identifiers[:num_to_select]
        logging.info(f"Список идентификаторов обрезан до требуемого размера: {len(final_config_identifiers)}.")

    return final_config_identifiers, query_fp_raw
