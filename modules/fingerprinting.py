import numpy as np
import torch
import os
import json
import logging
from tqdm import tqdm
from mace.calculators import MACECalculator
from typing import List, Dict, Tuple
from .configuration import Configuration

def process_dataset(
    dataset_config: Dict,
    model_path: str,
    type_map: Dict[int, str],
    device: str
) -> Tuple[int, np.ndarray, List[Dict]]: # Возвращаем КОЛИЧЕСТВО конфигов, а не сами конфиги
    """
    Обрабатывает один датасет: ... и возвращает общее число конфигов и прореженные данные.
    """
    path = dataset_config['path']
    cache_path = dataset_config['fingerprints_cache']
    sampling_rate = dataset_config['sampling_rate_N']
    meta_cache_path = cache_path + '.meta.json'

    logging.info(f"--- Обработка датасета: {os.path.basename(path)} ---")
    
    # 1. Получаем общее количество конфигураций БЕЗ загрузки в память
    total_configs_in_file = Configuration.get_config_count(path)
    if total_configs_in_file == 0:
        return 0, np.array([]), []

    # 2. Логика работы с кэшем (без изменений)
    fingerprints = None
    if os.path.exists(cache_path) and os.path.exists(meta_cache_path):
        logging.info(f"Найден кэш: {cache_path}")
        with open(meta_cache_path, 'r') as f:
            meta = json.load(f)
        
        n_counted = meta.get('N_counted', 1)
        
        if sampling_rate % n_counted == 0:
            logging.info(f"Кэш валиден (N_counted={n_counted}, sampling_rate={sampling_rate}). Загрузка и дополнительное прореживание...")
            sub_sampling_rate = sampling_rate // n_counted
            
            # Загружаем кэш и прореживаем его
            cached_fps = np.load(cache_path)
            
            # Создаем маску для прореживания
            # Индекс конфигурации в кэше хранится в последнем столбце
            config_indices_in_cache = cached_fps[:, -1]
            mask = (config_indices_in_cache % sub_sampling_rate == 0)
            
            fingerprints = cached_fps[mask]
            
            # Обновляем индексы конфигураций в прореженном кэше, чтобы они были последовательными
            unique_ids = np.unique(fingerprints[:, -1])
            id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
            fingerprints[:, -1] = np.vectorize(id_map.get)(fingerprints[:, -1])

        else:
            logging.warning(f"Кэш НЕ валиден (sampling_rate={sampling_rate} не кратен N_counted={n_counted}). Будет произведен пересчет.")
            
    if fingerprints is None:
        logging.info(f"Генерация фингерпринтов с прореживанием N={sampling_rate}...")
        
        # ЗАГРУЖАЕМ В ПАМЯТЬ ТОЛЬКО ПРОРЕЖЕННУЮ ВЫБОРКУ
        # Это временная и управляемая нагрузка на память
        all_configs_in_file = Configuration.from_file(path)
        thinned_configs_to_process = all_configs_in_file[::sampling_rate]
        
        fingerprints = _generate_mace_fingerprints_for_list(
            thinned_configs_to_process, model_path, type_map, device
        )
        
        logging.info(f"Сохранение нового кэша: {cache_path}")
        np.save(cache_path, fingerprints)
        with open(meta_cache_path, 'w') as f:
            json.dump({'source_path': path, 'N_counted': sampling_rate}, f)

    # 3. Создание метаданных для прореженных конфигураций
    thinned_configs_metadata = []
    original_indices = np.arange(0, total_configs_in_file, sampling_rate)
    
    for i, original_idx in enumerate(original_indices):
        thinned_configs_metadata.append({
            'original_path': path,
            'original_start_idx': int(original_idx),
            'sampling_rate': sampling_rate,
            # 'config_object' больше не храним
        })
        
    return total_configs_in_file, fingerprints, thinned_configs_metadata


def _generate_mace_fingerprints_for_list(
    configurations: List[Configuration],
    model_path: str,
    type_map: Dict[int, str],
    device: str
) -> np.ndarray:
    """
    Внутренняя функция: превращает СПИСОК объектов Configuration в массив фингерпринтов.
    """
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype='float64')
    all_fingerprints = []

    for config_idx, config in enumerate(tqdm(configurations, desc="Расчет фингерпринтов", leave=False)):
        if not config.atom_data:
            continue
            
        atoms = config.to_ase(type_map)
        descriptors = calc.get_descriptors(atoms)
        
        config_indices = np.full((descriptors.shape[0], 1), config_idx)
        combined_data = np.hstack([descriptors, config_indices])
        all_fingerprints.append(combined_data)

    if not all_fingerprints:
        return np.empty((0, calc.get_descriptors(Atoms('H')).shape[1] + 1))

    return np.vstack(all_fingerprints)