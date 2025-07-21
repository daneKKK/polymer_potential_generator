import numpy as np
import torch
import os
from tqdm import tqdm
from mace.calculators import MACECalculator
from typing import List, Dict
from .configuration import Configuration

def get_or_create_fingerprints(
    configs: List[Configuration],
    cache_path: str,
    model_path: str,
    type_map: Dict[int, str],
    device: str
) -> np.ndarray:
    """
    Загружает фингерпринты из кэша или генерирует их, если кэш отсутствует.
    """
    if os.path.exists(cache_path):
        print(f"Загрузка фингерпринтов из кэша: {cache_path}")
        return np.load(cache_path)
    
    print("Кэш фингерпринтов не найден. Начинается генерация...")
    fingerprints = _generate_mace_fingerprints(configs, model_path, type_map, device)
    
    print(f"Сохранение фингерпринтов в кэш: {cache_path}")
    np.save(cache_path, fingerprints)
    
    return fingerprints


def _generate_mace_fingerprints(
    configurations: List[Configuration],
    model_path: str,
    type_map: Dict[int, str],
    device: str
) -> np.ndarray:
    """
    Превращает список объектов Configuration в единый массив фингерпринтов.
    """
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype='float64')
    all_fingerprints = []

    for config_idx, config in enumerate(tqdm(configurations, desc="Генерация фингерпринтов")):
        if not config.atom_data:
            continue
            
        atoms = config.to_ase(type_map)
        
        descriptors = calc.get_descriptors(atoms)
        
        config_indices = np.full((descriptors.shape[0], 1), config_idx)
        combined_data = np.hstack([descriptors, config_indices])
        all_fingerprints.append(combined_data)

    return np.vstack(all_fingerprints)