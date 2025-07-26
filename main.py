import argparse
import json
import logging
import os
import ase.io
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from modules.configuration import Configuration
from modules.utils import setup_logging, smiles_to_ase_atoms
from modules.fingerprinting import process_dataset
from modules.selection import generate_relevant_dataset
from modules.training import train_mtp
from modules.visualization import generate_umap_plot
from modules.validation import calculate_grades

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Создаем выходную директорию
    output_dir = config['general']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Настраиваем логирование
    log_file = os.path.join(output_dir, "run.log")
    setup_logging(log_file)
    
    logging.info(f"--- НАЧАЛО РАБОТЫ: {config['general']['smiles_polymer']} ---")
    
    # --- ЭТАП 1: Итеративная обработка метаданных и фингерпринтов ---
    source_datasets_info = [] # Легкая структура: [{'path': ..., 'count': ...}, ...]
    thinned_configs_metadata = []
    all_ref_fingerprints = []
    
    type_map_str_keys = config['fingerprinting']['type_map_cfg_to_symbol']
    type_map = {int(k): v for k, v in type_map_str_keys.items()}

    for dataset_cfg in config['fingerprinting']['datasets']:
        total_count, fingerprints, metadata = process_dataset(
            dataset_config=dataset_cfg,
            model_path=config['fingerprinting']['mace_model_path'],
            type_map=type_map,
            device=config['general']['device']
        )
        
        if total_count > 0:
            source_datasets_info.append({'path': dataset_cfg['path'], 'count': total_count})
            
            # Обновляем глобальный индекс для фингерпринтов
            if fingerprints.shape[0] > 0:
                fingerprints[:, -1] += len(thinned_configs_metadata)
                all_ref_fingerprints.append(fingerprints)
            
            thinned_configs_metadata.extend(metadata)

    ref_fingerprints_combined = np.vstack(all_ref_fingerprints)
    logging.info(f"Всего проанализировано {sum(d['count'] for d in source_datasets_info)} конфигураций из {len(source_datasets_info)} файлов.")
    logging.info(f"Всего обработано {len(thinned_configs_metadata)} прореженных конфигураций.")
    
    logging.info("--- Генерация синтетических query-структур ---")
    query_atoms_list = smiles_to_ase_atoms(
        config['general']['smiles_polymer'],
        config.get('query_generation', {}) # Передаем новый блок
    )
    if not query_atoms_list:
        logging.error("Не удалось сгенерировать ни одной query-структуры. Прерывание работы.")
        return
    
    # --- ЭТАП 2: Сохранение query-структур (включая .cfg) ---
    post_cfg = config.get('postprocessing', {})
    
    # Сохраняем .xyz, если нужно
    if post_cfg.get('save_smiles_xyz', False):
        xyz_path = os.path.join(output_dir, post_cfg['smiles_xyz_filename'])
        ase.io.write(xyz_path, query_atoms_list, format='extxyz')
        logging.info(f"Сгенерированные по SMILES структуры сохранены в: {xyz_path}")
        
    # Сохраняем .cfg для последующего расчета грейдов
    query_cfg_path = None
    grade_cfg = config.get('mtp_validation', {})
    if grade_cfg.get('enabled', False):
        query_cfg_path = os.path.join(output_dir, grade_cfg['query_cfg_filename'])
        type_map_reverse = {v: int(k) for k, v in type_map_str_keys.items()}
        query_configs = [Configuration.from_ase_atoms(atoms, type_map_reverse) for atoms in query_atoms_list]
        Configuration.save_to_file(query_configs, query_cfg_path)
        logging.info(f"Синтетические конфигурации сохранены в .cfg для расчета грейдов: {query_cfg_path}")

    # --- ЭТАП 3: Генерация релевантного датасета (получение идентификаторов) ---
    logging.info("Генерация релевантного датасета...")
    # ИЗМЕНЕНО: получаем идентификаторы, а не объекты
    relevant_config_ids, query_fingerprints = generate_relevant_dataset(
        smiles=config['general']['smiles_polymer'],
        source_datasets_info=source_datasets_info,
        thinned_configs_metadata=thinned_configs_metadata,
        ref_fingerprints=ref_fingerprints_combined,
        selection_params=config['selection'],
        fp_params={
            "mace_model_path": config['fingerprinting']['mace_model_path'],
            "device": config['general']['device']
        },
        query_atoms_list=query_atoms_list
    )
    
    # --- ЭТАП 4: Ленивая загрузка и сохранение итогового датасета ---
    train_dataset_path_cfg = os.path.join(output_dir, "train.cfg")
    logging.info(f"Начинается ленивая загрузка и сохранение {len(relevant_config_ids)} конфигураций в {train_dataset_path_cfg}...")

    # Группируем идентификаторы по исходному файлу
    grouped_ids = defaultdict(list)
    for identifier in relevant_config_ids:
        grouped_ids[identifier['source_path']].append(identifier['index_in_file'])
        
    configs_to_write = None
    # Итеративно читаем из каждого файла только то, что нужно, и дописываем в выходной файл
    with open(train_dataset_path_cfg, 'w') as f_out:
        for source_path, indices in tqdm(grouped_ids.items(), desc="Сохранение .cfg"):
            # Читаем только нужные конфигурации из одного файла за раз
            configs_to_write = Configuration.from_file_by_indices(source_path, indices)
            # Записываем их в итоговый файл
            Configuration.save_to_file(configs_to_write, f_out) # Модифицируем save_to_file для приема файлового объекта
            
    logging.info("Релевантный датасет (.cfg) успешно сохранен.")
        
    
    # --- ЭТАП 5: Постобработка ---
    logging.info("Запуск постобработки...")
            
    # 1. Сохранение .xyz для релевантного датасета
    post_cfg = config.get('postprocessing', {})
    if post_cfg.get('save_relevant_xyz', False):
        xyz_path = os.path.join(output_dir, post_cfg['relevant_xyz_filename'])
        logging.info(f"Сохранение релевантного датасета в .xyz: {xyz_path}")
        # Записываем в XYZ порциями
        with open(xyz_path, 'w') as f_xyz: # Открываем в бинарном режиме для ASE
            for source_path, indices in tqdm(grouped_ids.items(), desc="Сохранение .xyz"):
                ase_configs = [cfg.to_ase(type_map) for cfg in configs_to_write]
                ase.io.write(f_xyz, ase_configs, format='extxyz', append=True)

    # 2. Генерация UMAP графика
    if post_cfg.get('generate_umap_plot', False):
        try:
            plot_path = os.path.join(output_dir, post_cfg['umap_plot_filename'])
            generate_umap_plot(
                reference_fps=ref_fingerprints_combined,
                query_fps=query_fingerprints,
                output_path=plot_path,
                umap_plot_params=post_cfg['umap_plot_params']
            )
        except Exception as e:
            logging.error(f"Не удалось сгенерировать UMAP-график: {e}")
            
    # --- ЭТАП 6: Обучение потенциала MTP ---
    trained_potential_path = None
    if config.get('mtp_training', {}).get('enabled', False):
        logging.info("Запуск обучения MTP...")
        trained_potential_path = train_mtp(config, train_dataset_path_cfg)
    else:
        logging.info("Обучение MTP пропущено (отключено в конфигурации).")
        # Если обучение пропущено, для расчета грейдов будем использовать initial_potential
        trained_potential_path = config['mtp_training']['initial_potential']
        
    # --- ЭТАП 7: АКТИВНОЕ ОБУЧЕНИЕ ---
    if config.get('active_learning', {}).get('enabled', False):
        if not query_cfg_path:
            raise ValueError("Для активного обучения необходимо сохранить query-структуры в .cfg. Установите 'postprocessing.save_smiles_cfg' в true.")
        if not trained_potential_path:
             raise ValueError("Для активного обучения необходим начальный потенциал.")
             
        run_active_learning_loop(config, train_dataset_path_cfg, trained_potential_path, query_cfg_path)

    logging.info("--- РАБОТА ЗАВЕРШЕНА ---")

    # --- ЭТАП 8: Постобработка (визуализация и РАСЧЕТ ГРЕЙДОВ) ---
    logging.info("Запуск постобработки...")
    
    # ... (сохранение .xyz для релевантного датасета) ...

    # Расчет грейдов
    if grade_cfg.get('enabled', False) and trained_potential_path:
        calculate_grades(config, trained_potential_path, train_dataset_path_cfg, query_cfg_path)
        
        # Читаем обновленный файл и выводим грейды в лог
        updated_query_configs = Configuration.from_file(query_cfg_path)
        logging.info("--- Extrapolation Grades для синтетических конфигураций ---")
        for i, cfg in enumerate(updated_query_configs):
            grade = cfg.features.get('MV_grade', 'N/A')
            name = cfg.features.get('name', f'Конфигурация {i}')
            logging.info(f"  - Структура: {os.path.basename(name)} | Grade: {grade}")
        logging.info("-----------------------------------------------------")
        
    logging.info("--- РАБОТА ЗАВЕРШЕНА ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инструмент для генерации потенциалов для полимеров.")
    parser.add_argument("config", help="Путь к JSON файлу конфигурации.")
    args = parser.parse_args()
    main(args.config)
