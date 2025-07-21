import argparse
import json
import logging
import os
import ase

from modules.configuration import Configuration
from modules.utils import setup_logging, smiles_to_ase_atoms
from modules.fingerprinting import get_or_create_fingerprints
from modules.selection import generate_relevant_dataset
from modules.training import train_mtp
from modules.visualization import generate_umap_plot # Импортируем новую функцию
# from modules.validation import run_md_validation # Раскомментировать, когда будет реализовано

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
    
    # --- ЭТАП 1: Загрузка данных и фингерпринтов ---
    logging.info("Загрузка основного датасета...")
    all_configurations = Configuration.from_file(config['fingerprinting']['large_dataset_path'])
    logging.info(f"Загружено {len(all_configurations)} конфигураций.")

    # Преобразование type_map из строк в int
    type_map_str_keys = config['fingerprinting']['type_map_cfg_to_symbol']
    type_map = {int(k): v for k, v in type_map_str_keys.items()}

    ref_fingerprints = get_or_create_fingerprints(
        configs=all_configurations,
        cache_path=config['fingerprinting']['fingerprints_cache_path'],
        model_path=config['fingerprinting']['mace_model_path'],
        type_map=type_map,
        device=config['general']['device']
    )
    
    # --- ЭТАП 2: Генерация релевантного датасета ---
    logging.info("Генерация релевантного датасета...")
    # ИЗМЕНЕНО: получаем два значения
    relevant_configs, query_fingerprints = generate_relevant_dataset(
        smiles=config['general']['smiles_polymer'],
        all_configs=all_configurations,
        ref_fingerprints=ref_fingerprints,
        selection_params=config['selection'],
        fp_params={
            "mace_model_path": config['fingerprinting']['mace_model_path'],
            "device": config['general']['device']
        }
    )
    
    train_dataset_path = os.path.join(output_dir, "train.cfg")
    Configuration.save_to_file(relevant_configs, train_dataset_path)
    logging.info(f"Релевантный датасет (.cfg) сохранен в: {train_dataset_path}")

    # --- ЭТАП 3: Обучение потенциала MTP ---
    if config.get('mtp_training', {}).get('enabled', False):
        logging.info("Запуск обучения MTP...")
        train_mtp(config, train_dataset_path)
    else:
        logging.info("Обучение MTP пропущено (отключено в конфигурации).")
        
     # --- НОВЫЙ ЭТАП 4: Постобработка и визуализация ---
    logging.info("Запуск постобработки...")
    post_cfg = config.get('postprocessing', {})

    # 1. Сохранение .xyz для query-структур
    if post_cfg.get('save_smiles_xyz', False):
        try:
            query_atoms = smiles_to_ase_atoms(config['general']['smiles_polymer'].replace('[*]', ''), num_conformers=3)
            xyz_path = os.path.join(output_dir, post_cfg['smiles_xyz_filename'])
            ase.io.write(xyz_path, query_atoms, format='extxyz', append=True)
            logging.info(f"Сгенерированные по SMILES структуры сохранены в: {xyz_path}")
        except Exception as e:
            logging.error(f"Не удалось сохранить query-структуры в .xyz: {e}")
            
    # 2. Сохранение .xyz для релевантного датасета
    if post_cfg.get('save_relevant_xyz', False):
        try:
            ase_relevant_configs = [cfg.to_ase(type_map) for cfg in relevant_configs]
            xyz_path = os.path.join(output_dir, post_cfg['relevant_xyz_filename'])
            ase.io.write(xyz_path, ase_relevant_configs, format='extxyz', append=True)
            logging.info(f"Релевантный датасет (.xyz) сохранен в: {xyz_path}")
        except Exception as e:
            logging.error(f"Не удалось сохранить релевантный датасет в .xyz: {e}")

    # 3. Генерация UMAP графика
    if post_cfg.get('generate_umap_plot', False):
        try:
            plot_path = os.path.join(output_dir, post_cfg['umap_plot_filename'])
            generate_umap_plot(
                reference_fps=ref_fingerprints,
                query_fps=query_fingerprints,
                output_path=plot_path
            )
        except Exception as e:
            logging.error(f"Не удалось сгенерировать UMAP-график: {e}")

    # --- ЭТАП 5: Валидация через МД (будущая работа) ---
    if config.get('md_validation', {}).get('enabled', False):
        logging.info("Запуск МД валидации (функционал в разработке)...")
        # run_md_validation(config, trained_potential_path)
        pass
        
    logging.info("--- РАБОТА ЗАВЕРШЕНА ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инструмент для генерации потенциалов для полимеров.")
    parser.add_argument("config", help="Путь к JSON файлу конфигурации.")
    args = parser.parse_args()
    main(args.config)