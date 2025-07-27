import os
import logging
import shutil
import ase
import subprocess

from typing import List, Dict
from .configuration import Configuration
from .validation import calculate_grades
from .md_sampler import run_lammps_md_sampling
from .ab_initio import run_vasp_calculations
from .training import train_mtp

def run_active_learning_loop(
    config: Dict,
    initial_train_path: str,
    initial_potential_path: str,
    query_cfg_path: str,
    type_map:Dict[int, str],
):
    """
    Запускает полный цикл активного обучения.
    """
    al_config = config['active_learning']
    output_dir = config['general']['output_dir']
    
    current_train_path = initial_train_path
    current_potential_path = initial_potential_path

    for i in range(al_config['n_iterations']):
        iter_num = i + 1
        logging.info(f"\n{'='*20} АКТИВНОЕ ОБУЧЕНИЕ: ИТЕРАЦИЯ {iter_num} {'='*20}")
        iter_dir = os.path.join(output_dir, f"iter_{iter_num:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 1. Валидация query-структур с текущим потенциалом
        state_als_path = os.path.join(iter_dir, "state.als")
        validated_queries = calculate_grades(config, current_potential_path, current_train_path, query_cfg_path, state_als_path)
        
        # 2. Определяем, нужно ли запускать AL
        configs_to_process = []
        for cfg in validated_queries:
            grade = float(cfg.features.get('MV_grade', 0.0))
            cfg.grade = grade # для удобства
            if grade >= al_config['thresholds']['md_start']:
                configs_to_process.append(cfg)
        
        
        if not configs_to_process:
            logging.info("Все query-структуры имеют грейд ниже порога. Активное обучение завершено.")
            break
            
        # Читаем обновленный файл и выводим грейды в лог
        logging.info("--- Extrapolation Grades для синтетических конфигураций в активном обучении ---")
        for i, cfg in enumerate(configs_to_process):
            grade = cfg.features.get('MV_grade', 'N/A')
            name = cfg.features.get('name', f'Конфигурация {i}')
            logging.info(f"  - Структура: {name} | Grade: {grade}")
        logging.info("-----------------------------------------------------")
            
        # 3. Сортируем кандидатов: сначала те, что > md_break, затем те, что > md_start
        configs_to_process.sort(key=lambda c: (c.grade >= al_config['thresholds']['md_break'], c.grade >= al_config['thresholds']['md_start']), reverse=True)
        
        if not al_config['calculate_all_at_once']:
            configs_to_process = configs_to_process[:1] # Берем только одного, самого "плохого"
        
        logging.info(f"На этой итерации будут обработаны {len(configs_to_process)} query-структуры.")
        
        # 4. Основной цикл MD -> select -> ab initio для каждого кандидата
        new_ab_initio_configs = []
        for cfg_to_run in configs_to_process:
            run_name = cfg_to_run.features.get('name', 'unknown').replace('/', '_')
            run_dir = os.path.join(iter_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            logging.info(f"Обработка конфигурации {run_name} по адресу {run_dir}")
            
            # Копируем .cfg файл этого кандидата для LAMMPS
            ase_data = cfg_to_run.to_ase(type_map)
            single_data_path = os.path.join(run_dir, "start.data")
            ase.io.write(single_data_path, ase_data, format='lammps-data', masses=True)
            
            # Запускаем MD-сэмплинг
            preselected_path = run_lammps_md_sampling(config, current_potential_path, state_als_path, single_data_path, run_dir)
            
            if preselected_path:
                # Запускаем select-add
                selected_path = os.path.join(run_dir, "selected.cfg")
                select_cmd = f"{config['mtp_training']['mtp_executable_path']} select-add {current_potential_path} {current_train_path} {preselected_path} {selected_path}"
                subprocess.run(select_cmd, shell=True, check=True)
                
                # Запускаем ab initio расчет
                configs_for_vasp = Configuration.from_file(selected_path)
                vasp_results = run_vasp_calculations(configs_for_vasp, config, run_dir)
                new_ab_initio_configs.extend(vasp_results)
        
        # 5. Обновляем датасет и переобучаем потенциал
        if not new_ab_initio_configs:
            logging.warning("Ни одной новой конфигурации не было посчитано на этой итерации. Завершение цикла.")
            break
            
        logging.info(f"Добавление {len(new_ab_initio_configs)} новых конфигураций в обучающий датасет.")
        
        # Создаем новый объединенный датасет
        next_train_path = os.path.join(iter_dir, "train.cfg")
        shutil.copy(current_train_path, next_train_path)
        with open(next_train_path, 'a') as f:
            Configuration.save_to_file(new_ab_initio_configs, f)
        
        # Переобучаем потенциал
        next_potential_path = os.path.join(iter_dir, "trained.mtp")
        #confif['mtp_training']["initial_potential"] = config['mtp_training']['output_potential_name']
        config['mtp_training']['output_potential_name'] = os.path.basename(next_potential_path) # обновляем имя для train_mtp
        train_mtp(config, next_train_path)

        # Обновляем пути для следующей итерации
        current_train_path = next_train_path
        current_potential_path = next_potential_path

    logging.info("Цикл активного обучения завершен.")
