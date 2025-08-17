import os
import logging
import subprocess
import time
from typing import List

# Обратите внимание на импорт List из typing

def run_lammps_md_sampling(
    config: dict,
    potential_path: str,
    state_als_path: str,
    input_data_paths: List[str],
    output_dirs: List[str],
) -> List[str]:
    """
    Запускает LAMMPS MD с MTP параллельно для набора конфигураций.
    
    Args:
        config: Словарь с конфигурацией.
        potential_path: Путь к файлу потенциала .mtp.
        state_als_path: Путь к файлу состояния .als.
        input_data_paths: Список путей к входным данным LAMMPS (.data).
        output_dirs: Список путей к директориям вывода для каждого запуска.

    Returns:
        Список путей к созданным 'preselected.cfg', где найдены экстраполяционные конфигурации.
    """
    md_cfg = config['active_learning']['md_sampler_config']
    thresholds = config['active_learning']['thresholds']
    
    max_processes = md_cfg.get('max_parallel_processes', 1)
    initial_seed = md_cfg.get('initial_seed', 4928459)
    
    processes = []
    successful_preselected_paths = []
    
    # Загружаем шаблон LAMMPS скрипта один раз
    with open(md_cfg['script_template_path'], 'r') as f:
        template = f.read()

    for i, (input_path, output_dir) in enumerate(zip(input_data_paths, output_dirs)):
        
        # 1. Создать mlip.ini в соответствующей директории
        mlip_ini_path = os.path.join(output_dir, "mlip.ini")
        with open(mlip_ini_path, 'w') as f:
            f.write(f"mtp-filename {os.path.abspath(potential_path)}\n")
            f.write("select TRUE\n")
            f.write(f"select:load-state {os.path.abspath(state_als_path)}\n")
            f.write(f"select:threshold {thresholds['md_start']}\n")
            f.write(f"write-cfgs:skip 9\n")
            f.write(f"select:threshold-break {thresholds['md_break']}\n")
            f.write(f"select:save-selected preselected.cfg\n")
        logging.info(f"Создан файл настроек mlip.ini: {mlip_ini_path}")

        # 2. Подготовить LAMMPS скрипт из шаблона с уникальным SEED
        seed = initial_seed + i
        script_content = template.format(
            TEMPERATURE=md_cfg['temperature'],
            STEPS=md_cfg['steps'],
            INPUT_CFG=os.path.abspath(input_path),
            SEED=seed  # Новое поле для рандомизации
        )
        script_path = "run_lammps.in"
        with open(os.path.join(output_dir, script_path), 'w') as f:
            f.write(script_content)

        # 3. Подготовить и запустить команду LAMMPS
        command = f"{md_cfg.get('run_params','')} {md_cfg['lammps_executable_path']} -in {script_path}"
        logging.info(f"Подготовка к запуску LAMMPS MD в {output_dir}: {command}")
        
        # Используем Popen для неблокирующего запуска
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=output_dir)
        processes.append({'process': process, 'output_dir': output_dir})

    # Ждем завершения оставшихся процессов
    for p_info in processes:
        p_info['process'].wait()
        logging.info(f"Процесс LAMMPS в {p_info['output_dir']} завершился.")
        stdout, stderr = p_info['process'].communicate()
        #logging.info(f"STDOUT from {p_info['output_dir']}:\n{stdout}")
        if p_info['process'].returncode != 0:
            logging.warning(f"LAMMPS в {p_info['output_dir']} завершился с ненулевым кодом {p_info['process'].returncode}!")
            #logging.warning(f"STDERR from {p_info['output_dir']}:\n{stderr}")
        
        preselected_path = os.path.join(p_info['output_dir'], "preselected.cfg")
        if os.path.exists(preselected_path) and os.path.getsize(preselected_path) > 0:
            logging.info(f"Найдены экстраполяционные конфигурации в {preselected_path}")
            successful_preselected_paths.append(preselected_path)
            
    if not successful_preselected_paths:
        logging.info("LAMMPS MD завершен. Новых экстраполяционных конфигураций не найдено.")
        return []
    else:
        return successful_preselected_paths
