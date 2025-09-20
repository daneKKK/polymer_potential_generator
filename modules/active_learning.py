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

BASIC_TEMPLATE = """
# LAMMPS input script for standard relaxation (monomer, ring, linear)
units           metal
atom_style      atomic

read_data       {INPUT_CFG}
pair_style      mlip mlip.ini
pair_coeff      * *

velocity        all create {TEMPERATURE} {SEED} mom yes rot yes dist gaussian
fix             1 all nvt temp {TEMPERATURE} {TEMPERATURE} 0.1
timestep        0.001

thermo          100
thermo_style    custom step temp pe etotal press
dump            1 all custom 100 dump.xyz id type x y z

run             {STEPS}
"""

FIXED_ATOMS_TEMPLATE = """
# LAMMPS input script for strained chains with fixed regions
units           metal
atom_style      atomic
read_data       {INPUT_CFG}
pair_style      mlip mlip.ini
pair_coeff      * *

# Определяем регион с атомами, которые НЕ будут двигаться (концы цепи)
region          toBeFixedBox block {pos1_x} {pos2_x} {pos1_y} {pos2_y} {pos1_z} {pos2_z}
group           toBeFixed region toBeFixedBox
group           allElse subtract all toBeFixed

# Придаем начальную скорость только подвижным атомам
velocity        allElse create {TEMPERATURE} {SEED} mom yes rot yes dist gaussian

# Первый этап: релаксация подвижной части при замороженных концах
fix             1 allElse nvt temp {TEMPERATURE} {TEMPERATURE} 0.1
timestep        0.001
thermo          100
thermo_style    custom step temp pe etotal press pxx pyy pzz lx
dump            1 all custom 100 dump.xyz id type x y z

run             1000

# Второй этап: релаксация всей системы
unfix           1
fix             2 all nvt temp {TEMPERATURE} {TEMPERATURE} 0.1
fix             mom all momentum 2000 linear 1 1 1

run             {STEPS}

# Сохраняем финальную структуру
write_data      final_step.data
"""
def _format_in_script(md_sampler_config, atoms, data_path, seed):
    lmp_cfg = md_sampler_config
    if 'pos1_x' in atoms.info.keys():
        in_script = FIXED_ATOMS_TEMPLATE.format(
            TEMPERATURE=md_cfg['temperature'],
            STEPS=md_cfg['steps'],
            INPUT_CFG=os.path.abspath(data_path),
            SEED=seed,  # Новое поле для рандомизации
            pos1_x = atoms.info['pos1_x'],
            pos2_x = atoms.info['pos2_x'],
            pos1_y = atoms.info['pos1_y'],
            pos2_y = atoms.info['pos2_y'],
            pos1_z = atoms.info['pos1_z'],
            pos2_z = atoms.info['pos2_z'],
        )
    else:
        in_script = FIXED_ATOMS_TEMPLATE.format(
            TEMPERATURE=md_cfg['temperature'],
            STEPS=md_cfg['steps'],
            INPUT_CFG=os.path.abspath(data_path),
            SEED=seed  # Новое поле для рандомизации
        )

        


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
    md_sampler_config = al_config.get('md_sampler_config', {})
    max_parallel_processes = md_sampler_config.get('max_parallel_processes', 1)
    
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
        
        # Добавляем grade в объект для удобства
        for cfg in validated_queries:
            cfg.grade = float(cfg.features.get('MV_grade', 0.0))
        
        # 2. Распределение задач для MD-сэмплинга
        runs_to_start = [] # Список кортежей (конфигурация, индекс запуска)
        
        # Сортируем по грейду для стабильности
        validated_queries.sort(key=lambda c: c.grade, reverse=True)

        # Разделяем конфигурации по грейдам
        ab_initio_candidates = [c for c in validated_queries if c.grade >= al_config['thresholds']['md_break']]
        md_candidates = [c for c in validated_queries if al_config['thresholds']['md_start'] <= c.grade < al_config['thresholds']['md_break']]

        logging.info("--- Распределение MD задач ---")
        if ab_initio_candidates or md_candidates:
            # Сценарий 1 и 2
            processes_left = max_parallel_processes
            # Вариант 1: кандидаты > md_break. Всегда запускаются по одному разу.
            for cfg in ab_initio_candidates:
                if processes_left > 0:
                    runs_to_start.append((cfg, 0)) # 0 - индекс запуска
                    processes_left -= 1
            logging.info(f"Запланировано {len(runs_to_start)} запусков для конфигураций с грейдом > {al_config['thresholds']['md_break']}.")

            # Вариант 2: кандидаты > md_start. Делим между ними оставшиеся процессы.
            if md_candidates and processes_left > 0:
                n_md_cand = len(md_candidates)
                base_runs_per_cand = processes_left // n_md_cand
                extra_runs = processes_left % n_md_cand
                
                logging.info(f"Распределяем {processes_left} процессов между {n_md_cand} MD-кандидатами.")

                for idx, cfg in enumerate(md_candidates):
                    num_runs = base_runs_per_cand + (1 if idx < extra_runs else 0)
                    for run_idx in range(num_runs):
                        runs_to_start.append((cfg, run_idx))
        else:
            # Сценарий 3: ни у кого нет грейда > md_start. Делим процессы между всеми.
            if validated_queries and max_parallel_processes > 0:
                n_queries = len(validated_queries)
                base_runs_per_query = max_parallel_processes // n_queries
                extra_runs = max_parallel_processes % n_queries
                logging.info(f"Ни одна конфигурация не превысила порог {al_config['thresholds']['md_start']}. Распределяем {max_parallel_processes} процессов между всеми {n_queries} конфигурациями.")

                for idx, cfg in enumerate(validated_queries):
                    num_runs = base_runs_per_query + (1 if idx < extra_runs else 0)
                    for run_idx in range(num_runs):
                        runs_to_start.append((cfg, run_idx))

        if not runs_to_start:
            logging.info("Нет конфигураций для запуска MD-сэмплинга на этой итерации. Цикл завершен.")
            break
        
        logging.info(f"Всего будет запущено {len(runs_to_start)} MD-симуляций.")
        logging.info("---------------------------------")
        
        # 3. Подготовка и запуск MD для всех запланированных задач
        input_data_paths = []
        output_dirs = []
        in_scripts = [] # <--- Создаем список для хранения сгенерированных скриптов
        
        for cfg_to_run, run_idx in runs_to_start:
            run_name = cfg_to_run.features.get('name', 'unknown').replace('/', '_')
            run_dir = os.path.join(iter_dir, f"{run_name}_run{run_idx}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Конвертируем в ASE. Теперь ase_data.info будет содержать все нужные нам данные!
            ase_data = cfg_to_run.to_ase(type_map)
            single_data_path = os.path.join(run_dir, "start.data")
            ase.io.write(single_data_path, ase_data, format='lammps-data', masses=True)
            
            input_data_paths.append(single_data_path)
            output_dirs.append(run_dir)
            
            # === НОВАЯ ЛОГИКА ВЫБОРА И ФОРМАТИРОВАНИЯ ШАБЛОНА ===
            script_content = ""
            generation_type = ase_data.info.get('generation_type', 'unknown')
            seed = 4928459 + run_idx # Уникальный seed для каждого запуска

            if generation_type == 'linear_strained':
                logging.info(f"  -> Генерация LAMMPS скрипта для растянутой цепи: {run_name}")
                # Проверяем наличие всех ключей для безопасности
                required_keys = ['pos1_x', 'pos2_x', 'pos1_y', 'pos2_y', 'pos1_z', 'pos2_z']
                if all(key in ase_data.info for key in required_keys):
                    script_content = FIXED_ATOMS_TEMPLATE.format(
                        TEMPERATURE=md_sampler_config['temperature'],
                        STEPS=md_sampler_config['steps'],
                        INPUT_CFG=os.path.abspath(single_data_path),
                        SEED=seed,
                        pos1_x=ase_data.info['pos1_x'], pos2_x=ase_data.info['pos2_x'],
                        pos1_y=ase_data.info['pos1_y'], pos2_y=ase_data.info['pos2_y'],
                        pos1_z=ase_data.info['pos1_z'], pos2_z=ase_data.info['pos2_z'],
                    )
                else:
                    logging.error(f"Для структуры {run_name} типа 'linear_strained' отсутствуют координаты для фиксации! Используется базовый шаблон.")
                    # Откатываемся к базовому шаблону в случае ошибки
                    generation_type = 'fallback'

            if generation_type != 'linear_strained': # Включая мономер, кольца, линейные и fallback
                 logging.info(f"  -> Генерация стандартного LAMMPS скрипта для: {run_name}")
                 script_content = BASIC_TEMPLATE.format(
                    TEMPERATURE=md_sampler_config['temperature'],
                    STEPS=md_sampler_config['steps'],
                    INPUT_CFG=os.path.abspath(single_data_path),
                    SEED=seed
                 )
            
            in_scripts.append(script_content)
            # =======================================================

        # Передаем сгенерированные скрипты в md_sampler
        preselected_paths = run_lammps_md_sampling(
            config, current_potential_path, state_als_path, input_data_paths, output_dirs,
            in_scripts=in_scripts  # <--- Передаем новый аргумент
        )
        
        new_ab_initio_configs = []
        if preselected_paths:
            # Объединяем все preselected.cfg в один файл для select-add
            all_preselected_path = os.path.join(iter_dir, "all_preselected.cfg")
            with open(all_preselected_path, 'wb') as outfile:
                for fname in preselected_paths:
                    with open(fname, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)

            # Запускаем select-add один раз на объединенном файле
            selected_path = os.path.join(iter_dir, "selected.cfg")
            select_cmd = f"{config['mtp_training']['mtp_executable_path']} select-add {current_potential_path} {current_train_path} {all_preselected_path} {selected_path}"
            logging.info(f"Запуск select-add: {select_cmd}")
            subprocess.run(select_cmd, shell=True, check=True)
            
            # Запускаем ab initio расчет для новых выделенных конфигураций
            if os.path.exists(selected_path) and os.path.getsize(selected_path) > 0:
                configs_for_vasp = Configuration.from_file(selected_path)
                # Директория для VASP расчетов
                vasp_dir = os.path.join(iter_dir, "vasp_calculations")
                vasp_results = run_vasp_calculations(configs_for_vasp, config, vasp_dir)
                new_ab_initio_configs.extend(vasp_results)
        
        # 4. Обновляем датасет и переобучаем потенциал
        if not new_ab_initio_configs:
            logging.info("Ни одной новой конфигурации не было посчитано на этой итерации. Переход к следующей итерации.")
            continue
            
        logging.info(f"Добавление {len(new_ab_initio_configs)} новых конфигураций в обучающий датасет.")
        
        # Создаем новый объединенный датасет
        next_train_path = os.path.join(iter_dir, "train.cfg")
        appended_path = os.path.join(iter_dir, "appended.cfg")
        shutil.copy(current_train_path, next_train_path)
        with open(next_train_path, 'a') as f:
            Configuration.save_to_file(new_ab_initio_configs, f)
        with open(appended_path, 'w') as f:
            Configuration.save_to_file(new_ab_initio_configs, f)
        
        # Переобучаем потенциал
        next_potential_path = os.path.join(os.path.basename(iter_dir), "trained.mtp")
        config['mtp_training']["initial_potential"] = os.path.join(output_dir, config['mtp_training']['output_potential_name'])
        config['mtp_training']['output_potential_name'] = next_potential_path 
        train_mtp(config, next_train_path)

        # Обновляем пути для следующей итерации
        current_train_path = next_train_path
        current_potential_path = os.path.join(output_dir, next_potential_path)
        config['mtp_training']["initial_potential"] = current_potential_path

    logging.info("Цикл активного обучения завершен.") 
