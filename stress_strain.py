import argparse
import json
import logging
import os
import sys
import subprocess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ase.io
from ase import Atoms

from modules.utils import setup_logging, smiles_to_ase_atoms
from modules.configuration import Configuration

# === Утилиты для запуска симуляций ===

def run_subprocess(command, cwd, log_title):
    """Универсальная функция для запуска внешних процессов."""
    logging.info(f"--- Запуск {log_title} в {cwd} ---")
    logging.info(f"Команда: {command}")
    try:
        process = subprocess.Popen(
            command, shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8',
            cwd=cwd
        )
        logging.info("--- Начало вывода (stdout) ---")
        
        for line in process.stdout:
            clean_line = line.strip()
            logging.info(clean_line) # Дублирование в лог

        logging.info("--- Конец вывода (stdout) ---")
        
        process.wait()
        stderr_output = process.stderr.read()

        if process.returncode != 0:
            logging.error("! ОШИБКА ПРИ РАСЧЁТЕ VASP !")
            logging.error(f"Код возврата: {process.returncode}")
            if stderr_output:
                logging.error("Вывод (stderr):\n" + stderr_output)
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr_output)
    
        if stderr_output:
            logging.warning("Вывод (stderr):\n" + stderr_output)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"! ОШИБКА при запуске {log_title}!")
        logging.error(f"Код возврата: {e.returncode}")
        logging.error("Stdout:\n" + e.stdout)
        logging.error("Stderr:\n" + e.stderr)
        return False

def parse_lammps_log(log_path: str, n_average: int) -> float:
    """Извлекает среднее давление Pxx из лога LAMMPS."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Лог-файл LAMMPS не найден: {log_path}")
        return np.nan

    data_lines = []
    in_data_section = False
    for line in lines:
        if line.strip().startswith('Step'):
            in_data_section = True
            continue
        if line.strip().startswith('Loop time'):
            in_data_section = False
            continue
        if in_data_section and line.strip():
            data_lines.append(line.strip().split())

    if not data_lines:
        logging.error(f"Не найдены данные термодинамики в {log_path}")
        return np.nan

    try:
        # Pxx - 5-й столбец (индекс 4)
        pxx_values = [float(row[4]) for row in data_lines[-n_average:]]
        return np.mean(pxx_values)
    except (IndexError, ValueError) as e:
        logging.error(f"Ошибка парсинга данных в {log_path}: {e}")
        return np.nan


# === Основной воркфлоу ===

def run_stress_strain_workflow(config: dict):
    """
    Выполняет полный цикл: генерация, минимизация, МД и анализ.
    """
    output_dir = config['general']['output_dir']
    lmp_cfg = config['lammps_config']
    minimizer = lmp_cfg.get('mtp_minimization', False)
    
    # --- 1. Генерация query-структур ---
    logging.info("1. Генерация растянутых/сжатых структур по SMILES...")
    strained_configs_ase = smiles_to_ase_atoms(
        config['general']['smiles_polymer'],
        config['query_generation']
    )
    if not strained_configs_ase:
        logging.error("Не удалось сгенерировать ни одной структуры. Завершение работы.")
        return
    
    logging.info(f"Сгенерировано {len(strained_configs_ase)} структур.")

    # --- ЭТАП 1: ПОДГОТОВКА И МИНИМИЗАЦИЯ (ПОСЛЕДОВАТЕЛЬНО) ---
    md_tasks = []
    logging.info("\n--- ЭТАП 1: Подготовка задач и минимизация структур (последовательно) ---")
    for atoms in tqdm(strained_configs_ase, desc="Подготовка и минимизация"):
        calc_name = atoms.info.get('name', f'struct_{len(md_tasks)}')
        calc_dir = os.path.join(output_dir, calc_name)
        os.makedirs(calc_dir, exist_ok=True)
        
        # Создаем mlip.ini
        potential_path = os.path.abspath(config['potential']['mtp_potential_path'])
        with open(os.path.join(calc_dir, "mlip.ini"), 'w') as f:
            f.write(f"mtp-filename {potential_path}\n")

        # Создаем lammps.data
        start_data_path = os.path.join(calc_dir, "start.data")
        ase.io.write(start_data_path, atoms, format='lammps-data', masses=True)
        
        # Создаем скрипт для минимизации
        min_in_script = f"""
        units           metal
        atom_style      atomic
        read_data       {os.path.basename(start_data_path)}
        pair_style      mlip mlip.ini
        pair_coeff      * *
        thermo          10
        thermo_style    custom step temp pe etotal press pxx pyy pzz lx
        minimize        1.0e-8 1.0e-8 1000 10000
        dump            1 all custom 10 dump_min.xyz id type x y z
        write_data      minimized.data
        """
        if minimizer:
            min_script_path = os.path.join(calc_dir, "minimize.in")
            with open(min_script_path, 'w') as f:
                f.write(min_in_script)
            
            # Запускаем LAMMPS минимизацию
            command = f"{lmp_cfg.get('run_params','')} {lmp_cfg['lammps_executable_path']} -in {os.path.basename(min_script_path)}"
            if not run_subprocess(command, calc_dir, "LAMMPS Minimization"):
                continue

            minimized_data_path = os.path.join(calc_dir, "minimized.data")
            if not os.path.exists(minimized_data_path):
                logging.error(f"Файл с минимизированной структурой не найден в {calc_dir}. Пропуск.")
                continue
        else:
            minimized_data_path = start_data_path
        # Сохраняем информацию для следующего этапа
        md_tasks.append({
            "calc_dir": calc_dir,
            "initial_atoms": atoms,
            "minimized_atoms_path": minimized_data_path,
            "atom_pos1": np.min((atoms.pos2, atoms.pos1), axis=0),
            "atom_pos2": np.max((atoms.pos1, atoms.pos2), axis=0)
        })

    # --- ЭТАП 2: ЗАПУСК LAMMPS MD (ПАРАЛЛЕЛЬНО) ---
    logging.info(f"\n--- ЭТАП 2: Запуск {len(md_tasks)} LAMMPS MD симуляций (параллельно, до {lmp_cfg['max_parallel_processes']} процессов) ---")
    running_processes = []
    for task in tqdm(md_tasks, desc="Запуск LAMMPS MD"):
        calc_dir = task['calc_dir']
        eps = 0.1
        pos1 = task['atom_pos1'] - eps
        pos2 = task['atom_pos2'] + eps
        
        md_in_script = f"""
        units           metal
        atom_style      atomic
        read_data       start.data
        pair_style      mlip mlip.ini
        pair_coeff      * *
        group           Hs type 2
        region          toBeFixedBox block {pos1[0]} {pos2[0]} {pos1[1]} {pos2[1]} {pos1[2]} {pos2[2]}
        group           toBeFixed region toBeFixedBox
        group           allElse subtract all Hs toBeFixed
        velocity        Hs create {lmp_cfg['temperature']} 4928459 mom yes rot yes dist gaussian
        fix		Hsrelax Hs nvt temp {lmp_cfg['temperature']} {lmp_cfg['temperature']} 0.1
        timestep        {lmp_cfg['timestep']}
        thermo          100
        thermo_style    custom step temp pe etotal press pxx pyy pzz lx
        dump            1 all custom 10 dump.xyz id type x y z
        #run             2000
        velocity        allElse create {lmp_cfg['temperature']} 4928459 mom yes rot yes dist gaussian

        fix             1 allElse nvt temp {lmp_cfg['temperature']} {lmp_cfg['temperature']} 0.1
        run             2000
        fix             2 toBeFixed nvt temp {lmp_cfg['temperature']} {lmp_cfg['temperature']} 0.1
        run             {lmp_cfg['md_steps']}
        """
        md_script_path = os.path.join(calc_dir, "md.in")
        with open(md_script_path, 'w') as f:
            f.write(md_in_script)
        
        command = f"{lmp_cfg.get('run_params','')} {lmp_cfg['lammps_executable_path']} -in {os.path.basename(md_script_path)}"
        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=calc_dir)
        running_processes.append({'process': process, 'task': task})

    # Ожидаем завершения всех процессов
    for p_info in tqdm(running_processes, desc="Ожидание завершения LAMMPS MD"):
        p_info['process'].wait()

    # --- ЭТАП 3: СБОР РЕЗУЛЬТАТОВ И ЗАПУСК VASP MD (ПОСЛЕДОВАТЕЛЬНО) ---
    logging.info("\n--- ЭТАП 3: Сбор результатов LAMMPS и запуск VASP MD (последовательно) ---")
    results = []
    for p_info in tqdm(running_processes, desc="Обработка результатов"):
        process = p_info['process']
        task = p_info['task']
        calc_dir = task['calc_dir']
        
        # Проверяем код возврата
        if process.returncode != 0:
            logging.warning(f"LAMMPS MD в {calc_dir} завершился с ненулевым кодом {process.returncode}!")
            stdout, stderr = process.communicate()
            logging.warning(f"STDERR from {calc_dir}:\n{stderr}")

        # Парсим результат LAMMPS
        lammps_stress_bar = parse_lammps_log(os.path.join(calc_dir, "log.lammps"), lmp_cfg['thermo_average_last_n_steps'])
        lammps_stress_gpa = -lammps_stress_bar / 10000.0 if not np.isnan(lammps_stress_bar) else np.nan
        
        # Загружаем минимизированную структуру для VASP
        minimized_atoms = ase.io.read(task['minimized_atoms_path'], format='lammps-data')

        # Запускаем VASP MD
        vasp_stress_gpa = np.nan
        vasp_cfg = config.get('ab_initio_config', {})
        if vasp_cfg.get('enabled', False):
            logging.info(f"Запуск МД симуляции в VASP для {os.path.basename(calc_dir)}...")
            vasp_dir = os.path.join(calc_dir, "vasp_md")
            os.makedirs(vasp_dir, exist_ok=True)
            
            try:
                # Подготовка входных файлов VASP
                ase.io.write(os.path.join(vasp_dir, "POSCAR"), minimized_atoms, format='vasp', sort=True)
                
                # Используем симлинки, чтобы не копировать большие файлы
                incar_path = os.path.join(vasp_dir, 'INCAR')
                if not os.path.lexists(incar_path):
                    os.symlink(os.path.abspath(vasp_cfg['input_files']['INCAR_MD']), incar_path)
                
                kpoints_path = os.path.join(vasp_dir, 'KPOINTS')
                if not os.path.lexists(kpoints_path):
                    os.symlink(os.path.abspath(vasp_cfg['input_files']['KPOINTS']), kpoints_path)

                # Создаем POTCAR
                potcar_path = os.path.join(vasp_dir, "POTCAR")
                unique_symbols = sorted(list(set(minimized_atoms.get_chemical_symbols())))
                with open(potcar_path, 'w') as potcar_out:
                    for symbol in unique_symbols:
                        potcar_in_path = os.path.abspath(vasp_cfg['input_files']['POTCARS'][symbol])
                        with open(potcar_in_path, 'r') as potcar_in:
                            potcar_out.write(potcar_in.read())
                
                command = f"{vasp_cfg.get('run_params','')} {vasp_cfg['executable_path']}"
                if not run_subprocess(command, vasp_dir, "VASP MD"):
                    raise RuntimeError("VASP calculation failed.")

                vasprun_path = os.path.join(vasp_dir, "vasprun.xml")
                traj = ase.io.read(vasprun_path, format="vasp-xml", index=":")
                stresses = [step.get_stress(voigt=False)[0,0] for step in traj[len(traj)//2:]]
                avg_stress_kBar = np.mean(stresses)
                vasp_stress_gpa = -avg_stress_kBar / 10.0

            except Exception as e:
                logging.error(f"Ошибка при расчете VASP в {vasp_dir}: {e}")

        results.append({
            "name": os.path.basename(calc_dir),
            "initial_cell_x": task['initial_atoms'].get_cell()[0, 0],
            "minimized_cell_x": minimized_atoms.get_cell()[0,0],
            "lammps_stress_gpa": lammps_stress_gpa,
            "vasp_stress_gpa": vasp_stress_gpa
        })

    # --- 5. Построение графиков и сохранение результатов ---
    if not results:
        logging.error("Не получено ни одного результата для анализа.")
        return

    results.sort(key=lambda r: r['minimized_cell_x'])
    
    cell_lengths = np.array([r['minimized_cell_x'] for r in results])
    lammps_stresses = np.array([r['lammps_stress_gpa'] for r in results])
    vasp_stresses = np.array([r['vasp_stress_gpa'] for r in results])

    csv_path = os.path.join(output_dir, config['analysis']['results_csv_filename'])
    with open(csv_path, 'w') as f:
        f.write("name,initial_cell_x,minimized_cell_x,lammps_stress_gpa,vasp_stress_gpa\n")
        for r in results:
            f.write(f"{r['name']},{r['initial_cell_x']},{r['minimized_cell_x']},{r['lammps_stress_gpa']},{r['vasp_stress_gpa']}\n")
    logging.info(f"Таблица с результатами сохранена в: {csv_path}")


    # 1) График stress vs length
    plt.figure(figsize=(10, 7))
    plt.plot(cell_lengths, lammps_stresses, 'o-', label='LAMMPS (MTP)')
    if vasp_cfg.get('enabled', False):
        plt.plot(cell_lengths, vasp_stresses, 's--', label='VASP')
    plt.axhline(0, color='grey', linestyle='--')
    plt.xlabel('Длина ячейки по оси X, Å')
    plt.ylabel('Напряжение (xx), ГПа')
    plt.title('Зависимость напряжения от длины ячейки')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, config['analysis']['plot_filename_stress_length'])
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"График 'Stress vs Length' сохранен в: {plot_path}")

    # 2) График stress vs strain
    # Находим L0 (длину при нулевом напряжении) интерполяцией
    valid_mask = ~np.isnan(lammps_stresses)
    if np.any(valid_mask):
        try:
            L0 = np.interp(0, lammps_stresses[valid_mask], cell_lengths[valid_mask])
            logging.info(f"Равновесная длина ячейки (L0), определенная по LAMMPS: {L0:.4f} Å")
            
            strains = (cell_lengths - L0) / L0
            
            plt.figure(figsize=(10, 7))
            plt.plot(strains, lammps_stresses, 'o-', label='LAMMPS (MTP)')
            if vasp_cfg.get('enabled', False):
                plt.plot(strains, vasp_stresses, 's--', label='VASP')
                
            plt.axhline(0, color='grey', linestyle='--')
            plt.axvline(0, color='grey', linestyle='--')
            plt.xlabel('Инженерная деформация (strain)')
            plt.ylabel('Напряжение (xx), ГПа')
            plt.title('Диаграмма "Напряжение-деформация"')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, config['analysis']['plot_filename_stress_strain'])
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logging.info(f"График 'Stress vs Strain' сохранен в: {plot_path}")
        except Exception as e:
            logging.error(f"Не удалось построить график Stress-Strain: {e}")

    logging.info("--- РАБОТА ЗАВЕРШЕНА ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для построения кривых напряжение-деформация для полимеров.")
    parser.add_argument("config", help="Путь к JSON файлу конфигурации (stress_strain_config.json).")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    os.makedirs(config['general']['output_dir'], exist_ok=True)
    log_file = os.path.join(config['general']['output_dir'], "stress_strain.log")
    setup_logging(log_file)

    run_stress_strain_workflow(config)
