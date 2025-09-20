import argparse
import json
import logging
import os
import sys
import subprocess
import shutil
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
            logging.error(f"! ОШИБКА ПРИ РАСЧЁТЕ {log_title} !")
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
        # e.stdout может быть None, если процесс не успел ничего вывести
        stdout_str = e.stdout if e.stdout else ""
        stderr_str = e.stderr if e.stderr else ""
        logging.error("Stdout:\n" + stdout_str)
        logging.error("Stderr:\n" + stderr_str)
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
            try:
                # Проверяем, что строка содержит числа
                [float(x) for x in line.strip().split()]
                data_lines.append(line.strip().split())
            except ValueError:
                # Строка не является строкой данных (например, "WARNING: ...")
                continue


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

def run_stress_strain_workflow_consecutive(config: dict):
    """
    Выполняет последовательный цикл: растяжение -> МД LAMMPS -> МД VASP.
    """
    output_dir = config['general']['output_dir']
    lmp_cfg = config['lammps_config']
    vasp_cfg = config['ab_initio_config']

    # --- 1. Генерация одной полимерной структуры с нулевым стрейном и массива стрейнов ---
    logging.info("1. Генерация начальной структуры (нулевая деформация) и массива деформаций...")

    # Создаем массив деформаций
    q_gen_cfg = config['query_generation']
    strain_params=  q_gen_cfg["linear_strained"]["strain_range"]
    strains = np.arange(strain_params[0], strain_params[1], strain_params[2])
    logging.info(f"Массив деформаций для расчета: {strains}")

    # Создаем одну структуру с нулевой деформацией
    # Для этого временно изменяем конфиг
    gen_cfg_single = config['query_generation'].copy()
    gen_cfg_single["linear_strained"]["strain_range"] = [0.00, 0.01, 0.02]
    initial_polymer_list = smiles_to_ase_atoms(
        config['general']['smiles_polymer'],
        gen_cfg_single
    )
    if not initial_polymer_list:
        logging.error("Не удалось сгенерировать начальную структуру. Завершение работы.")
        return
    initial_polymer = initial_polymer_list[0]
    initial_length_L0 = initial_polymer.get_cell()[0, 0]
    logging.info(f"Начальная структура сгенерирована. Равновесная длина ячейки L0 = {initial_length_L0:.4f} Å")

    

    # --- ЭТАП 2: ПОСЛЕДОВАТЕЛЬНЫЙ ЗАПУСК РАСЧЕТОВ LAMMPS + VASP ---
    logging.info("\n--- ЭТАП 2: Последовательное моделирование для каждой деформации ---")

    results = []
    # Готовим первую структуру для первого шага цикла
    first_strain = strains[0]
    atoms_for_next_step = initial_polymer.copy()
    cell = atoms_for_next_step.get_cell()
    eps = 0.03
    pos1 = np.min((initial_polymer.pos2, initial_polymer.pos1), axis=0) - eps
    pos2 = np.max((initial_polymer.pos2, initial_polymer.pos1), axis=0) + eps
    cell[0, 0] = initial_length_L0 * (1 + first_strain)
    atoms_for_next_step.set_cell(cell, scale_atoms=True)

    # Путь к файлу данных, который будет передаваться между итерациями
    current_input_data_path = os.path.join(output_dir, "next_step_start.data")
    ase.io.write(current_input_data_path, atoms_for_next_step, format='lammps-data', masses=True)


    for i, strain in enumerate(tqdm(strains, desc="Обработка деформаций")):
        calc_name = f"strain_{strain:.4f}".replace('.', '_').replace('-', 'm')
        calc_dir = os.path.join(output_dir, calc_name)
        os.makedirs(calc_dir, exist_ok=True)
        logging.info(f"\n--- Обработка деформации: {strain:.4f} в директории {calc_dir} ---")

        # --- Шаг 3: Запуск LAMMPS для текущей деформации ---
        # Копируем .data файл, подготовленный на предыдущем шаге
        lammps_input_data = os.path.join(calc_dir, "start.data")
        shutil.copy(current_input_data_path, lammps_input_data)
        
        # Создаем mlip.ini
        potential_path = os.path.abspath(config['potential']['mtp_potential_path'])
        with open(os.path.join(calc_dir, "mlip.ini"), 'w') as f:
            f.write(f"mtp-filename {potential_path}\n")

        md_in_script = f"""
        processors      * 1 1
        units           metal
        atom_style      atomic
        read_data       start.data
        pair_style      mlip mlip.ini
        pair_coeff      * *

        region          toBeFixedBox block {pos1[0]*(1+strain)} {pos2[0]*(1+strain)} {pos1[1]} {pos2[1]} {pos1[2]} {pos2[2]}
        group           toBeFixed region toBeFixedBox
        group           allElse subtract all toBeFixed
        
        velocity        allElse create {lmp_cfg['temperature']} 4928459 mom yes rot yes dist gaussian
        
        fix             1 allElse nvt temp {lmp_cfg['temperature']} {lmp_cfg['temperature']} 0.1
        
        timestep        {lmp_cfg['timestep']}
        thermo          100
        thermo_style    custom step temp pe etotal press pxx pyy pzz lx
        dump            1 all custom 100 dump.xyz id type x y z

        run             1000
        
        unfix           1
        #velocity        toBeFixed create {lmp_cfg['temperature']} 4928459 mom yes rot yes dist gaussian
        fix             2 all nvt temp {lmp_cfg['temperature']} {lmp_cfg['temperature']} 0.1
        fix		mom all momentum {int(lmp_cfg['md_steps']/10)} linear 1 1 1
   
        
        run             {lmp_cfg['md_steps']}
        
        # Шаг 3: Сохранение .data файла с последнего шага
        write_data      final_step.data
        """
        md_script_path = os.path.join(calc_dir, "md.in")
        with open(md_script_path, 'w') as f:
            f.write(md_in_script)

        # Запускаем LAMMPS
        command = f"{lmp_cfg.get('run_params','')} {lmp_cfg['lammps_executable_path']} -in {os.path.basename(md_script_path)}"
        if not run_subprocess(command, calc_dir, "LAMMPS MD"):
            logging.error(f"Пропускаем шаг с деформацией {strain} из-за ошибки LAMMPS.")
            continue # Переходим к следующей деформации
        
        # Парсим результат LAMMPS
        lammps_log_path = os.path.join(calc_dir, "log.lammps")
        lammps_stress_bar = parse_lammps_log(lammps_log_path, lmp_cfg['thermo_average_last_n_steps'])
        lammps_stress_gpa = -lammps_stress_bar / 10000.0 if not np.isnan(lammps_stress_bar) else np.nan
        
        lammps_output_data_path = os.path.join(calc_dir, "final_step.data")
        if not os.path.exists(lammps_output_data_path):
            logging.error(f"Не найден выходной .data файл LAMMPS: {lammps_output_data_path}. Пропуск VASP и подготовки следующего шага.")
            continue
        logging.info(f"LAMMPS strain is {lammps_stress_gpa} GPa")

        # --- Шаг 4: Конвертация .data в POSCAR и запуск VASP ---
        vasp_stress_gpa = np.nan
        if strain >= vasp_cfg.get('enabled_from', 100000.0):
            vasp_cfg['enabled'] == True
        if (vasp_cfg.get('enabled', False)):
            logging.info(f"Запуск МД симуляции в VASP для деформации {strain:.4f}...")
            vasp_dir = os.path.join(calc_dir, "vasp_md")
            os.makedirs(vasp_dir, exist_ok=True)
            
            try:
                # Читаем результат LAMMPS
                final_atoms_from_lammps = ase.io.read(lammps_input_data, format='lammps-data')
                
                # Подготовка входных файлов VASP
                ase.io.write(os.path.join(vasp_dir, "POSCAR"), final_atoms_from_lammps, format='vasp', sort=True)
                
                # Используем симлинки
                for file_key in ['INCAR', 'KPOINTS']:
                    target_path = os.path.abspath(vasp_cfg['input_files'][file_key])

                    link_path = os.path.join(vasp_dir, file_key)
                    if not os.path.lexists(link_path):
                        os.symlink(target_path, link_path)

                # Создаем POTCAR
                potcar_path = os.path.join(vasp_dir, "POTCAR")
                unique_symbols = sorted(list(set(final_atoms_from_lammps.get_chemical_symbols())))
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

        final_cell_x = ase.io.read(lammps_output_data_path, format='lammps-data').get_cell()[0,0]
        results.append({
            "name": calc_name,
            "strain": strain,
            "final_cell_x": final_cell_x,
            "lammps_stress_gpa": lammps_stress_gpa,
            "vasp_stress_gpa": vasp_stress_gpa
        })

        # --- Шаги 5 и 6: Растяжение структуры для СЛЕДУЮЩЕЙ итерации ---
        if i + 1 < len(strains):
            logging.info(f"Подготовка структуры для следующей деформации...")
            next_strain = strains[i+1]
            
            # Считываем результат текущего шага LAMMPS
            atoms_to_stretch = ase.io.read(lammps_output_data_path, format='lammps-data')
            
            # Растягиваем до следующего значения деформации относительно L0
            new_cell = atoms_to_stretch.get_cell()
            new_Lx = initial_length_L0 * (1 + next_strain)
            logging.info(f"Растяжение ячейки по оси X до {new_Lx:.4f} Å (деформация {next_strain:.4f})")
            new_cell[0, 0] = new_Lx
            atoms_to_stretch.set_cell(new_cell, scale_atoms=True)
            
            # Сохраняем во временный файл, который будет использован на следующей итерации
            ase.io.write(current_input_data_path, atoms_to_stretch, format='lammps-data', masses=True)

    # --- 7. Построение графиков и сохранение результатов ---
    if not results:
        logging.error("Не получено ни одного результата для анализа.")
        return

    results.sort(key=lambda r: r['strain'])
    
    strains_res = np.array([r['strain'] for r in results])
    cell_lengths = np.array([r['final_cell_x'] for r in results])
    lammps_stresses = np.array([r['lammps_stress_gpa'] for r in results])
    vasp_stresses = np.array([r['vasp_stress_gpa'] for r in results])

    csv_path = os.path.join(output_dir, config['analysis']['results_csv_filename'])
    with open(csv_path, 'w') as f:
        f.write("name,strain,final_cell_x,lammps_stress_gpa,vasp_stress_gpa\n")
        for r in results:
            f.write(f"{r['name']},{r['strain']},{r['final_cell_x']},{r['lammps_stress_gpa']},{r['vasp_stress_gpa']}\n")
    logging.info(f"Таблица с результатами сохранена в: {csv_path}")

    # График stress vs strain
    plt.figure(figsize=(10, 7))
    plt.plot(strains_res, lammps_stresses, 'o-', label='LAMMPS (MTP)')
    if vasp_cfg.get('enabled', False):
        plt.plot(strains_res, vasp_stresses, 's--', label='VASP')
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

    logging.info("--- РАБОТА ЗАВЕРШЕНА ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для построения кривых напряжение-деформация для полимеров (последовательный метод).")
    parser.add_argument("config", help="Путь к JSON файлу конфигурации (stress_strain_config.json).")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    os.makedirs(config['general']['output_dir'], exist_ok=True)
    log_file = os.path.join(config['general']['output_dir'], "stress_strain_consecutive.log")
    setup_logging(log_file)

    run_stress_strain_workflow_consecutive(config)
