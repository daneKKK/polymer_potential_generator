import os
import logging
import shutil
import subprocess
import ase.io
from typing import List
from tqdm import tqdm
from .configuration import Configuration

def run_vasp_calculations(
    configs_to_calc: List[Configuration],
    config: dict,
    output_dir: str,
) -> List[Configuration]:
    """
    Запускает VASP single-point расчеты для списка конфигураций.
    """
    vasp_cfg = config['active_learning']['ab_initio_config']
    type_map = {int(k): v for k, v in config['fingerprinting']['type_map_cfg_to_symbol'].items()}
    type_map_reverse = {v: k for k, v in type_map.items()}

    calculated_configs = []

    for i, cfg_in in enumerate(tqdm(configs_to_calc, desc="Ab initio расчеты")):
        calc_dir = os.path.join(output_dir, f"vasp_calc_{i:03d}")
        os.makedirs(calc_dir, exist_ok=True)
        
        try:
            # 1. Подготовить входные файлы
            atoms = cfg_in.to_ase(type_map)
            ase.io.write(os.path.join(calc_dir, "POSCAR"), atoms, format='vasp', sort=True)

            # Копируем INCAR, KPOINTS
            shutil.copy(vasp_cfg['input_files']['INCAR'], calc_dir)
            shutil.copy(vasp_cfg['input_files']['KPOINTS'], calc_dir)
            #shutil.copy(vasp_cfg['input_files']['POTCAR'], calc_dir) #не забываем, что в ПОТКАРе элементы в алфавитном порядке!!! н-р CHO
            
            # Создаем POTCAR
            potcar_path = os.path.join(calc_dir, "POTCAR")
            unique_symbols = sorted(list(set(atoms.get_chemical_symbols())))
            logging.info(f"Собираем один POTCAR для следующего списка элементов: {unique_symbols}")
            with open(potcar_path, 'w') as potcar_out:
                for symbol in unique_symbols:
                    try:
                        potcar_in_path = vasp_cfg['input_files']['POTCARS'][symbol]
                        with open(potcar_in_path, 'r') as potcar_in:
                            for line in potcar_in:
                                potcar_out.write(line)
                    except:
                        logging.error(f'Не найден POTCAR для элемента {symbol}')
                        raise ValueError("Не найден псевдопотенциал")
            command = f"{vasp_cfg.get('run_params','')} {vasp_cfg['executable_path']}"
            logging.info(f"Запуск VASP в {calc_dir}")
            
            
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding='utf-8',
                cwd=calc_dir
            )
            logging.info("--- Начало вывода VASP (stdout) ---")
        
            for line in process.stdout:
                clean_line = line.strip()
                logging.info(clean_line) # Дублирование в лог

            logging.info("--- Конец вывода VASP (stdout) ---")

            # Ждем завершения процесса и проверяем ошибки
            process.wait()
            stderr_output = process.stderr.read()

            if process.returncode != 0:
                logging.error("! ОШИБКА ПРИ РАСЧЁТЕ VASP !")
                logging.error(f"Код возврата: {process.returncode}")
                if stderr_output:
                    logging.error("Вывод VASP (stderr):\n" + stderr_output)
                raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr_output)
    
            if stderr_output:
                logging.warning("Вывод MTP (stderr):\n" + stderr_output)

            # 3. Прочитать результаты
            result_atoms = ase.io.read(os.path.join(calc_dir, "vasprun.xml"), format="vasp-xml")
            
            # Конвертируем обратно в Configuration, сохраняя исходные фичи
            cfg_out = Configuration.from_ase_atoms(result_atoms, type_map_reverse)
            for key, value in cfg_in.features.items():
                if key not in cfg_out.features:
                    cfg_out.features[key] = value

            calculated_configs.append(cfg_out)

        except Exception as e:
            logging.error(f"Ошибка при расчете VASP в {calc_dir}: {e}")
            continue
            
    return calculated_configs
