import os
import logging
import subprocess
from .configuration import Configuration

def run_lammps_md_sampling(
    config: dict,
    potential_path: str,
    state_als_path: str,
    input_data_path: str,
    output_dir: str
) -> str:
    """
    Запускает LAMMPS MD с MTP для отбора экстраполяционных конфигураций.
    """
    md_cfg = config['active_learning']['md_sampler_config']
    thresholds = config['active_learning']['thresholds']
    
    # 1. Создать mlip.ini
    mlip_ini_path = os.path.join(output_dir, "mlip.ini")
    with open(mlip_ini_path, 'w') as f:
        f.write(f"mtp-filename {os.path.abspath(potential_path)}\n")
        f.write("select TRUE\n")
        f.write(f"select:load-state {os.path.abspath(state_als_path)}\n")
        f.write(f"select:threshold {thresholds['md_start']}\n")
        f.write(f"select:threshold-break {thresholds['md_break']}\n")
        f.write(f"select:save-selected preselected.cfg\n")
    logging.info(f"Создан файл настроек mlip.ini: {mlip_ini_path}")

    # 2. Подготовить LAMMPS скрипт из шаблона
    with open(md_cfg['script_template_path'], 'r') as f:
        template = f.read()

    script_content = template.format(
        TEMPERATURE=md_cfg['temperature'],
        STEPS=md_cfg['steps'],
        INPUT_CFG=os.path.abspath(input_data_path)
    )
    script_path = "run_lammps.in"
    with open(os.path.join(output_dir, script_path), 'w') as f:
        f.write(script_content)

    # 3. Запустить LAMMPS
    command = f"{md_cfg.get('run_params','')} {md_cfg['lammps_executable_path']} -in {script_path}"
    logging.info(f"Запуск LAMMPS MD: {command}")
    
    # Используем Popen для вывода в реальном времени
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=output_dir)
    
    for line in process.stdout:
        #print(line.strip())
        logging.info(line.strip())
    
    process.wait()
    if process.returncode != 0:
        logging.error("LAMMPS завершился с ошибкой!")
        logging.error(process.stderr.read())
        # Не бросаем исключение, т.к. LAMMPS может завершиться по threshold-break, что не является ошибкой
    
    preselected_path = os.path.join(output_dir, "preselected.cfg")
    if os.path.exists(preselected_path) and os.path.getsize(preselected_path) > 0:
        logging.info(f"LAMMPS MD завершен. Найдены экстраполяционные конфигурации в {preselected_path}")
        return preselected_path
    else:
        logging.info(f"LAMMPS MD завершен. Новых экстраполяционных конфигураций в {preselected_path} не найдено.")
        return None
