import subprocess
import logging
import os

def calculate_grades(
    config: dict,
    potential_path: str,
    training_set_path: str,
    query_cfg_path: str
):
    """
    Запускает `mlp calc-grade` для оценки "новизны" query-конфигураций.
    Команда перезаписывает query_cfg_path, добавляя в него Feature grade.
    """
    mtp_exec = config['mtp_training']['mtp_executable_path']
    
    tmp_cfg_path = os.path.join(config['general']['output_dir'], 'tmp.cfg')
    command_parts = [
        mtp_exec,
        "calc-grade",
        potential_path,
        training_set_path,
        query_cfg_path, # Файл для чтения
        tmp_cfg_path  # Временный файл для записи
    ]
    command = ' '.join(command_parts)
    logging.info(f"Запуск команды расчета грейдов: {command}")
    
    try:
        process = subprocess.run(
            command,
            shell=True, check=True, text=True,
            capture_output=True, encoding='utf-8'
        )
        if process.stdout:
            logging.info("Вывод MTP calc-grade (stdout):\n" + process.stdout)
        if process.stderr:
            logging.warning("Вывод MTP calc-grade (stderr):\n" + process.stderr)
        
        process = subprocess.run(
            f'mv {tmp_cfg_path} {query_cfg_path}',
            shell=True, check=True, text=True,
            capture_output=True, encoding='utf-8'
        )
        logging.info(f"Расчет грейдов успешно завершен. Файл обновлен: {query_cfg_path}")

    except subprocess.CalledProcessError as e:
        logging.error("! ОШИБКА ПРИ РАСЧЕТЕ ГРЕЙДОВ !")
        logging.error(f"Код возврата: {e.returncode}")
        logging.error("Stdout:\n" + e.stdout)
        logging.error("Stderr:\n" + e.stderr)
        raise
