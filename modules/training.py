import subprocess
import logging
import os

def train_mtp(config: dict, training_set_path: str) -> str:
    """
    Запускает обучение MTP с использованием сгенерированного датасета.
    """
    mtp_config = config['mtp_training']
    output_dir = config['general']['output_dir']

    command_parts = [
        mtp_config['mtp_executable_path'],
        mtp_config['training_command'],
        mtp_config['initial_potential'],
        training_set_path,
        f"--save-to={os.path.join(output_dir, mtp_config['output_potential_name'])}"
    ]
    
    logging.info(f"Запуск команды обучения MTP: {' '.join(command_parts)}")
    
    try:
        process = subprocess.run(
            ' '.join(command_parts),
            shell=True, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info("Вывод MTP (stdout):\n" + process.stdout)
        if process.stderr:
            logging.warning("Вывод MTP (stderr):\n" + process.stderr)
        
        trained_potential_path = os.path.join(output_dir, mtp_config['output_potential_name'])
        logging.info(f"Обучение MTP успешно завершено. Потенциал сохранен в: {trained_potential_path}")
        return trained_potential_path

    except subprocess.CalledProcessError as e:
        logging.error("Ошибка при обучении MTP!")
        logging.error(f"Код возврата: {e.returncode}")
        logging.error("Stdout:\n" + e.stdout)
        logging.error("Stderr:\n" + e.stderr)
        raise