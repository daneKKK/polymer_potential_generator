import subprocess
import logging
import os

def train_mtp(config: dict, training_set_path: str) -> str:
    """
    Запускает обучение MTP с использованием сгенерированного датасета,
    выводя лог обучения в реальном времени.
    """
    mtp_config = config['mtp_training']
    output_dir = config['general']['output_dir']

    # Формируем команду динамически
    command_parts = [
    	mtp_config['run_params'],
        mtp_config['mtp_executable_path'],
        mtp_config['training_command'],
        mtp_config['initial_potential'],
        training_set_path,
        f"--trained-pot-name={os.path.join(output_dir, mtp_config['output_potential_name'])}"
    ]

    # Добавляем кастомные параметры из конфига
    params = mtp_config.get('training_params', {})
    for key, value in params.items():
        if isinstance(value, bool) and value:
            command_parts.append(f"--{key}")
        elif not isinstance(value, bool):
            command_parts.append(f"--{key}={value}")
            
    command = ' '.join(command_parts)
    logging.info(f"Запуск команды обучения MTP: {command}")

    try:
        # Используем Popen для захвата вывода в реальном времени
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )

        logging.info("--- Начало вывода MTP (stdout) ---")
        
        # Читаем stdout в реальном времени
        for line in process.stdout:
            clean_line = line.strip()
            logging.info(clean_line) # Дублирование в лог

        logging.info("--- Конец вывода MTP (stdout) ---")

        # Ждем завершения процесса и проверяем ошибки
        process.wait()
        stderr_output = process.stderr.read()

        if process.returncode != 0:
            logging.error("! ОШИБКА ПРИ ОБУЧЕНИИ MTP !")
            logging.error(f"Код возврата: {process.returncode}")
            if stderr_output:
                logging.error("Вывод MTP (stderr):\n" + stderr_output)
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr_output)
        
        if stderr_output:
            logging.warning("Вывод MTP (stderr):\n" + stderr_output)

        trained_potential_path = os.path.join(output_dir, mtp_config['output_potential_name'])
        logging.info(f"Обучение MTP успешно завершено. Потенциал сохранен в: {trained_potential_path}")
        return trained_potential_path

    except FileNotFoundError:
        logging.error(f"Ошибка: исполняемый файл MTP не найден по пути '{mtp_config['mtp_executable_path']}'")
        raise
    except Exception as e:
        logging.error(f"Не удалось запустить или завершить процесс MTP: {e}")
        raise
