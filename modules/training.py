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

    # Формируем команду так же, как и раньше
    command_parts = [
        mtp_config['mtp_executable_path'],
        mtp_config['training_command'],
        mtp_config['initial_potential'],
        training_set_path,
        f"--trained-pot-name={os.path.join(output_dir, mtp_config['output_potential_name'])}"
    ]
    command = ' '.join(command_parts)
    logging.info(f"Запуск команды обучения MTP: {command}")

    try:
        # --- ИЗМЕНЕНИЕ: ИСПОЛЬЗУЕМ Popen ВМЕСТО run ---
        # Popen запускает процесс и не ждет его завершения,
        # а сразу возвращает объект для управления этим процессом.
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,       # Перенаправляем stdout в "трубу", чтобы мы могли его читать
            stderr=subprocess.PIPE,       # То же самое для stderr
            text=True,                    # Декодируем вывод в текст (UTF-8 по умолчанию)
            bufsize=1,                    # Включаем построчную буферизацию для немедленного вывода
            encoding='utf-8'              # Явно указываем кодировку
        )

        # --- НОВАЯ ЛОГИКА: ЧТЕНИЕ ВЫВОДА В РЕАЛЬНОМ ВРЕМЕНИ ---
        logging.info("--- Начало вывода MTP (stdout) ---")
        
        # Читаем stdout по одной строке, пока процесс не завершится
        # process.stdout - это файловый объект, по которому можно итерироваться
        for line in process.stdout:
            # Убираем лишние пробелы и переносы строк
            clean_line = line.strip()
            # Выводим строку в консоль для пользователя
            print(clean_line)
            # И дублируем её в лог-файл
            logging.info(clean_line)

        logging.info("--- Конец вывода MTP (stdout) ---")

        # --- ПРОВЕРКА ЗАВЕРШЕНИЯ И ОШИБОК ---
        # Ждем окончательного завершения процесса, чтобы получить код возврата
        process.wait()

        # После завершения читаем все, что могло попасть в stderr
        stderr_output = process.stderr.read()

        # Проверяем код возврата. Если он не 0, значит, произошла ошибка.
        if process.returncode != 0:
            logging.error("! ОШИБКА ПРИ ОБУЧЕНИИ MTP !")
            logging.error(f"Код возврата: {process.returncode}")
            if stderr_output:
                # Если в stderr есть информация, она, скорее всего, содержит причину ошибки
                logging.error("Вывод MTP (stderr):\n" + stderr_output)
            # Вызываем исключение, чтобы остановить выполнение всей программы
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr_output)
        
        # Если процесс завершился успешно, но в stderr все же что-то было
        # (некоторые программы пишут туда служебную информацию), выведем это как предупреждение.
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
