import numpy as np
import random
import ase
from typing import List, Dict, Optional, Any, Tuple
from ase.calculators.singlepoint import SinglePointCalculator
import re
import os

class Configuration:
    """
    Класс для представления одной конфигурации из файла .cfg,
    используемого в пакете MLIP.
    """
    def __init__(self):
        """Инициализирует пустую конфигурацию."""
        self.size: Optional[int] = None
        self.supercell: Optional[List[List[float]]] = None
        self.atom_data: List[Dict[str, Any]] = []
        self.energy: Optional[float] = None
        self.plus_stress: Optional[Dict[str, float]] = None
        # Словарь для хранения всех 'Feature' полей
        self.features: Dict[str, str] = {}

    def __repr__(self) -> str:
        """Возвращает строковое представление объекта для отладки."""
        if self.size is not None:
            return f"<Configuration with {self.size} atoms>"
        return "<Empty Configuration>"

    @staticmethod
    def from_file(filepath: str) -> List['Configuration']:
        """
        Создает список объектов Configuration из текстового файла формата .cfg.

        Args:
            filepath (str): Путь к файлу .cfg.

        Returns:
            List[Configuration]: Список объектов Configuration.
        """
        configurations = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Ошибка: Файл не найден по пути {filepath}")
            return []

        i = 0
        while i < len(lines):
            # Ищем начало нового блока
            if "BEGIN_CFG" in lines[i]:
                config = Configuration()
                config.features['name'] = filepath
                i += 1
                
                # Читаем блок до его конца
                while i < len(lines) and "END_CFG" not in lines[i]:
                    line = lines[i].strip()

                    if line.startswith("Size"):
                        i += 1
                        config.size = int(lines[i].strip())
                    elif line.startswith("Supercell"):
                        config.supercell = []
                        for _ in range(3):
                            i += 1
                            # Используем re.split для обработки любого количества пробелов
                            row = [float(x) for x in re.split(r'\s+', lines[i].strip())]
                            config.supercell.append(row)
                    elif line.startswith("AtomData:"):
                        # Заголовки данных об атомах
                        headers = line.split()[1:] # Пропускаем "AtomData:"
                        # Читаем данные для каждого атома
                        if config.size is None:
                            raise ValueError("Ошибка формата: 'Size' должен идти перед 'AtomData:'")
                        
                        for _ in range(config.size):
                            i += 1
                            values = re.split(r'\s+', lines[i].strip())
                            atom_dict = {}
                            for header, value in zip(headers, values):
                                # Преобразуем типы данных на основе заголовка
                                if header in ['id', 'type']:
                                    atom_dict[header] = int(value)
                                else:
                                    atom_dict[header] = float(value)
                            config.atom_data.append(atom_dict)
                    elif line.startswith("Energy"):
                        i += 1
                        config.energy = float(lines[i].strip())
                    elif line.startswith("PlusStress:"):
                        stress_headers = line.split()[1:] # Пропускаем "PlusStress:"
                        i += 1
                        stress_values = [float(x) for x in re.split(r'\s+', lines[i].strip())]
                        config.plus_stress = dict(zip(stress_headers, stress_values))
                    elif line.startswith("Feature"):
                        parts = line.split(maxsplit=2)
                        key = parts[1]
                        value = parts[2]
                        # Согласно документации, повторяющиеся фичи дописываются
                        if key in config.features:
                            config.features[key] += "\n" + value
                        else:
                            config.features[key] = value
                    
                    i += 1 # Переходим к следующей строке в блоке
                
                configurations.append(config)
            else:
                i += 1 # Если не начало блока, просто идем дальше
                
        return configurations
    @staticmethod
    def save_to_file(configurations: List['Configuration'], filepath: str):
        """
        Сохраняет список объектов Configuration в файл формата .cfg.

        Args:
            configurations (List[Configuration]): Список объектов для сохранения.
            filepath (str): Путь к файлу для сохранения.
        """
        with open(filepath, 'w') as f:
            for i, config in enumerate(configurations):
                f.write("BEGIN_CFG\n")
                if config.size is not None:
                    f.write("Size\n")
                    f.write(f"{config.size}\n")
                if config.supercell:
                    f.write("Supercell\n")
                    for row in config.supercell:
                        f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
                if config.atom_data:
                    headers = list(config.atom_data[0].keys())
                    f.write(f"AtomData: {' '.join(headers)}\n")
                    for atom in config.atom_data:
                        f.write(" ".join(str(atom[h]) for h in headers) + "\n")
                if config.energy is not None:
                    f.write("Energy\n")
                    f.write(f"{config.energy}\n")
                if config.plus_stress:
                    headers = list(config.plus_stress.keys())
                    f.write(f"PlusStress: {' '.join(headers)}\n")
                    f.write(" ".join(str(config.plus_stress[h]) for h in headers) + "\n")
                if config.features:
                    for key, value in config.features.items():
                        # Для многострочных фич
                        for line in value.strip().split('\n'):
                            f.write(f"Feature {key} {line}\n")
                f.write("END_CFG\n")
                if i < len(configurations) - 1:
                    f.write("\n") # Добавляем пустую строку между блоками
        print(f"Данные успешно сохранены в файл: {filepath}")
    def to_ase(self, type_map: Dict[int, str]) -> ase.Atoms:
        """
        Преобразует объект Configuration в объект ase.Atoms.

        Args:
            type_map (Dict[int, str]): Словарь для сопоставления числовых
                                       типов атомов с их химическими
                                       символами. Пример: {0: 'Si', 1: 'O'}

        Returns:
            ase.Atoms: Готовый объект ASE с позициями, ячейкой, энергией,
                       силами и напряжением (если они есть).
        """
        if not self.atom_data:
            raise ValueError("Невозможно создать объект ASE из пустой конфигурации (нет данных об атомах).")

        symbols = []
        positions = []
        forces = []
        has_forces = 'fx' in self.atom_data[0] # Проверяем наличие сил по первому атому

        for atom_dict in self.atom_data:
            atom_type = atom_dict.get('type')
            if atom_type is None:
                raise KeyError("В данных об атомах отсутствует обязательное поле 'type'.")
            
            symbol = type_map.get(atom_type)
            if symbol is None:
                raise KeyError(f"Тип атома {atom_type} не найден в предоставленном словаре type_map.")
            
            symbols.append(symbol)
            positions.append([atom_dict['cartes_x'], atom_dict['cartes_y'], atom_dict['cartes_z']])
            
            if has_forces:
                forces.append([atom_dict['fx'], atom_dict['fy'], atom_dict['fz']])

        # Создаем базовый объект Atoms
        atoms = ase.Atoms(
            symbols=symbols,
            positions=positions,
            cell=self.supercell,
            pbc=self.supercell is not None # Включаем периодичность, если есть ячейка
        )

        # Добавляем рассчитанные свойства (энергию, силы, напряжение) через SinglePointCalculator
        # Это стандартный способ хранения "одиночных" вычислений в ASE.
        
        # Словарь для хранения результатов
        results = {}
        if self.energy is not None:
            results['energy'] = self.energy
        
        if has_forces:
            results['forces'] = np.array(forces)

        if self.plus_stress is not None:
            # ASE ожидает напряжение в формате Voigt (xx, yy, zz, yz, xz, xy)
            voigt_order = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
            stress_voigt = [self.plus_stress[key] for key in voigt_order]
            results['stress'] = np.array(stress_voigt)
        
        if results:
            calculator = SinglePointCalculator(atoms, **results)
            atoms.set_calculator(calculator)
            
        return atoms
        
    @staticmethod
    def from_ase_atoms(
        atoms: ase.Atoms,
        type_map_reverse: Dict[str, int]
    ) -> 'Configuration':
        """
        Генерирует объект Configuration из объекта ase.Atoms.

        Args:
            atoms (ase.Atoms): Объект ASE для преобразования.
            type_map_reverse (Dict[str, int]): Словарь для преобразования химических
                                               символов в целочисленные типы,
                                               например {'C': 0, 'H': 1}.

        Returns:
            Configuration: Новый объект Configuration.
        """
        config = Configuration()

        # 1. Размер
        config.size = len(atoms)

        # 2. Ячейка
        if np.any(atoms.cell):
            config.supercell = atoms.cell.tolist()

        # 3. Энергия
        try:
            config.energy = atoms.get_potential_energy()
        except:
            pass # Энергия не задана

        # 4. Напряжения
        try:
            # ASE возвращает стресс в Voigt-нотации [xx, yy, zz, yz, xz, xy]
            # .cfg формат PlusStress - это часто (вириальный стресс * объем)
            stress = atoms.get_stress(voigt=True)
            volume = atoms.get_volume()
            plus_stress_values = stress * volume
            stress_keys = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
            config.plus_stress = dict(zip(stress_keys, plus_stress_values))
        except:
            pass # Напряжения не заданы

        # 5. Данные по атомам
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        has_forces = 'forces' in atoms.arrays
        forces = atoms.get_forces() if has_forces else None

        for i in range(config.size):
            atom_dict = {}
            # ID (1-индексированный)
            atom_dict['id'] = i + 1
            # Тип (преобразованный из символа)
            symbol = symbols[i]
            if symbol not in type_map_reverse:
                raise KeyError(f"Символ '{symbol}' не найден в словаре type_map_reverse.")
            atom_dict['type'] = type_map_reverse[symbol]
            # Координаты
            atom_dict['cartes_x'], atom_dict['cartes_y'], atom_dict['cartes_z'] = positions[i]
            # Силы (если есть)
            if has_forces and forces is not None:
                atom_dict['fx'], atom_dict['fy'], atom_dict['fz'] = forces[i]
            
            config.atom_data.append(atom_dict)
        
        # 6. Дополнительные поля (Features)
        # Берем данные из словаря .info объекта ASE
        for key, value in atoms.info.items():
            config.features[key] = str(value)
            
        return config