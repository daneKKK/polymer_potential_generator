import logging
import re
import numpy as np
from typing import List, Dict, Optional
from ase import Atoms
from ase.build.rotate import rotation_matrix_from_points
#from ase.geometry import get_rotation_matrix
from rdkit import Chem
from rdkit.Chem import AllChem


def setup_logging(log_file: str):
    """Настраивает логирование в файл и консоль."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def _generate_monomer(smiles: str, r_max=8) -> Optional[Atoms]:
    """Генерирует 3D структуру для одного мономера."""
    monomer_smiles_str = smiles.replace('[*]', '')
    logging.info(f"  -> Генерация мономера: {monomer_smiles_str}")
    try:
        mol = Chem.MolFromSmiles(monomer_smiles_str)
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            logging.warning("Не удалось встроить 3D координаты для мономера.")
            return None

        AllChem.MMFFOptimizeMolecule(mol)
        
        positions = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions, cell=[100, 100, 100], pbc=True)
        borders = np.max(atoms.get_positions(), axis=0) - np.min(atoms.get_positions(), axis=0) + 2 * (r_max + 1) #по-хорошему надо работать сразу с positions, но мне впадлу тестить работоспособность - я хз какая форма у positions и где какая ось
        atoms.set_cell(borders)
        atoms.translate(borders * 0.5)
        
        # Добавляем метаданные
        atoms.info['name'] = 'monomer'
        atoms.info['generation_type'] = 'monomer'
        
        return atoms
    except Exception as e:
        logging.error(f"Ошибка при генерации мономера: {e}", exc_info=True)
        return None

def _generate_linear_oligomer(polymer_smiles: str, n: int, r_max=5) -> Optional[Atoms]:
    """Генерирует 3D структуру для линейного олигомера из n звеньев."""
    logging.info(f"  -> Генерация линейного олигомера (n={n})")
    
    # Собираем SMILES для линейной цепочки
    monomer_body = polymer_smiles.replace('[*]', '')
    chain_smiles = monomer_body*n
    
    try:
        mol_chain = Chem.MolFromSmiles(chain_smiles)
        mol_chain = Chem.AddHs(mol_chain)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_chain, params) == -1:
            logging.warning(f"Не удалось встроить 3D координаты для линейного олигомера n={n}")
            return None
        
        AllChem.MMFFOptimizeMolecule(mol_chain)

        positions = mol_chain.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol_chain.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0], pbc=True)
        borders = np.max(atoms.get_positions(), axis=0) - np.min(atoms.get_positions(), axis=0) + 2 * (r_max + 1)
        atoms.set_cell(borders)
        atoms.translate(borders * 0.5)

        # Добавляем метаданные
        atoms.info['name'] = f'linear_n{n}'
        atoms.info['generation_type'] = 'linear'
        atoms.info['length'] = n

        return atoms
    except Exception as e:
        logging.error(f"Ошибка при генерации линейного олигомера n={n}: {e}", exc_info=True)
        return None

def _generate_ring(polymer_smiles: str, n: int, r_max=5.0) -> Optional[Atoms]:
    """Генерирует 3D структуру для циклического олигомера из n звеньев."""
    logging.info(f"  -> Генерация кольца (n={n})")

    # Сначала создаем линейный олигомер
    monomer_head = polymer_smiles[::-1].replace(']*[','',1)[::-1]
    monomer_tail = polymer_smiles.replace('[*]', '', 1)
    monomer_body = polymer_smiles.replace('[*]', '')
    chain_smiles = monomer_head + monomer_body*(n-2) + monomer_tail

    try:
        mol_chain = Chem.MolFromSmiles(chain_smiles)
        wildcard_atoms = [atom for atom in mol_chain.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(wildcard_atoms) != 2: return None

        rw_mol = Chem.RWMol(mol_chain)
        idx1, idx2 = wildcard_atoms[0].GetIdx(), wildcard_atoms[1].GetIdx()
        neighbor1 = wildcard_atoms[0].GetNeighbors()[0].GetIdx()
        neighbor2 = wildcard_atoms[1].GetNeighbors()[0].GetIdx()
        rw_mol.AddBond(neighbor1, neighbor2, Chem.BondType.SINGLE)
        rw_mol.RemoveAtom(max(idx1, idx2))
        rw_mol.RemoveAtom(min(idx1, idx2))
        
        mol_ring = rw_mol.GetMol()
        Chem.SanitizeMol(mol_ring)
        final_smiles = Chem.MolToSmiles(mol_ring)
        logging.info(f"    -> Итоговый SMILES кольца: {final_smiles}")
        
        mol_ring = Chem.AddHs(mol_ring)
        
        params = AllChem.ETKDGv3()
        params.maxIterations = 200
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_ring, params) == -1:
            logging.warning(f"Не удалось встроить 3D координаты для кольца: {final_smiles}")
            return None
        
        AllChem.MMFFOptimizeMolecule(mol_ring)
        
        positions = mol_ring.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol_ring.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0], pbc=True)
        borders = np.max(atoms.get_positions(), axis=0) - np.min(atoms.get_positions(), axis=0) + 2 * (r_max + 1)
        atoms.set_cell(borders)
        atoms.translate(borders * 0.5)
        
        # Добавляем метаданные
        atoms.info['name'] = f'ring_n{n}'
        atoms.info['generation_type'] = 'ring'
        atoms.info['length'] = n

        return atoms
    except Exception as e:
        logging.error(f"Ошибка при генерации кольца n={n}: {e}", exc_info=True)
        return None

def _generate_strained_linear_oligomer(polymer_smiles: str, n: int, strain: float, bond_buffer: float = 2.0, r_max=8.0, _seed=42, _try_number=0) -> Optional[Atoms]:
    """Генерирует 3D структуру для линейного олигомера, растянутого вдоль оси X."""
    if _try_number == 0:
        logging.info(f"  -> Генерация растянутой цепи (n={n}, strain={strain:.3f}, bond_buffer={bond_buffer})")

    monomer_body_smiles = polymer_smiles.replace('[*]', '')
    monomer_head = polymer_smiles[::-1].replace(']*[','',1)[::-1]
    monomer_tail = polymer_smiles.replace('[*]', '', 1)
    monomer_body = polymer_smiles.replace('[*]', '')
    chain_smiles = monomer_head + monomer_body*(n-2) + monomer_tail
    
    MIN_DISTANCE = 1.0
    MAX_TRIES = 100
    
    if _try_number >= MAX_TRIES:
        logging.error(f"Ошибка при генерации растянутой цепи n={n}, strain={strain}: не получилось сгенерировать линейный конформер")
        return None

    try:
        # 1. Генерация 3D структуры с помощью RDKit
        mol = Chem.MolFromSmiles(chain_smiles)
        wildcard_polymers = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
        idx1, idx2 = wildcard_polymers[0].GetIdx(), wildcard_polymers[1].GetIdx()
        #rw_mol = Chem.RWMol(mol)
        for wildcard in wildcard_polymers:
            wildcard.SetAtomicNum(1)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.maxIterations = 200
        params.randomSeed = _seed
        if AllChem.EmbedMolecule(mol, params) == -1:
            logging.warning(f"Не удалось встроить 3D координаты для цепи n={n}")
            return None
        
        AllChem.MMFFOptimizeMolecule(mol)
        rw_mol = Chem.RWMol(mol)
        
        rw_mol.RemoveAtom(max(idx1, idx2))
        rw_mol.RemoveAtom(min(idx1, idx2))
        mol = rw_mol
        
        
        # 2. Надёжное определение концов основной цепи
        # Анализируем шаблон мономера, чтобы найти атомы, присоединенные к [*]
        mol_template = Chem.MolFromSmiles(polymer_smiles)
        wildcards = [a for a in mol_template.GetAtoms() if a.GetAtomicNum() == 0]
        if len(wildcards) != 2:
            logging.error("Шаблон SMILES должен содержать ровно 2 [*].")
            return None
        
        # Индексы "атомов сшивки" в рамках шаблона мономера
        attachment_indices_in_template = [w.GetNeighbors()[0].GetIdx() for w in wildcards]
        
        # Находим все вхождения мономера в длинную цепь
        mol_monomer_body = Chem.MolFromSmiles(monomer_body_smiles)
        matches = mol.GetSubstructMatches(mol_monomer_body, uniquify=False)
        
        if len(matches) < n:
            logging.error(f"Не удалось найти {n} мономеров в цепи. Найдено {len(matches)}.")
            return None

        # Определяем индексы конечных атомов в полной молекуле
        first_monomer_map = matches[0]
        last_monomer_map = [match for match in matches if match[0] < match[1]][-1] #такая сложная штука, чтобы удостовериться, что мы получили действительно последний мономер
        
        # Примечание: предполагается, что первый [*] в SMILES - начало, второй - конец
        end_atom1_idx = first_monomer_map[attachment_indices_in_template[0]-1]
        end_atom2_idx = last_monomer_map[attachment_indices_in_template[1]-1]

        # 3. Создание объекта Atoms и ориентация вдоль оси X
        positions = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.center()

        main_axis_vector = atoms.positions[end_atom2_idx] - atoms.positions[end_atom1_idx]
        
        atoms.rotate(main_axis_vector, (1, 0, 0), center='COM')
        
        # 4. Установка периодической ячейки
        min_coords = np.min(atoms.positions, axis=0)
        max_coords = np.max(atoms.positions, axis=0)
        cell_dims = max_coords - min_coords
        #cell_x = cell_dims[0] + bond_buffer # Используем настраиваемый буфер
        cell_x = (atoms.positions[end_atom2_idx] - atoms.positions[end_atom1_idx])[0] + bond_buffer
        cell_y = cell_dims[1] + 2 * r_max
        cell_z = cell_dims[2] + 2 * r_max
        atoms.set_cell([cell_x, cell_y, cell_z])
        atoms.set_pbc([True, False, False])
        atoms.center(about=(0, cell_y / 2, cell_z / 2))
        
        distances = atoms.get_all_distances(mic=True)
        mask = np.ones(distances.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        min_distance_observed = distances[mask].min()
        
        if min_distance_observed < MIN_DISTANCE:
            logging.info(f"Попытка {_try_number} сгенерировать конформер не удалась. Минимальное расстояние - {min_distance_observed:.3f}")
            return _generate_strained_linear_oligomer(polymer_smiles, n, strain, bond_buffer, r_max, _seed+1, _try_number+1)

        # 5. Применение растяжения
        original_positions = atoms.get_positions()
        new_positions = original_positions.copy()
        h_bond_map = {h.GetIdx(): h.GetNeighbors()[0].GetIdx() for h in mol.GetAtoms() if h.GetSymbol() == 'H'}
        heavy_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']
        heavy_displacements = {}

        for idx in heavy_atom_indices:
            orig_pos = original_positions[idx]
            new_pos = np.array([orig_pos[0] * (1 + strain), orig_pos[1], orig_pos[2]])
            new_positions[idx] = new_pos
            heavy_displacements[idx] = new_pos - orig_pos
        
        for h_idx, heavy_idx in h_bond_map.items():
            if heavy_idx in heavy_displacements:
                new_positions[h_idx] = original_positions[h_idx] + heavy_displacements[heavy_idx]

        # 6. Создание финального объекта Atoms
        strained_atoms = atoms.copy()
        strained_atoms.set_positions(new_positions)
        new_cell = strained_atoms.get_cell()
        new_cell[0, 0] *= (1 + strain)
        strained_atoms.set_cell(new_cell, scale_atoms=False)
        strained_atoms.wrap()

        strained_atoms.info['name'] = f'linear_strained_n{n}_s{strain:.3f}_b{bond_buffer:.2f}'
        strained_atoms.info['generation_type'] = 'linear_strained'
        strained_atoms.info['length'] = n
        strained_atoms.info['strain'] = strain
        strained_atoms.info['bond_length'] = bond_buffer
        
        return strained_atoms

    except Exception as e:
        logging.error(f"Ошибка при генерации растянутой цепи n={n}, strain={strain}: {e}", exc_info=True)
        return None

def smiles_to_ase_atoms(polymer_smiles: str, generation_config: Dict) -> List[Atoms]:
    """
    Главная функция-диспетчер для генерации синтетических структур по конфигу.
    """
    if polymer_smiles.count('[*]') != 2:
        raise ValueError(f"SMILES полимера должен содержать ровно две точки сшивки '[*]'. Получено: {polymer_smiles}")

    ase_atoms_list = []
    
    if generation_config.get("monomer", False):
        monomer_atoms = _generate_monomer(polymer_smiles)
        if monomer_atoms: ase_atoms_list.append(monomer_atoms)

    if "rings" in generation_config:
        for n in generation_config["rings"]:
            if n < 2:
                logging.warning(f"Длина кольца n={n} некорректна (< 2). Пропуск.")
                continue
            ring_atoms = _generate_ring(polymer_smiles, n)
            if ring_atoms: ase_atoms_list.append(ring_atoms)

    if "linear" in generation_config:
        for n in generation_config["linear"]:
            if n < 2:
                logging.warning(f"Длина линейного олигомера n={n} некорректна (< 2). Пропуск.")
                continue
            linear_atoms = _generate_linear_oligomer(polymer_smiles, n)
            if linear_atoms: ase_atoms_list.append(linear_atoms)

    if "linear_strained" in generation_config:
        ls_config = generation_config["linear_strained"]
        linear_sizes = ls_config.get("linear_sizes", [])
        strain_params = ls_config.get("strain_range", [])
        # <<< ИЗМЕНЕНИЕ ЗДЕСЬ: считываем новый параметр bond_length >>>
        bond_buffers = ls_config.get("bond_lengths", [2.0]) # По умолчанию 2.0 Ангстрем

        if not linear_sizes:
            logging.warning("В 'linear_strained' отсутствует или пуст ключ 'linear_sizes'.")
        if not bond_buffers:
            logging.warning("В 'linear_strained' отсутствует или пуст ключ 'bond_lengths'. Ставим 2.0 по умолчанию. ")
        if len(strain_params) != 3:
            logging.warning(f"Ключ 'strain_range' должен содержать 3 элемента [min, max, step]. Получено: {strain_params}")
        else:
            strains = np.arange(strain_params[0], strain_params[1], strain_params[2])
            for n in linear_sizes:
                if n < 2:
                    logging.warning(f"Длина растягиваемой цепи n={n} некорректна (< 2). Пропуск.")
                    continue
                for strain_val in strains:
                    for bond_buffer in bond_buffers:
                    # <<< ИЗМЕНЕНИЕ ЗДЕСЬ: передаем bond_buffer в функцию >>>
                        strained_atoms = _generate_strained_linear_oligomer(
                            polymer_smiles, n, strain_val, bond_buffer=bond_buffer
                        )
                        if strained_atoms:
                            ase_atoms_list.append(strained_atoms)
            
    return ase_atoms_list
