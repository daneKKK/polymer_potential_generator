import logging
import re
from typing import List
from ase import Atoms
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

def smiles_to_ase_atoms(
    polymer_smiles: str,
    generate_rings: bool = True,
    max_ring_size: int = 4
) -> List[Atoms]:
    """
    Генерирует 3D структуры из полимерного SMILES, включая мономер
    и небольшие циклические олигомеры, корректно обрабатывая точки сшивки [*].

    Args:
        polymer_smiles (str): SMILES повторяющегося звена, например, '[*]CC([*])c1ccccc1'.
        generate_rings (bool): Если True, генерирует циклические олигомеры.
        max_ring_size (int): Максимальное число звеньев в кольце (например, 3 для тримера).

    Returns:
        List[Atoms]: Список сгенерированных объектов ase.Atoms.
    """
    if polymer_smiles.count('[*]') != 2:
        raise ValueError(f"SMILES полимера должен содержать ровно две точки сшивки '[*]'. Получено: {polymer_smiles}")

    ase_atoms_list = []
    
    # 1. Обработка мономера
    monomer_smiles_str = polymer_smiles.replace('[*]', '').replace('()', '')
    logging.info(f"Генерация 3D структуры для мономера: {monomer_smiles_str}")
    try:
        mol = Chem.MolFromSmiles(monomer_smiles_str)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        
        positions = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0], pbc=True)
        atoms.info['name'] = 'monomer'
        ase_atoms_list.append(atoms)
    except Exception as e:
        logging.warning(f"Не удалось сгенерировать структуру для мономера '{monomer_smiles_str}': {e}")
        return []

    if not generate_rings or max_ring_size < 2:
        return ase_atoms_list

    # 2. Обработка колец
    # Создаем "тело" мономера для вставки в цепочку.
    # Это SMILES, где первая [*] заменена на пустую строку.
    #monomer_head = ''.join(polymer_smiles.rsplit('[*]',1))
    monomer_head = polymer_smiles[::-1].replace(']*[','',1)[::-1]
    monomer_tail = polymer_smiles.replace('[*]', '', 1)
    monomer_body = polymer_smiles.replace('[*]', '')

    for n in range(2, max_ring_size + 1):
        # Шаг 2.1: Собираем SMILES для ЛИНЕЙНОГО олигомера со [*] на концах
        #chain_smiles = polymer_smiles
        #for _ in range(n - 1):
        #    # Последовательно заменяем первую доступную точку сшивки на тело следующего мономера
        #    chain_smiles = chain_smiles.replace('[*]', monomer_body, 1)
        #chain_smiles = chain_smiles.replace('[*]', '').replace('()', '')
        chain_smiles = monomer_head + monomer_body*(n-2) + monomer_tail

        logging.info(f"Генерация кольца (n={n}). Промежуточная цепочка: {chain_smiles}")
        
        try:
            # Шаг 2.2: Создаем RDKit Mol из линейной цепочки
            mol_chain = Chem.MolFromSmiles(chain_smiles)
            if mol_chain is None:
                logging.warning(f"RDKit не смог обработать промежуточный SMILES: {chain_smiles}")
                continue

            # Шаг 2.3: Находим атомы-заглушки для сшивки
            wildcard_atoms = [atom for atom in mol_chain.GetAtoms() if atom.GetAtomicNum() == 0]
            if len(wildcard_atoms) != 2:
                logging.warning(f"В промежуточной цепочке найдено не 2, а {len(wildcard_atoms)} атомов-заглушек. Пропуск.")
                continue
            
            # Шаг 2.4: Создаем редактируемую молекулу и сшиваем кольцо
            rw_mol = Chem.RWMol(mol_chain)
            idx1, idx2 = wildcard_atoms[0].GetIdx(), wildcard_atoms[1].GetIdx()
            
            # Добавляем связь между соседями "звездочек"
            # Сосед у такого атома всегда один
            neighbor1 = wildcard_atoms[0].GetNeighbors()[0].GetIdx()
            #neighbor1 = 
            neighbor2 = wildcard_atoms[1].GetNeighbors()[0].GetIdx()
            rw_mol.AddBond(neighbor1, neighbor2, Chem.BondType.SINGLE)

            # Шаг 2.5: Удаляем сами атомы-заглушки (в обратном порядке, чтобы не сбить индексы)
            rw_mol.RemoveAtom(max(idx1, idx2))
            rw_mol.RemoveAtom(min(idx1, idx2))
            
            # Получаем финальную, чистую молекулу кольца
            mol_ring = rw_mol.GetMol()
            Chem.SanitizeMol(mol_ring)
            final_smiles = Chem.MolToSmiles(mol_ring)
            logging.info(f"  -> Итоговый SMILES кольца: {final_smiles}")
            
            # Шаг 2.6: Генерация 3D структуры и конвертация в ASE
            mol_ring = Chem.AddHs(mol_ring)
             # Создаем объект параметров и задаем в нем нужные опции
            params_ring = AllChem.ETKDGv3()
            params_ring.maxIterations = 200
            params_ring.randomSeed = 42 # Для воспроизводимости

            # Передаем только объект с параметрами
            if AllChem.EmbedMolecule(mol_ring, params_ring) == -1:
                logging.warning(f"Не удалось сгенерировать 3D конформер для кольца: {final_smiles}")
                continue
            
            AllChem.MMFFOptimizeMolecule(mol_ring)

            positions = mol_ring.GetConformer().GetPositions()
            symbols = [atom.GetSymbol() for atom in mol_ring.GetAtoms()]
            atoms = Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0], pbc=True)
            atoms.info['name'] = f'ring_n{n}'
            ase_atoms_list.append(atoms)
            
        except Exception as e:
            logging.error(f"Не удалось сгенерировать структуру для кольца (n={n}): {e}", exc_info=True)
            continue
            
    return ase_atoms_list