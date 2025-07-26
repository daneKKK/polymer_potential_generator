import logging
import re
from typing import List, Dict, Optional
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

def _generate_monomer(smiles: str) -> Optional[Atoms]:
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
        atoms = Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0], pbc=True)
        
        # Добавляем метаданные
        atoms.info['name'] = 'monomer'
        atoms.info['generation_type'] = 'monomer'
        
        return atoms
    except Exception as e:
        logging.error(f"Ошибка при генерации мономера: {e}", exc_info=True)
        return None

def _generate_linear_oligomer(polymer_smiles: str, n: int) -> Optional[Atoms]:
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

        # Добавляем метаданные
        atoms.info['name'] = f'linear_n{n}'
        atoms.info['generation_type'] = 'linear'
        atoms.info['length'] = n

        return atoms
    except Exception as e:
        logging.error(f"Ошибка при генерации линейного олигомера n={n}: {e}", exc_info=True)
        return None

def _generate_ring(polymer_smiles: str, n: int) -> Optional[Atoms]:
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
        
        # Добавляем метаданные
        atoms.info['name'] = f'ring_n{n}'
        atoms.info['generation_type'] = 'ring'
        atoms.info['length'] = n

        return atoms
    except Exception as e:
        logging.error(f"Ошибка при генерации кольца n={n}: {e}", exc_info=True)
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
            
    return ase_atoms_list
