import logging
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

def smiles_to_ase_atoms(smiles: str, num_conformers: int = 1) -> List[Atoms]:
    """
    Генерирует 3D структуры из SMILES и конвертирует их в объекты ase.Atoms.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"Не удалось обработать SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    
    # Генерируем несколько конформеров для большего разнообразия
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=AllChem.ETKDGv3())
    
    # Оптимизируем каждый конформер
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    
    ase_atoms_list = []
    for cid in cids:
        positions = mol.GetConformer(cid).GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # Создаем большую ячейку, чтобы избежать взаимодействия при расчете фингерпринтов
        atoms = Atoms(symbols=symbols, positions=positions, cell=[100.0, 100.0, 100.0], pbc=True)
        ase_atoms_list.append(atoms)
        
    return ase_atoms_list