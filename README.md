# PolyPot: Automated Interatomic Potential Generation for Polymers

PolyPot is a powerful and flexible command-line tool designed to automate the data-driven generation of machine-learned interatomic potentials (MLIPs), specifically Moment Tensor Potentials (MTP), for molecular dynamics (MD) simulations of polymers.

The core philosophy of this package is to build a high-quality, relevant training dataset for a target polymer by intelligently selecting structures from a vast, pre-existing database of quantum mechanical calculations. It then trains an initial potential and refines it through an active learning loop, ensuring the potential is robust and accurate for the specific polymer system of interest.

## Key Features

-   **Data-Driven Dataset Creation**: Automatically selects the most relevant atomic configurations for a target polymer (defined by a SMILES string) from large, generic datasets.
-   **Intelligent Structure Selection**: Uses MACE-based atomic fingerprints, UMAP for dimensionality reduction, and clustering algorithms to identify and sample diverse and relevant chemical environments.
-   **Automated MTP Training**: Integrates with the `MLIP` package to train Moment Tensor Potentials on the generated dataset.
-   **Active Learning Loop**: Iteratively improves the potential by:
    -   Validating the potential's extrapolation grade on generated polymer structures.
    -   Running LAMMPS MD simulations to sample new configurations where the potential is uncertain.
    -   Automatically selecting the most uncertain structures for high-accuracy *ab initio* recalculation (integrates with VASP).
    -   Retraining the potential with the newly acquired data.
-   **Synthetic Structure Generation**: Generates various initial polymer structures (monomers, linear oligomers, rings, strained chains) directly from a SMILES string using RDKit and ASE.
-   **Extensive Configuration**: A single JSON file controls the entire workflow, from data sources and model paths to training parameters and active learning thresholds.
-   **Built-in Caching and Logging**: Caches intermediate results like fingerprints to speed up subsequent runs and provides detailed logging for traceability.

## Workflow Overview

The tool follows a multi-stage pipeline, which can be visualized as follows:

1.  **Input**: The user provides a `config.json` file specifying the target polymer's SMILES string, paths to source datasets, and various parameters for each stage.

2.  **Fingerprinting**: The source datasets (`.cfg` files) are processed. Atomic environments in each configuration are converted into high-dimensional feature vectors (fingerprints) using a pre-trained MACE model. This stage is efficiently cached.

3.  **Query Generation**: A set of "query" structures (monomer, ring, linear, strained oligomers) is generated in 3D from the target polymer's SMILES string. Fingerprints are calculated for these query structures.

4.  **Relevant Dataset Selection**: The core of the data selection process. The algorithm identifies which atomic environments in the vast source datasets are most similar to those in the query structures. It does this by:
    a. Clustering fingerprints within each source configuration.
    b. Reducing the dimensionality of these cluster centroids using UMAP.
    c. Performing a final clustering on the low-dimensional UMAP embedding.
    d. Identifying which clusters are "relevant" by finding the nearest neighbors to the query fingerprints.
    e. Proportionally selecting a diverse set of configurations from these relevant clusters to create a `train.cfg` file.

5.  **Initial MTP Training**: An initial Moment Tensor Potential (`.mtp` file) is trained using the generated `train.cfg` dataset.

6.  **Validation & Grading**: The quality of the newly trained potential is assessed by calculating the "extrapolation grade" for each of the query structures. This grade quantifies how confident the potential is when predicting forces and energies for that structure.

7.  **Active Learning Loop (Optional)**: If enabled, the tool enters a loop to iteratively refine the potential:
    a. **MD Sampling (LAMMPS)**: Based on the calculated grades, MD simulations are run for the query structures. Structures with high uncertainty (high grades) are chosen as starting points to explore new areas of the potential energy surface. LAMMPS is used to find new, high-energy, or "extrapolative" configurations.
    b. **Selection (`select-add`)**: The most uncertain configurations from the MD runs are selected.
    c. **Ab Initio Calculation (VASP)**: These selected configurations are passed to a quantum mechanics engine (VASP) to calculate accurate energies and forces.
    d. **Dataset Augmentation**: The new, accurately calculated configurations are added to the training set.
    e. **Retraining**: The MTP is retrained with the augmented dataset.
    f. The loop repeats, starting again with the validation of the newly retrained potential.

8.  **Output**: The final products are the optimized training dataset (`train.cfg`) and the robust, trained Moment Tensor Potential (`trained.mtp`), ready for use in large-scale MD simulations.

## Prerequisites

### Python Libraries
The tool relies on several Python libraries. You can install them using pip:```bash
pip install numpy ase tqdm scikit-learn umap-learn matplotlib rdkit
```
Additionally, the MACE-calculator is required. Please follow its official installation instructions.

### External Software
This package is a wrapper that orchestrates several external computational chemistry codes. You must have them installed and accessible in your environment.

-   **MLIP**: The core package for training MTP. The `mlp` executable must be in your system's PATH or its path must be specified in `config.json`.
-   **LAMMPS**: Used for the MD sampling part of the active learning loop. The `lmp` executable path is required in `config.json`.
-   **VASP**: (Optional, for active learning) Used for the *ab initio* calculations. The `vasp_std` executable path and input files (INCAR, KPOINTS, POTCARs) are required.

## Configuration (`config.json`)

The entire workflow is controlled by a single `config.json` file. Below is a detailed explanation of each section.

| Section | Key | Description |
| :--- | :--- | :--- |
| **`general`** | `smiles_polymer` | **Required.** The SMILES string of the polymer repeating unit. Must contain exactly two wildcard atoms `[*]` to indicate linking points. |
| | `output_dir` | **Required.** Path to the directory where all output files (logs, datasets, potentials) will be saved. |
| | `device` | The compute device for MACE fingerprinting (`"cpu"` or `"cuda"`). |
| **`fingerprinting`** | `datasets` | A list of source dataset objects. Each object needs: `path` (to the `.cfg` file), `fingerprints_cache` (path to save/load cached fingerprints), and `sampling_rate_N` (process every Nth configuration). |
| | `mace_model_path` | Path to the pre-trained MACE model file used for generating atomic fingerprints. |
| | `type_map_cfg_to_symbol` | A dictionary mapping the integer atom types in the `.cfg` files to their chemical symbols (e.g., `{"0": "C", "1": "H"}`). |
| **`query_generation`** | `monomer`, `rings`, `linear`| Defines which types of synthetic query structures to generate from the SMILES string. `rings` and `linear` take a list of oligomer lengths (n-mers). |
| | `linear_strained` | Generates stretched linear chains. `linear_sizes` is a list of lengths, `strain_range` is `[min, max, step]`, and `bond_lengths` is a list of distances between chain ends. |
| **`selection`** | `num_output_configs` | The target total number of atomic configurations in the final `train.cfg` file. |
| | `intra_config_clusters` | Number of local environments (clusters) to identify within each single source configuration. |
| | `umap_params` | Parameters for the UMAP dimensionality reduction algorithm (`n_neighbors`, `min_dist`, `n_components`). |
| | `clustering_params` | Parameters for the final Agglomerative Clustering step (`distance_threshold`, `n_clusters`). |
| **`mtp_training`** | `enabled` | If `true`, the MTP training stage will be executed. |
| | `run_params` | Optional execution prefix for the training command (e.g., `srun -N 3 -n 72` for SLURM). |
| | `mtp_executable_path` | **Required.** Absolute path to the `mlp` executable. |
| | `initial_potential` | Path to an existing potential to start the training from. |
| | `output_potential_name` | Filename for the trained potential, saved in the `output_dir`. |
| | `training_command` | The command for MLIP (usually `"train"`). |
| | `training_params` | A dictionary of command-line arguments to pass to the MLIP training command (e.g., `energy-weight`, `max-iter`). |
| **`mtp_validation`** | `enabled` | If `true`, enables the calculation of extrapolation grades. Required for active learning. |
| | `query_cfg_filename` | Filename for the generated query structures, saved in `.cfg` format. |
| **`postprocessing`** | `...` | Options to save intermediate and final datasets in `.xyz` format and to generate a UMAP visualization plot. |
| **`active_learning`** | `enabled` | If `true`, the active learning loop will start after the initial training. |
| | `n_iterations` | The maximum number of active learning cycles to perform. |
| | `thresholds` | `md_start`: Grade above which MD sampling is initiated. `md_break`: Grade above which the structure is considered highly extrapolative and sent directly for *ab initio* calculation (if MD also runs). |
| | `md_sampler_config` | Configuration for LAMMPS runs, including the `lammps_executable_path`, `steps`, `temperature`, and `max_parallel_processes`. |
| | `ab_initio_config` | Configuration for VASP runs, including the `executable_path` and paths to `INCAR`, `KPOINTS`, and `POTCAR` files. |

## Usage

1.  **Prepare your environment**: Make sure all prerequisite software and libraries are installed and accessible.
2.  **Create your datasets**: You need at least one large `.cfg` file containing atomic configurations with energies and forces calculated at the *ab initio* level (e.g., DFT).
3.  **Configure `config.json`**: Create a `config.json` file and edit it according to your system, paths, and desired workflow. Pay close attention to the file paths and the polymer SMILES string.
4.  **Run the pipeline**: Execute the main script from your terminal, passing the configuration file as an argument.

```bash
python main.py /path/to/your/config.json
```

The tool will then start executing the pipeline, printing detailed logs to both the console and a `run.log` file inside your specified `output_dir`. The process can be lengthy, especially the training and active learning stages.

## Code Structure

The project is organized into several modules, each responsible for a specific part of the workflow:

-   `main.py`: The main entry point of the application. It parses the config file and orchestrates the calls to other modules in the correct sequence.
-   `configuration.py`: A utility class (`Configuration`) for parsing, manipulating, and writing MLIP's `.cfg` file format.
-   `fingerprinting.py`: Handles the calculation of MACE fingerprints for all configurations in the source datasets.
-   `selection.py`: Implements the core logic for selecting the relevant training dataset.
-   `training.py`: A wrapper for executing the MTP training command.
-   `validation.py`: A wrapper for the `mlp calc-grade` command to assess potential quality.
-   `active_learning.py`: Contains the main logic for the active learning loop, coordinating between validation, MD sampling, and *ab initio* calculations.
-   `md_sampler.py`: A wrapper for running LAMMPS MD simulations.
-   `ab_initio.py`: A wrapper for running VASP calculations.
-   `utils.py`: Contains helper functions, most notably the `smiles_to_ase_atoms` function for generating 3D polymer structures.
-   `visualization.py`: Generates the UMAP plot for visualizing the fingerprint space.