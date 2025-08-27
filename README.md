# README for Emergent Directedness in Social Contagion Code

This repository contains Python code for simulations and visualizations related to the paper "Emergent Directedness in Social Contagion" by Fabian Tschofenig and Douglas Guilbeault (August, 2025). The code implements the causal inference framework described in the paper, which simulates network pathways for complex contagions and identifies causally impactful edges and nodes. It generates datasets and figures illustrating emergent directedness, asymmetry in weak ties, inversions in centrality, and other key findings.

## Main Package
- **CPC_package.py**: The core package (Causal Path Contagion package) used by all simulations and figures to generate results. Import this package in scripts to access necessary functions and classes for modeling complex contagions under the General Influence Model, Linear Threshold Model, Independent Cascade Model and Noisy Threshold-based Contagions. Located in `./Emergent_Directedness_Code/Asymmetry paper`.

## Data Sources
The simulations utilize datasets from two empirical studies:
- Banerjee graphs: Derived from the study on the diffusion of Microfinance. The dataset can be downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U3BIHX. The dataset is located in the folder `./datav4.0/Data/1. Network Data/Adjacency Matrices/`.
- AddHealth graphs: Giant components extracted from the Add Health dataset. The original dataset is available from https://addhealth.cpc.unc.edu/data/. The processed data is available in the folder `./AddHealth_Networks_Largest_Components/`.

## Folder Structure
The code is organized into the following folders, each containing scripts for calculations (simulations and dataset generation) and visualizations (figure production). Scripts typically generate intermediate datasets that are then used by visualization notebooks to produce figures for the paper. All calculation and visualization scripts are physically located in `./Emergent_Directedness_Code`.

### asymmetry_threshold_correlation
This folder contains scripts for analyzing asymmetry in thresholds and correlations, revealing how complex contagions induce directed pathways.

- **Calculation Scripts** (located in `./Emergent_Directedness_Code`):
  - `asymmetry_calculations_ICM_only.py`: Generates datasets for Figure 13 (asymmetry in Independent Cascade Model variants).
  - `asymmetry_calculations_increasing_thresholds.py`: Generates datasets for Figures 2 (basic asymmetry), 5b (comparison with tie strength), 8 (weak tie directedness), 9 (noisy thresholds), 10 (stochastic effects), and 14 (threshold scaling).
  - `asymmetry_calculations_NOISY.py`: Generates datasets for Figures 9 and 10 (incorporating noisy stochastic subthreshold adoptions).
  - `asymmetry_calculations_LTM_only.py`: Generates datasets for Figures 11 and 12 (asymmetry in Linear Threshold Model variants).

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `visualize_asymmetry_threshold.ipynb`: Uses datasets from the calculation scripts to produce Figures 2, 5b, 8, 9, 10, 11, 12, 13, and 14.

### bridge_building
This folder handles simulations of bridge formation and their role in directed flow.

- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `bridge_building_calculations.ipynb`: Runs simulations and produces datasets for bridge structures.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `bridge_building_visualization.ipynb`: Uses the generated datasets to create Figure 6 (directed bridges in small-world networks).

### causal_path_visualization
This folder focuses on visualizing causal paths and emergent directedness.

- **Script** (located in `./Emergent_Directedness_Code`):
  - `causal_path_visualization.ipynb`: Contains both simulations and visualizations to produce Figure 20 (extended causal path chains).

### comparison_plots
This folder generates comparison plots for tie strength and contagion types.

- **Script** (located in `./Emergent_Directedness_Code`):
  - `comparison_plots.ipynb`: Contains simulations and visualizations to produce Figure 5a (comparing simple vs. complex contagions).

### convergence
This folder analyzes convergence of node/edge rankings across simulations.

- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `convergence_calculations.py`: Runs simulations and creates datasets for sampling stability.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `convergence_visualization.ipynb`: Uses the datasets to produce Figures 15, 16, and 17 (convergence of causal importance measures).

### dip_calculation
This folder performs calculations for "dip" effects, likely related to nonlinearity in tie impacts.

- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `dip_calculation.py`: Runs simulations and creates datasets.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `visualize_dip_calculations.ipynb`: Uses the datasets to produce Figure 22 (dip in influence patterns).

### heatmaps_tie_ranges
This folder generates heatmaps and analyzes tie ranges for weak ties.

- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `calculations_heatmaps.py`: Runs simulations and creates datasets.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `heatmaps_and_tie_ranges_visualization.ipynb`: Uses the datasets to produce Figures 3a to 3e (heatmaps of asymmetry) and 18 (tie range distributions).

### inverse_V_and_Tie_Range
This folder handles inverse V (inverted U-shape) calculations and tie range analyses, explaining nonlinearity in weak tie impacts (e.g., from LinkedIn job diffusion study).

- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `inverse_V_calculation.py`: Runs simulations and creates datasets.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `inverse_V_visualization.R`: Uses the datasets to produce Figures 4A (inverted U-shape) and 21 (tie range inversions).

## Usage Instructions
### **Dependencies**: 
Ensure Python 3.x is installed, the code was tested on Python 3.12.2. 

Required libraries include:
  - numpy==2.3.2
  - scipy==1.16.1
  - pandas==2.3.0
  - matplotlib==3.10.0
  - seaborn==0.13.2
  - networkx==3.4.2
  - scikit-learn==1.7.1
  - joblib (latest version recommended)
  - tqdm==4.67.1
  - pathos==0.3.3
  - plotly==6.0.1
  - statsmodels==0.14.5
  - psutil (latest version recommended)
1. Install via `pip install numpy scipy pandas matplotlib seaborn networkx scikit-learn joblib tqdm pathos plotly statsmodels psutil`.
2. Import `CPC_package.py` in any script or notebook as needed for core functions like seed set generation (RRS/RCS), threshold models, and causal importance calculations.
3. Run calculation scripts first to generate datasets (outputs saved as joblib files in the folder). Then, run visualization notebooks to produce figures (saved as PNG).
4. Example workflow: To reproduce Figure 2, run `asymmetry_calculations_increasing_thresholds.py` then `visualize_asymmetry_threshold.ipynb`.

## Notes
### Key Insights Reproduced: 
The code enables replication of core findings, including how complex contagions create asymmetric paths in undirected networks, preferential one-way spread via weak ties leading to cultural transmission inequalities, inversion of standard centrality (periphery-to-core flow), nonlinearity in tie strength effects, and biases in network rewiring toward directed pathways, moderated by factors like triadic closure.
### Limitations and Extensions:
 Simulations emphasize threshold-based complex contagions; extensions to other models (e.g., noisy or probabilistic) are included in specific scripts.
### For questions or issues:
contact the authors at tschofe@stanford.edu or dguilb@stanford.edu 
