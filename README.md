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
- **Calculation Scripts** (located in `./Emergent_Directedness_Code`):
  - `asymmetry_calculations_ICM_only.py`: Generates datasets for Figure S5.
  - `asymmetry_calculations_increasing_thresholds.py`: Generates datasets for Figures 2, 5b, S2, S6, S7, S16
  - `asymmetry_calculations_NOISY.py`: Generates datasets for Figures S6 and S7.
  - `asymmetry_calculations_LTM_only.py`: Generates datasets for Figures S3 and S4.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `visualize_asymmetry_threshold.ipynb`: Uses datasets from the calculation scripts to produce Figures 2, 5b, S2, S3, S4, S5, S6, S7, S16.

### bridge_building
- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `bridge_building_calculations.ipynb`: Runs simulations and produces datasets for bridging simulations.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `bridge_building_visualization.ipynb`: Uses the generated datasets to create Figure 6.

### causal_path_visualization
- **Script** (located in `./Emergent_Directedness_Code`):
  - `causal_path_visualization.ipynb`: Contains both simulations and visualizations to produce Figure S14.

### comparison_plots
- **Script** (located in `./Emergent_Directedness_Code`):
  - `comparison_plots.ipynb`: Contains simulations and visualizations to produce Figure 5a.

### convergence
- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `convergence_calculations.py`: Runs simulations and creates datasets for convergence analysis.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `convergence_visualization.ipynb`: Uses the datasets to produce Figures S9, S10, and S11.

### dip_calculation
- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `dip_calculation.py`: Runs simulations and creates datasets.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `visualize_dip_calculations.ipynb`: Uses the datasets to produce Figure S8.

### heatmaps_tie_ranges
- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `calculations_heatmaps.py`: Runs simulations and creates datasets.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `heatmaps_and_tie_ranges_visualization.ipynb`: Uses the datasets to produce Figures 3a to 3e and S12.

### inverse_V_and_Tie_Range
- **Calculation Script** (located in `./Emergent_Directedness_Code`):
  - `inverse_V_calculation.py`: Runs simulations and creates datasets.

- **Visualization Script** (located in `./Emergent_Directedness_Code`):
  - `inverse_V_visualization.R`: Uses the datasets to produce Figures 4A and S15.

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
### Limitations and Extensions:
 Simulations emphasize threshold-based complex contagions; extensions to other models (e.g., noisy or probabilistic) are included in specific scripts.
### For questions or issues:
contact the authors at tschofe@stanford.edu or dguilb@stanford.edu 
