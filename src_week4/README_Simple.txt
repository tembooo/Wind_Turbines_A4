Wind Turbine PCA and Kernel PCA Analysis (MSPC Project)
------------------------------------------------------
Team: Lazy Geniuses (A4) | LUT University | ADML Course 2025

Overview:
This project performs fault detection in wind turbines using PCA and Kernel PCA.
It uses healthy-based scaling, contiguous cross-validation, and computes FAR, ARL, and detection metrics.

How to Run:
1. Open MATLAB and go to the project folder.
2. Make sure analysis/ and data_loading/ folders are added to path.
3. Run the script: Main_mspc_analysis.m

Folder Structure:
- analysis/        : all PCA, kPCA, and validation functions
- data_loading/    : preprocessing and time-aware cleaning utilities
- data/            : Excel data file (e.g., data.xlsx)
- Main_mspc_analysis.m : main driver script to run everything
- Main_pca_analysis.m  : PCA only analysis (optional)
- dataset_analysis.m   : checks dataset dimensions

Main Steps:
1. Load turbine data (healthy and faulty).
2. Apply time-aware preprocessing (interpolate small NaN gaps).
3. Perform PCA and Kernel PCA using healthy scaling.
4. Compute T² and SPE limits.
5. Validate using 5-fold contiguous CV (FAR and ARL).
6. Plot results and compare PCA vs kPCA performance.

Output:
- Figures are saved automatically.
- Console shows FAR, ARL, and key PCA/kPCA statistics.

Developed by:
Arman Golbidi, Fasie Haider, Haider Ali
