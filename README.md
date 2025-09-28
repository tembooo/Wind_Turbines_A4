⚠️ **Note on Folder Structure (Important for Professor)**  for investigation please investigate the finla version. 

To keep our workflow organized, we created two folders for each weekly submission:  

- `1.14_September_2025-09-14_Raw` → All group members upload their individual contributions here.  
- `1.14_September_2025-09-14_Final` → After reviewing and cleaning the raw files, the organized final version for submission is placed here.  

This ensures clarity: the **Raw folder** shows all initial work, while the **Final folder** contains the polished version we submit.




# Wind_Turbines_A4

**Group A4 – Wind Turbine Failure Detection**  
Using SCADA data, we analyze healthy vs faulty turbines with PCA and kernel-PCA models.  
Control charts (T², SPEx) are built for fault detection, sensor diagnostics, and variable contributions to residuals.

**Name of the group:** Lazy Geniuses  
- Fasie Haider  
- Haider Ali  
- Arman Golbidi  

---

# Intermediary Submission 1  Deadline: 14-Sept  

# Wind Turbine SCADA Dataset Structure

- **Sheets (Turbines):**
  - `No.2WT` → Turbine 2 (Healthy / Baseline)
  - `No.3` → Turbine 3 (Faulty)
  - `No.14WT` → Turbine 14 (Faulty)
  - `No.39WT` → Turbine 39 (Faulty)

- **Rows:**  
  Each row represents one time snapshot of data collected every **10 seconds**.  
  Example: 1570 rows in `No.2WT` = 1570 time steps of Turbine 2 data.

- **Columns:**  
  Each column is a sensor variable (e.g., wind speed, power output, temperature, vibration).  
  In this dataset, the sensor names are not provided — columns are labeled **1 … 28**.

---

## Summary Table

| Sheet   | Rows | Cols | Description                |
|---------|------|------|----------------------------|
| No.2WT  | 1570 | 28   | Healthy turbine (baseline) |
| No.3    | xxxx | 28   | Faulty turbine             |
| No.14WT | xxxx | 28   | Faulty turbine             |
| No.39WT | xxxx | 28   | Faulty turbine             |

---

## What is SCADA?

SCADA = **Supervisory Control and Data Acquisition**.  
It is a system for **monitoring and controlling industrial equipment**.  
In wind turbines, SCADA records sensor data such as temperature, pressure, voltage, power, and vibration at fixed intervals (e.g., every 10 seconds).







# Wind Turbine PCA Analysis – ADAML Project

## 🔹 How to Run
The **main entry point** of the project is:

```
Main_pca_analysis.m
```

➡️ Please run this file to reproduce the full workflow.  
It automatically calls all required functions from the `data_loading/` and `analysis/` modules.

---

## 🔹 Repository Structure

```
├── Reports/                # Weekly and final reports
├── src_week1/              # Source code for Week 1 tasks
├── src_week2/              # Source code for Week 2 tasks
```

📌 **Note:** Each week's task is located in its respective folder (`src_week1/`, `src_week2/`).  
All related MATLAB functions and scripts for that week are organized there.  

---

## 🔹 Detailed Project Modules

### 1. **Main Script**
- `Main_pca_analysis.m`: the primary entry point; orchestrates data loading, preprocessing, PCA, and visualization.

### 2. **Data Loading (`src_week1/data_loading/`)**
- `load_turbine_data.m` – Load datasets for Healthy, Faulty1, and Faulty2 turbines.
- `drop_numeric_header_row.m` – Clean numeric headers from Excel sheets.
- `time_aware_preprocess.m` – Handle missing data with short-gap interpolation & long-gap trimming.
- `impute_short_gaps.m`, `select_longest_complete_window.m`, `longest_true_run.m` – Utilities for preprocessing.

### 3. **Analysis (`src_week2/analysis/`)**
- `pca_implementation.m` – Healthy-based PCA, variance explained, eigenvalues.
- `correlation_analysis.m` – Correlation heatmaps and exploratory analysis.
- `plot_time_gradient_scores.m` – Time-gradient PCA score plots per turbine.
- `pca_analysis.m` – Scree plot, cumulative variance, biplot, loading contributions.

---

## 🔹 Team Roles & Responsibilities

### 👤 Arman Golbidi  
**Focus:** Data loading & preprocessing  
- Implemented `load_turbine_data.m` and header-cleaning utility `drop_numeric_header_row.m`.  
- Designed `time_aware_preprocess.m` with gap interpolation (`impute_short_gaps.m`) and window selection (`select_longest_complete_window.m`, `longest_true_run.m`).  
- Ensured data integrity before PCA (handling missing values, row consistency).  

### 👤 Fasie Haider  
**Focus:** PCA model calibration  
- Implemented `pca_implementation.m`.  
- Applied **healthy-based scaling**, removed zero-variance sensors, computed explained variance.  
- Determined number of PCs required for 80/90/95% variance.  
- Reported top contributing variables for PC1 and PC2.  

### 👤 Haider Ali  
**Focus:** Visualization & interpretation  
- Developed `correlation_analysis.m` (correlation matrix, heatmap, distribution plots).  
- Created `plot_time_gradient_scores.m` (PC1–PC2 scores with time gradient & drift arrow).  
- Extended `pca_analysis.m` with scree plots, cumulative variance, PC1 vs PC2 scatter, loadings, biplots, and advanced interpretation of fault separation.  

---

## 🔹 Notes
- The pipeline is **modularized**: each function is reusable and independently testable.  
- **Healthy-based scaling** ensures PCA is not biased by faulty turbines, improving interpretability in score space.  
- Dataset includes three turbines: **Healthy**, **Faulty1**, and **Faulty2**.  

---

📌 **Instructor Reminder:** Please start execution from `Main_pca_analysis.m`.  

