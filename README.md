⚠️ **Note on Folder Structure (Important for Professor)**  

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
