# Global Logistics Movement - Week 2: Data Collection, Cleaning & Preprocessing

This repository contains the official Week 2 technical submission for the Logistics Data Analyst Internship at YuvaIntern. Following the strategic planning phase, this project establishes a robust, programmatic data preprocessing pipeline designed to transform raw, noisy operational logistics data into a high-quality, normalized dataset for supply chain intelligence.

---

## 🚀 Executive Summary & Objective
In modern supply chain analytics, unrefined operational data introduces cascading failures—ranging from miscalculated safety stocks to inaccurate ETAs and flawed resource allocations. This module builds an automated Python pipeline using `pandas` and `scikit-learn` to ingest, diagnose, clean, and normalize global logistics movement data.

---

## 📁 Repository Structure
* **`Week2_Logistics_Data_Preprocessing_Report.docx`**: Comprehensive technical documentation detailing schema definitions, methodological strategies, code snippets, and business impact reflections.
* **`preprocessing.py`**: Production-grade Python script executing structural cleaning, conditional missing value imputation, IQR Winsorization, and feature scaling.

---

## 📊 Dataset Simulation & Schema (`global_logistics_movement.csv`)
The project models a global supply chain performance dataset consisting of **50,000 records** across a 12-month temporal scope, with intentionally injected anomalies for rigorous testing:

| Feature Name | Data Type | Description | Potential Issues Modeled |
|---|---|---|---|
| **Shipment_ID** | String / Object | Unique alphanumeric identifier for each order. | Duplicate records, formatting inconsistencies. |
| **Origin_Hub** | Categorical | Country/City of dispatch node. | Typos, case-sensitivity issues (`london` vs `London`). |
| **Destination_Hub** | Categorical | Country/City of delivery node. | Missing values due to system logging timeouts. |
| **Planned_Duration_Days** | Integer | Contractual SLA time allocated for delivery. | Structural outliers (e.g., negative duration). |
| **Actual_Duration_Days** | Float | True time elapsed from dispatch to delivery. | Missing data for in-transit shipments; extreme outliers. |
| **Shipment_Weight_KG** | Float | Total gross weight of the cargo. | Skewed distributions; zero/negative error entries. |
| **Transport_Mode** | Categorical | Air, Ocean, Rail, or Road. | Inconsistent naming conventions (`Air` vs `AIR`). |
| **Delivery_Status** | Categorical | Target Variable: Delayed, On-Time, Cancelled. | Imbalanced classes, null values. |

---

## 🛠️ Methodological Pipeline & Treatments

1. **Handling Missing Values:**
   * Critical identifiers with missing values (`Shipment_ID`) are dropped to prevent database contamination.
   * Numerical metrics (`Actual_Duration_Days`) utilize conditional imputation—filling missing elements using the median grouped by specific `Origin_Hub` and `Transport_Mode` to prevent global mean bias.
2. **Outlier Detection & Mitigation (Winsorization via IQR):**
   * Rather than eliminating valuable operational extremes (such as port congestion bottlenecks), an Interquartile Range (IQR) capping mechanism is applied at calculated bounds ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$) to preserve data volume while protecting downstream models.
3. **Feature Normalization & Scaling:**
   * **Min-Max Scaling** transforms bounded features strictly between 0 and 1.
   * **Standard Scaling (Z-score normalization)** centers continuous values around a mean of 0 with a standard deviation of 1, preventing distance-based algorithms from misinterpreting raw magnitude scales.

---

## 💻 Tech Stack & Requirements
* **Language:** Python
* **Libraries & Frameworks:** Pandas, NumPy, scikit-learn
* **Version Control:** Git & GitHub

---
*Author: Aastha Jain*  
*Internship: Logistics Data Analyst Intern at YuvaIntern (October 2026)*
