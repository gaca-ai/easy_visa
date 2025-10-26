# 🧠 EasyVisa — Visa Approval Prediction

## 📄 Project Context

Business communities in the United States face a continuous challenge in identifying and attracting the right talent to remain competitive. To fill skill gaps, companies often seek foreign workers under the regulations of the **Immigration and Nationality Act (INA)**, administered by the **Office of Foreign Labor Certification (OFLC)**.

The OFLC reviews employer applications for foreign labor certification to ensure U.S. workers are not adversely affected and that hiring conditions meet statutory wage and employment standards.  

In FY 2016 alone, OFLC processed **775,979 applications** for **1.7 million positions** — a 9% increase from the previous year — making manual review increasingly time-consuming.

To streamline this process, **EasyVisa**, a data-driven consulting firm, has been tasked with developing a **machine learning model** to help **predict the likelihood of visa approval** and **identify key drivers** influencing outcomes.

---

## 🎯 Project Objective

Build a **classification model** to:
- Predict whether a visa application will be **certified** or **denied**.
- Identify which factors (education, wage, experience, company size, etc.) most strongly influence approval decisions.

---

## 🔍 Guiding Questions

The exploratory and modeling analysis addresses the following key questions:

1. What is the distribution of visa case statuses (**certified vs. denied**)?
2. How does the **education level** of employees impact visa approval rates?
3. Is there a significant difference in approval rates between employees **with and without job experience**?
4. How does the **prevailing wage** affect visa approval?  
   → Do higher wages lead to higher chances of approval?
5. Do certain **U.S. regions** have higher visa approval rates than others?
6. How does the **number of employees** in a company influence visa approval?  
   → Do larger companies have higher approval rates?
7. Are approval rates different across **continents of employees**?  
   → Which continent shows the highest and lowest approval rates?

---

## 📊 Dataset Description

The dataset contains **26,000 records** with attributes describing both **employees** and **employers**.

| Column | Description |
|---------|-------------|
| `case_id` | ID of each visa application |
| `continent` | Continent of the employee |
| `education_of_employee` | Education level of the employee |
| `has_job_experience` | Y/N — prior job experience |
| `requires_job_training` | Y/N — requirement for job training |
| `no_of_employees` | Number of employees in the company |
| `yr_of_estab` | Year company was established |
| `region_of_employment` | U.S. region of intended employment |
| `prevailing_wage` | Average wage for the occupation in that region |
| `unit_of_wage` | Hourly, Weekly, Monthly, or Yearly |
| `full_time_position` | Y/N — full-time or part-time |
| `case_status` | Target variable — Certified / Denied |

Dataset size: ~2 MB  
Source: Provided by EasyVisa (OFLC historical data)

---

## ⚙️ Environment & Execution

The project runs entirely in **Google Colab** for improved computation speed.

To open and execute:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaca-ai/easy_visa/blob/main/GACA_Project_Full_Code_EasyVisa.ipynb)

```python
import pandas as pd

url = "https://raw.githubusercontent.com/gaca-ai/easy_visa/main/data/easyvisa.csv"
df = pd.read_csv(url)
df.head()
