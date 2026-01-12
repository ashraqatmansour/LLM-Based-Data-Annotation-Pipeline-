# LLM-Based Data Annotation Pipeline

This project implements an **LLM-assisted qualitative data annotation pipeline** to analyze and compare topics discussed in Reddit university communities. The pipeline focuses on posts from the **r/mcgill** and **r/concordia** subreddits, combining human open coding with automated support to study thematic differences between the two institutions.

---

## Project Overview

The goal of this project is to:

* Collect Reddit posts from the McGill and Concordia subreddits
* Perform **open coding** to identify recurring themes
* Develop a **typology (codebook)** for consistent annotation
* Apply the typology to annotate the datasets
* Analyze and visualize **differences in topic distributions** between the two subreddits
* Use gemini-2.5-flash AI through a script to annotate reddit posts 

This project explores the intersection of **qualitative research methods** and **LLM-assisted annotation workflows**.

---

## Data Sources

* **r/mcgill** subreddit posts
* **r/concordia** subreddit posts

The datasets were preprocessed and stored as TSV/CSV files prior to annotation.

---

## Methodology

1. **Data Collection**
   Reddit posts were imported from the McGill and Concordia subreddits.

2. **Open Coding**
   An initial round of open coding was conducted to identify recurring topics and patterns in the data.

3. **Typology Development**
   Based on open coding, a structured typology (codebook) was developed and refined to ensure consistent annotation across datasets.

4. **Annotation Pipeline**
   The typology was applied to annotate posts. An LLM-assisted approach was used to support and scale the annotation process.

5. **Analysis & Visualization**
   Topic frequencies were aggregated and compared between the two subreddits, and a plot was generated to visualize differences in discussion themes.

---

## Output

* Annotated Reddit datasets for McGill and Concordia
* A unified annotated dataset combining both subreddits
* Visualizations comparing topic distributions between r/mcgill and r/concordia

<img width="681" height="335" alt="Screenshot 2026-01-12 at 12 33 14 PM" src="https://github.com/user-attachments/assets/93e6103b-2f65-4fb6-8b7f-91584aa88ee8" />

---

## Technologies Used

* Python
* Pandas
* Large Language Models (LLMs) for assisted annotation
* Data visualization libraries (e.g., Matplotlib / Seaborn)


