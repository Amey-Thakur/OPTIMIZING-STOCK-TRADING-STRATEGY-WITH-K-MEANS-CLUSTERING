<div align="center">

  # Optimizing Stock Trading Strategy with K-Means Clustering

  [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
  ![Status](https://img.shields.io/badge/Status-Completed-2EA043)
  [![Technology](https://img.shields.io/badge/Technology-Python%20%7C%20Scikit--Learn%20%7C%20Pandas-8250DF)](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING)
  [![Developed by Amey Thakur, Hasan Rizvi & Mega Satish](https://img.shields.io/badge/Developed%20by-Amey%20Thakur%2C%20Hasan%20Rizvi%20%26%20Mega%20Satish-0969DA)](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING)

  An analytical project utilizing unsupervised machine learning to cluster stocks based on their volatility and returns, identifying latent market patterns and optimizing diversified trading strategies.

  **[Source Code](Source%20Code/)** &nbsp;·&nbsp; **[Technical Specification](docs/SPECIFICATION.md)** &nbsp;·&nbsp; **[Video Demo](https://youtu.be/Q82a93hjxJE)** &nbsp;·&nbsp; **[Live Demo](https://Amey-Thakur.github.io/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING/)**

  [![Optimizing Stock Trading Strategy Demo](https://img.youtube.com/vi/Q82a93hjxJE/0.jpg)](https://youtu.be/Q82a93hjxJE)

</div>

---

<div align="center">

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

</div>

---

<!-- AUTHORS -->
<div align="center">

  ## Authors

  **Terna Engineering College | Computer Engineering | Batch of 2022**

| <a href="https://github.com/Amey-Thakur"><img src="https://github.com/Amey-Thakur.png" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39.svg)](https://orcid.org/0000-0001-5644-1575) | <a href="https://github.com/rizvihasan"><img src="https://github.com/rizvihasan.png" width="150" height="150" alt="Hasan Rizvi"></a><br>[**Hasan Rizvi**](https://github.com/rizvihasan)<br><br>[![GitHub](https://img.shields.io/badge/GitHub-rizvihasan-181717.svg?style=flat&logo=github&logoColor=white)](https://github.com/rizvihasan) | <a href="https://github.com/msatmod"><img src="https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING/blob/main/Mega/Mega.png?raw=true" width="150" height="150" alt="Mega Satish"></a><br>[**Mega Satish**](https://github.com/msatmod)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1844--9557-A6CE39.svg)](https://orcid.org/0000-0002-1844-9557) |
| :---: | :---: | :---: |

</div>

> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> *Special thanks to **[Hasan Rizvi](https://github.com/rizvihasan)** and **[Mega Satish](https://github.com/msatmod)** for their meaningful contributions, guidance, and support that helped shape this work.*

---

<!-- OVERVIEW -->
## Overview

This project investigates the application of **K-Means Clustering** on financial market data. By categorizing stocks into distinct clusters based on their historical price movements, the system provides a data-driven approach to understanding market dynamics and constructing balanced investment portfolios.

Developed as a mini-project for the **Big Data Analytics & Computational Lab - I** curriculum, this implementation showcases the full data science pipeline: from data acquisition via Yahoo Finance to feature engineering (volatility/returns) and unsupervised model validation.

### Resources

| # | Resource | Description |
|---|---|---|
| 1 | [**Project Model**](Source%20Code/OPTIMIZING%20STOCK%20TRADING%20STRATEGY%20WITH%20K-MEANS%20CLUSTERING.ipynb) | Complete Jupyter Notebook implementation |
| 2 | [**Technical Specification**](docs/SPECIFICATION.md) | Technical Architecture & Specification |
| 3 | [**Technical Report**](Mini-Project/BDA_MINI-PROJECT_REPORT_BE-COMPS_B-50,51,58.pdf) | Comprehensive project documentation |
| 4 | [**Technical Presentation**](Mini-Project/BDA_MINI-PROJECT_PPT_BE-COMPS_B-50,51,58.pdf) | Visual overview of methodology and results |
| 5 | [**Project Demo**](https://youtu.be/Q82a93hjxJE) | Real-time demonstration of the analysis |

> [!TIP]
> **Cluster Validation Best Practices**
>
> Use the **Elbow Method** to identify the optimal number of clusters by plotting Within-Cluster Sum of Squares (WCSS). Complement this with the **Silhouette Score** to validate cluster cohesion and separation for robust market segmentation.

---

<!-- FEATURES -->
## Features

| Feature | Description |
|---------|-------------|
| **K-Means Clustering** | Unsupervised segmentation of stocks based on volatility and returns metrics. |
| **Data Acquisition** | Automated historical data retrieval via Yahoo Finance API (yfinance). |
| **Feature Engineering** | Calculation of annualized volatility and returns for each stock. |
| **Cluster Validation** | Elbow Method and Silhouette Score for optimal cluster determination. |
| **Visualization** | Interactive scatter plots and cluster centroid analysis. |
| **Portfolio Optimization** | Data-driven insights for diversified investment strategies. |

### Tech Stack
- **Language**: Python 3.8+
- **ML Framework**: Scikit-Learn (K-Means, Silhouette Analysis)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Data Source**: yfinance (Yahoo Finance API)

---

<!-- STRUCTURE -->
## Project Structure

```python
OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING/
│
├── docs/                                          # Formal Documentation
│   └── SPECIFICATION.md                           # Technical Architecture & Specification
│
├── Mega/                                          # Archival Attribution Assets
│   ├── Filly.jpg                                  # Companion (Filly)
│   ├── Mega.png                                   # Author Profile Image (Mega Satish)

│
├── Mini-Project/                                  # Research & Academic Assets
│   ├── BDA_MINI-PROJECT_PPT...pdf                 # Project Presentation (PDF)
│   ├── BDA_MINI-PROJECT_PPT...pptx                # Project Presentation (PPTX)
│   ├── BDA_MINI-PROJECT_REPORT...docx             # Technical Project Report (DOCX)
│   └── BDA_MINI-PROJECT_REPORT...pdf              # Technical Project Report (PDF)
│
├── Source Code/                                   # Model Implementation
│   ├── OPTIMIZING STOCK TRADING STRATEGY...ipynb  # Core K-Means Analysis Notebook
│   └── Stock_Market_Clustering.py                 # Production-ready Python Script
│
├── .gitattributes                                 # Global Git LFS & Config
│   └── .gitignore                                 # Asset Exclusion Manifest
├── requirements.txt                               # Dependency Manifest
├── CITATION.cff                                   # Scholarly Citation Metadata
├── codemeta.json                                  # Software Metadata Manifest
├── LICENSE                                        # MIT License Terms
├── README.md                                      # Comprehensive Archival Entrance
└── SECURITY.md                                    # Vulnerability Exposure Policy
```

---

<!-- QUICK START -->
## Quick Start

### 1. Prerequisites
Ensure your environment meets the minimum specifications:
- **Python**: Version **3.8** or higher.
- **Hardware**: 4GB Minimum RAM (8GB recommended for large datasets).
- **Environment**: Virtual environment (venv) is highly recommended.

> [!WARNING]
> **Technical Dependencies & Data Variability**
>
> This system is built using **Python 3.8+** and **Scikit-Learn**. Stock market data is inherently volatile; results may vary based on the date range and ticker symbols selected. For stable execution and reproducible analysis, it is recommended to run this in an isolated virtual environment.

### 2. Setup & Deployment
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING.git
    cd OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Launch Application
1.  **Run the Python Script**:
    ```bash
    cd "Source Code"
    python Stock_Market_Clustering.py
    ```
2.  **Explore the Notebook**:
    -   Open `OPTIMIZING STOCK TRADING STRATEGY WITH K-MEANS CLUSTERING.ipynb` in Jupyter Notebook for interactive analysis.

> [!TIP]
> **Optimizing Stock Trading Strategy with K-Means Clustering**
>
> Experience a high-fidelity interactive simulation grouping major S&P 500 companies based on volatility and return patterns to identify optimal trading opportunities through unsupervised machine learning and advanced market segmentation.
>
> [**Launch Interactive Notebook**](https://amey-thakur.github.io/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING/)
>
> Recent enhancements also include a Reinforcement Learning (RL) gateway for advanced strategy optimization.
>
> [**Launch Stock Trading RL Web App**](https://huggingface.co/spaces/ameythakur/Stock-Trading-RL)

---

<!-- =========================================================================================
                                     USAGE SECTION
     ========================================================================================= -->
## Usage Guidelines

This repository is openly shared to support learning and knowledge exchange across the academic community.

**For Students**  
Use this project as a reference for understanding clustering algorithms, financial data preprocessing, and the application of Big Data Analytics in stock market optimization.

**For Educators**  
This project may serve as a practical example or supplementary teaching resource for Big Data Analytics (`CSDLO7032`) and Computational Laboratory–I (`CSL704`) modules. Attribution is appreciated when utilizing content.

**For Researchers**  
The implementation provides a foundation for exploring more advanced clustering techniques (e.g., DBSCAN, Hierarchical Clustering) and sentiment-integrated market analysis.

---

<!-- LICENSE -->
## License

This repository and all linked academic content are made available under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms.

> [!NOTE]
> **Summary**: You are free to share and adapt this content for any purpose, even commercially, as long as you provide appropriate attribution to the original author.

Copyright © 2022 Amey Thakur, Hasan Rizvi, Mega Satish

---

<!-- ABOUT -->
## About This Repository

**Created & Maintained by**: [Amey Thakur](https://github.com/Amey-Thakur), [Hasan Rizvi](https://github.com/rizvihasan) & [Mega Satish](https://github.com/msatmod)  
**Academic Journey**: Bachelor of Engineering in Computer Engineering (2018-2022)  
**Institution**: [Terna Engineering College](https://ternaengg.ac.in/), Navi Mumbai  
**University**: [University of Mumbai](https://mu.ac.in/)

This project features the **Optimizing Stock Trading Strategy with K-Means Clustering**, an analytical utility developed as a **7th Semester Mini-Project**. It explores the application of unsupervised machine learning for financial market analysis and portfolio optimization.

**Connect**: [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/amey-thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

### Acknowledgments

Grateful acknowledgment to [**Hasan Rizvi**](https://github.com/rizvihasan) and [**Mega Satish**](https://github.com/msatmod) for their exceptional collaboration and scholarly partnership during the development of this project. Their technical expertise, constant support, and dedication to software quality were instrumental in achieving the project's analytical objectives. Learning alongside them was a transformative experience; their thoughtful approach to problem-solving and encouragement turned complex challenges into meaningful learning moments. This work reflects the growth and insights gained from our side-by-side academic journey. Thank you, Hasan and Mega, for everything you shared and taught along the way.

Grateful acknowledgment to the faculty members of the Department of Computer Engineering at Terna Engineering College for their guidance and instruction in Big Data Analytics. Their expertise in data science and machine learning helped shape the technical foundation of this project.

Special thanks to the mentors and peers whose encouragement, discussions, and support contributed meaningfully to this learning experience.

---

<!-- FOOTER -->
<div align="center">

  [↑ Back to Top](#optimizing-stock-trading-strategy-with-k-means-clustering)

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

  <br>

  🔬 **[Big Data Analytics Laboratory](https://github.com/Amey-Thakur/BIG-DATA-ANALYTICS-AND-COMPUTATIONAL-LAB-I)** &nbsp;·&nbsp; 📊 **[Optimizing Stock Trading Strategy](https://Amey-Thakur.github.io/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING/)**

  ---

  #### Presented as part of the 7th Semester Mini-Project @ Terna Engineering College
  
  ---
  
  ### 🎓 [Computer Engineering Repository](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

  **Computer Engineering (B.E.) - University of Mumbai**

  *Semester-wise curriculum, laboratories, projects, and academic notes.*

</div>

