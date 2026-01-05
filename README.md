<div align="center">

  <a name="readme-top"></a>
  # Optimizing Stock Trading Strategy with K-Means Clustering

  [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
  ![Status](https://img.shields.io/badge/Status-Completed-success)
  [![Technology](https://img.shields.io/badge/Technology-Python%20%7C%20Scikit--Learn%20%7C%20Pandas-blueviolet)](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING)
  [![Developed by Amey Thakur, Hasan Rizvi & Mega Satish](https://img.shields.io/badge/Developed%20by-Amey%20Thakur%2C%20Hasan%20Rizvi%20%26%20Mega%20Satish-blue)](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING)

  An analytical project utilizing unsupervised machine learning to cluster stocks based on their volatility and returns, identifying latent market patterns and optimizing diversified trading strategies.

  **[Source Code](Source%20Code/)** &nbsp;&middot;&nbsp; **[Mini-Project](Mini-Project/)** &nbsp;&middot;&nbsp; **[Curriculum Repository](https://github.com/Amey-Thakur/BIG-DATA-ANALYTICS-AND-COMPUTATIONAL-LAB-I)** &nbsp;&middot;&nbsp; **[Project Demo](https://youtu.be/Q82a93hjxJE)**

  [![Optimizing Stock Trading Strategy Demo](https://img.youtube.com/vi/Q82a93hjxJE/0.jpg)](https://youtu.be/Q82a93hjxJE)

</div>

---

<div align="center">

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

</div>

---

<!-- AUTHORS -->
<div align="center">

  ## Authors

  **Terna Engineering College | Computer Engineering | Batch of 2022**

  <table>
  <tr>
  <td align="center">
  <a href="https://github.com/Amey-Thakur">
  <img src="https://github.com/Amey-Thakur.png" width="150px;" alt="Amey Thakur"/><br />
  <sub><b>Amey Thakur</b></sub>
  </a>
  </td>
  <td align="center">
  <a href="https://github.com/rizvihasan">
  <img src="https://github.com/rizvihasan.png" width="150px;" alt="Hasan Rizvi"/><br />
  <sub><b>Hasan Rizvi</b></sub>
  </a>
  </td>
  <td align="center">
  <a href="https://github.com/msatmod">
  <img src="Mega/Mega.png" width="150px;" alt="Mega Satish"/><br />
  <sub><b>Mega Satish</b></sub>
  </a>
  </td>
  </tr>
  </table>

</div>

---

<!-- OVERVIEW -->
## Overview

This project investigates the application of **K-Means Clustering** on financial market data. By categorizing stocks into distinct clusters based on their historical price movements, the system provides a data-driven approach to understanding market dynamics and constructing balanced investment portfolios.

Developed as a mini-project for the **Big Data Analytics & Computational Lab - I** curriculum, this implementation showcases the full data science pipeline: from data acquisition via Yahoo Finance to feature engineering (volatility/returns) and unsupervised model validation.

### Resources

| # | Resource | Description | Link |
|---|---|---|---|
| 1 | **Project Model** | Complete Jupyter Notebook implementation | [View](Source%20Code/OPTIMIZING%20STOCK%20TRADING%20STRATEGY%20WITH%20K-MEANS%20CLUSTERING.ipynb) |
| 2 | **Technical Report** | Comprehensive project documentation | [View](Mini-Project/BDA_MINI-PROJECT_REPORT_BE-COMPS_B-50,51,58.pdf) |
| 3 | **Technical Presentation** | Visual overview of methodology and results | [View](Mini-Project/BDA_MINI-PROJECT_PPT_BE-COMPS_B-50,51,58.pdf) |
| 4 | **Project Demo (YouTube)** | Real-time demonstration of the analysis | [View](https://youtu.be/Q82a93hjxJE) |

---

<!-- STRUCTURE -->
## Project Structure

```bash
OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING/
│
├── docs/                                          # Formal Documentation
│   └── SPECIFICATION.md                           # Technical Architecture & Spec
│
├── Mega/                                          # Archival Attribution Assets
│   ├── Filly.jpg                                  # Project-related Content Asset
│   ├── Mega.png                                   # Author Profile Image (Mega Satish)
│   ├── Mega_Chair.png                             # Author Profile Context
│   ├── Mega_Dining.jpg                            # Author Personal Context
│   ├── Mega_Professional.jpg                      # Author Professional Portrait
│   └── Mega_and_Hetvi.png                         # Collaborative Identity Asset
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
├── .gitignore                                     # Asset Exclusion Manifest
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

**Connect**: [GitHub](https://github.com/Amey-Thakur) · [LinkedIn](https://www.linkedin.com/in/amey-thakur)

### Acknowledgments

Grateful acknowledgment to **[Hasan Rizvi](https://github.com/rizvihasan)** and **[Mega Satish](https://github.com/msatmod)** for their pivotal roles and collaborative excellence during the development of this project. Their intellectual contributions, technical insights, and dedicated commitment to software quality were fundamental in achieving the system's analytical and functional objectives. This technical record serves as a testament to their scholarly partnership and significant impact on the final implementation.

Special thanks to the faculty members of the Department of Computer Engineering at Terna Engineering College for their guidance during the course of this project. Gratitude is also extended to the mentors and peers who supported this learning endeavor.

---

<!-- FOOTER -->
<div align="center">

  [↑ Back to Top](#readme-top)

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

  <br>

  🔬 **[Big Data Analytics Laboratory](https://github.com/Amey-Thakur/BIG-DATA-ANALYTICS-AND-COMPUTATIONAL-LAB-I)** &nbsp;·&nbsp; 📊 **[Optimizing Stock Trading Strategy](https://github.com/Amey-Thakur/OPTIMIZING-STOCK-TRADING-STRATEGY-WITH-K-MEANS-CLUSTERING)**

  ---

  ### Presented as part of the 7th Semester Mini-Project @ Terna Engineering College

  ### 🎓 [Computer Engineering Repository](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

  **Computer Engineering (B.E.) - University of Mumbai**

  *Semester-wise curriculum, laboratories, projects, and academic notes.*

</div>
