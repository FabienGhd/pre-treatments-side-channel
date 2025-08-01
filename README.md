## How do Pre-treatments Influence Side-Channel Attacks?
ABSTRACT: The effectiveness of side-channel attacks has been a critical topic in cybersecurity since their emergence in the late 1990s. SCAs exploit physical measurements like power consumption to extract cryptographic keys, and while many studies have focused on direct attack methodologies, the impact of preprocessing techniques on the efficiency of SCAs remains underexplored. This research investigates the influence of various preprocessing methods on the success rates of Correlation Power Analysis and Linear Regression Analysis attacks. By analyzing simulated power traces from both software and hardware implementations of the AES algorithm, we evaluate preprocessing methods including raw, squared, absolute value, centered, and standardized traces. Our findings reveal that our current preprocessing methods
don’t significantly improve attack effectiveness. As the number of traces increases, the impact of preprocessing diminishes even more, with all methods converging towards accurate key identification. This study highlights the potential of preprocessing to refine SCA strategies and suggests future integration of deep learning techniques to further enhance preprocessing and attack methodologies.

<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<h3 align="center">Influence of Pre-treatments on Side-channel Attacks
</h3>

  <p align="center">
    Research Project - M1 Cybersecurity
    <br />
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This research project was carried out in the second semester of the M1 Cybersecurity program at the University of Rennes.

This research project explores the impact of data pre-treatments on the success and effectiveness of side-channel attacks, particularly focusing on cryptographic systems. Side-channel attacks are sophisticated security breaches where an attacker gains insights into the cryptographic operations of a device by analyzing unintended physical outputs, such as power consumption. Our project investigates how careful preprocessing of this output can reveal sensitive information about cryptographic keys.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

The project requires the following packages:

* numpy
  ```sh
  pip install numpy
  ```
* matplotlib - for results visualization
  ```sh
  python -m pip install -U pip
  python -m pip install -U matplotlib
  ```
* h5py
  ```sh
  pip install h5py
  ```
* tqdm
  ```sh
  pip install tqdm
  ```
* numba
  ```sh
  pip install numba
  ```
* lascar
  ```sh
  pip3 install "git+https://github.com/Ledger-Donjon/lascar.git"
  ```
* scipy
  ```sh
  python -m pip install scipy
  ```
For the extensive list with the versions used, please refer to the "Requirements.txt" file. This file contains a list of all packages and libraries needed to work on the project.
  ```sh
  pip install -r Requirements.txt
  ```
### Installation

1. Clone the repo
   ```sh
   git clone https://gitlab.istic.univ-rennes1.fr/faguihard/side-channel.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- STRUCTURE -->
## Repository Structure
The repository contains 6 folders:

* images:\
It contains visual evidence of the progress and results of our attacks. There are graphs that track the progression of rank scores over time for different analysis methods. There are graphs that illustrate the power traces collected from cryptographic devices under different key conditions. Each filename indicates the number of traces and key index used during the analysis
* paper:\
It contains the research paper studied to start the project.
* presentation:\
It includes the slides from our official project presentation to professors and students.
* report:\
It includes the technical report of the project, explaining the theory and results in details. The report is written in LaTeX. (see main.pdf)
* src:\
It contains the Python scripts for analysis.
* traces:\
It contains the physical outputs used in analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact
<div align="center">
Students:
</div>

Fabien Guihard - fabien.guihard@etudiant.univ-rennes.fr\
<div align="center">

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Behind the Scene of Side Channel Attacks](https://eprint.iacr.org/2013/794.pdf)
* [AES Hardware Dataset](https://github.com/AISyLab/AES_HD_Ext)
* [Lascar's Documentation](https://github.com/Ledger-Donjon/lascar)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[Python]: https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white
[Python-url]: https://www.python.org/

