# Artifact Generator

## Overview

This repository is involved in ongoing research focused on developing models for cleansing pulsatile signals with artifact augmentation. Currently, a paper is in progress and is expected to be submitted to the archivx by June 2024. A link to the paper will be provided thereafter.

## Usage

Please note that this version is primarily for developmental use and may contain bugs or incomplete features. Users are advised to proceed with caution and are encouraged to contact the authors for more information or to discuss potential collaborations.

**Unauthorized Use Prohibited**: Unauthorized use, duplication, or distribution of this software and its associated documentation is strictly prohibited without prior written permission from the author(s). Please contact the author(s) to obtain permission before using this software in any manner not explicitly authorized.

## Contact Information

If you plan to use our simulator or model structure for your research or if you need specific details about the implementation, please contact the author(s) before proceeding. This will ensure you have the most up-to-date information and guidance. Contact details can be found below:

- **Email:** [lyjune0070@gmail.com, bluemk00@gmail.com]

- **Institution:** [Department of Cancer AI and Digital Health, Graduate School of Cancer Science & Policy, National Cancer Center, KOREA]

## Contribution

**J. Kim** initiated this project by establishing the concept, conducting literature research, and drafting codes for artifact generation. 
Subsequently, **K. Park** conducted a proof of concept, significantly enhancing the generation process and contributing to major improvements in its sophistication.

## Project Structure

This project is organized into several directories, each serving a specific purpose in the research and development process:

### lib/
Contains essential libraries and functions needed for the project:
- `artifact_augmentation.py`: Functions for augmenting artifacts during model training.
- `artifact_simulation.py`: Functions required by the Artifact Simulator.

### train/
Contains scripts for training different models:
- `ModelStructure_DI.py`: Basic structure for the DI model.
- `train_DI.py`: Training script for the DI model.
- `train_DI_D.py`: Training script for the DI-D model variant.
- `train_DI_A.py`: Training script for the DI-A model variant.
- `ModelStructure_DA.py`: Basic structure for the DA model.
- `train_DA.py`: Training script for the DA model.
- `train_DA_D.py`: Training script for the DA-D model variant.
- `train_DA_A.py`: Training script for the DA-A model variant.

### data/
Contains sample files for the simulator:
- `ABP_60s_sample.npy`: Sample file for artifact simulation.

### simulator/
Houses the artifact simulator:
- `Example_for_Artifact_Simulation.ipynb`: Jupyter notebook demonstrating artifact simulation.
