## cMCA: Contrastive Multiple Correspondence Analyis

New
-----
* R coding example is added (sample.R) based on multiple received requests (2024-06-16)

About
-----
* cMCA is from: Fujiwara and Liu, "Contrastive multiple correspondence analysis (cMCA): using contrastive learning to identify latent subgroups in political parties." PLOS ONE, vol. 18, no.7, e0287180 (20 pages), 2023. [PLOS ONE](https://doi.org/10.1371/journal.pone.0287180), [arXiv](https://arxiv.org/abs/2007.04540)
  * cMCA finds principal components along which a target categorical dataset has more variance when compared to a background categorical dataset.
  * This implementation also integrates automatic selection of a contrast parameter (a hyperparameter of cMCA), which is originally developed for contrastive principal component analysis (cPCA) in: Fujiwara et al., "Network Comparison with Interpretable Contrastive Network Representation Learning", Journal of Data Science, Statistics, and Visualisation, 2022. [JDSSV](https://doi.org/10.52933/jdssv.v2i5), [arXiv](https://arxiv.org/abs/2005.12419)
  * Other use cases (using machine maintenance data) can be found in: Zhan et al., "A Visual Analytics Approach for the Diagnosis of Heterogeneous and Multidimensional Machine Maintenance Data." Proc. PacificVis, 2021. [IEEE Xplore](https://doi.org/10.1109/PacificVis52677.2021.00033)
  

******

Requirements
-----
* Python3
* R users can still utilize cMCA library by using reticulate package. Examples can be found in sample.R
* Note: Tested on macOS Sequoia, Ubuntu 22.0.4 LTS, and Windows 10.
******

Setup
-----
* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

******

Usage
-----
* Import installed modules from python (e.g., `from cmca import CMCA`). See sample.py for examples.
* For detailed documentations, please see doc/index.html or directly see comments in cmca/cmca.py.

* To use cMCA from R, you can utilize reticulate package. See sample.R
  - Data type conversions are important for several case. For example:
    - When identifying optional parameters of cMCA with R integer values, convert like `n_components=as.integer(2)` (otherwise, 2 is considered as an index data type)
    - cMCA returns outputs as numpy objects in general. So, to convert in R native objects, use `py_to_r`. Example code to convert numpy array to r matrix object can be found in sample.R (`nparr_to_rmat` function).
  - sample.R covers only a part of sample.py. Other remaining examples can be implemented if requested.

******

## How to Cite
Please, cite:    
* Fujiwara and Liu, "Contrastive multiple correspondence analysis (cMCA): using contrastive learning to identify latent subgroups in political parties." PLOS ONE, vol. 18, no.7, e0287180 (20 pages), 2023.