## cMCA: Contrastive Multiple Correspondence Analyis

About
-----
* cMCA is from: Fujiwara and Liu, "Contrastive multiple correspondence analysis (cMCA): using contrastive learning to identify latent subgroups in political parties." PLOS ONE, forthcoming. [arXiv](https://arxiv.org/abs/2007.04540)
  * cMCA finds principal components along which a target categorical dataset has more variance when compared to a background categorical dataset.
  * This implementation also integrates automatic selection of a contrast parameter (a hyperparameter of cMCA), which is originally developed for contrastive principal component analysis (cPCA) in: Fujiwara et al., "Network Comparison with Interpretable Contrastive Network Representation Learning", Journal of Data Science, Statistics, and Visualisation, 2022. [arXiv](https://arxiv.org/abs/2005.12419)
  * Other use cases (using machine maintenance data) can be found in: Zhan et al., "A Visual Analytics Approach for the Diagnosis of Heterogeneous and Multidimensional Machine Maintenance Data." Proc. PacificVis, 2021. 
  

******

Requirements
-----
* Python3
* Note: Tested on macOS Ventura, Ubuntu 22.0.4 LTS, and Windows 10.
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

******

## How to Cite
Please, cite:    
* Fujiwara and Liu, "Contrastive multiple correspondence analysis (cMCA): using contrastive learning to identify latent subgroups in political parties." PLOS ONE, forthcoming.
