## cMCA: Contrastive Multiple Correspondence Analyis

About
-----
* cMCA is from: Fujiwara and Liu, "Contrastive Multiple Correspondence Analysis (cMCA): Applying the Contrastive Learning Method to Identify Political Subgroups", arXiv:2007.04540, 2020.
  * cMCA finds principal components along which a target categorical dataset has more variance when compared to a background categorical dataset.
  * This implementation also integrates automatic selection of a contrast parameter (a hyperparameter of cMCA), which is originally developed for contrastive principal component analysis (cPCA) in: Fujiwara et al., "Interpretable Contrastive Learning for Networks", arXiv:2005.12419, 2020.

******

Requirements
-----
* Python3
* Note: Tested on macOS BigSur, Ubuntu 20.0.4 LTS, and Windows 10.
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
* cMCA is from: Fujiwara and Liu, "Contrastive Multiple Correspondence Analysis (cMCA): Applying the Contrastive Learning Method to Identify Political Subgroups", arXiv:2007.04540, 2020.
