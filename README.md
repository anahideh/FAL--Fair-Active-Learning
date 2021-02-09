# Fair Active Learning
This repository provides Python implementation for our "Fair Active Learning" paper. https://arxiv.org/abs/2001.01796

## Experiments:
tes_FAL.py trains a classifier given a sampling strategy:
- FAL_sklearn: our propossed fair active learning approach using expected fairness .
- FAL_COVXY: our proposed efficient fair active learning by covariance approach. 
- FAL_sklearn_Nested: our propossed Nested approach .
- FAL_COVXY_Nested: our proposed Nested approach with FBC. 
- FAL_sklearn_Nested_Append: our propossed Nested-Append approach .
- FAL_COVXY_Nested_Append: our proposed Nested-Append approach with FBC. 
- AL: standard active learning with uncertainty sampling.
- RL: random sampling.
