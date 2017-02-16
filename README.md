# BPR_MCA 
--------
Python library for Bayesian Personalized Ranking Matrix Factorization extended with Multiple Content Alignments
--------

version 1.0, February 16, 2016

--------
This package is written by:

Ladislav Peska,

Dept. of Software Engineering, Charles University in Prague, Czech Republic

Email: peska@ksi.mff.cuni.cz

Furthre information can be found on:
http://www.ksi.mff.cuni.cz/~peska/BPR_MCA

-------
Some functions used in this package are based on the PyDTI package by Yong Liu,
https://github.com/stephenliu0423/PyDTI

This package also uses Rank metrics implementation by Brandyn White (included as rank_metrics.py)

BPR_MCA implementation is based on the BPR implementation by Mark Levy, 
https://github.com/gamboviol/bpr

--------
BPR_MCA works on Python 2.7 (tested on Intel Python 2.7.12).
--------
BPR_MCA requires NumPy, scikit-learn and SciPy to run.
To get the results of different methods, please run bpr_mca.py. The __main__ part of the bpr_mca.py runs monte-carlo CV on extended MovieLens1M dataset with internal hyperparameter tuning. The dataset is available on http://www.ksi.mff.cuni.cz/~peska/BPR_MCA. However, BPR_MCA method can input any binary user preference matrix and arbitrary many user and object-based similarity matrices - see different input variants on lines 500-570.
