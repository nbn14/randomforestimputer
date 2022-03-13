# randomforestimputer
Imputation of missing data based on random forest's clustering property
Version 0.1
Random Forest algorithm to fill in missing data
    Algorithm followed steps proposed in https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#missing1
    Note: 
        - A practical definition of convergence has been derived - not part of the original algorithm
        - Refer to sklearn documentation for in-depth explanation of base parameters. A selection of base estimator's parameters - decision tree is
        included at class initialisation for convenience. Setting base estimator's other parameters can be done using method set_params(**param)
        
Packages requirements:
import pandas as pd
import numpy as np
import warnings

from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as im_pipeline  
from sklearn.base import BaseEstimator,clone
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# In addition, 2 customised files required
from feature_transformation import *
from data_processing import *
