### randomforestimputer - version 0.1
-------------------------------------
Imputation of missing data based on random forest's clustering property. This technique measures data points' proximity by counting the number of times they appear in the same leaf node. By building a decision tree in the forest one by one and updating the proximity matrix at the end of each tree, the algorithm estimates the value of missing data points based on a weighted average of existing samples appearing in the same leaf.

Simple imputation methods such as mean or mode fill in tends to provide biased results for large amount of missing data. RandomForestImputer can deal with various types and large volume of missing data. The process of building a random forest is repeated several times (usually <10) until convergence is reached. 

1. Note:    
    * Algorithm followed steps proposed in [link] https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#missing1
    * A practical definition of convergence has been derived - not part of the original algorithm
    * Refer to sklearn documentation for an in-depth explanation of base estimators' parameters
    * For this version, only one column (i.e. 1 feature) with missing values can be imputed at one time, assuming the rest of the features contain known data 
    
2. Python code example to run the imputer:
```python
import sys
import importlib
sys.path.append('/Users/nbngu/Documents/Python learning/Machine learning material/all_projects/customised_functions')

import pandas as pd
import numpy as np
from rfimputer import RandomForestImputer

from sklearn import tree

# Initial data cleaning and organisation
df_original = pd.read_csv("healthcare-dataset-stroke-data.csv")
df_original = df_original.drop(labels = ["id"],axis=1)
df_original.drop(df_original.loc[df_original["gender"]=="Other"].index,inplace=True,axis=0)
df_original.reset_index(drop=True,inplace=True)
df1 = df_original.copy()

# Initialise imputer
rffc = RandomForestImputer(base_estimator=tree.DecisionTreeClassifier(),class_weight="balanced",max_depth=None,n_estimators=2,
                        n_iter=2,miss_val_type="continuous")
                        
# Fit_transform imputer -> return imputed array (known and imputed unknown values)
rffc.fit_transform(df=df1,target_name="stroke",miss_col_name="bmi", miss_val=np.nan,simple_fill="median", 
            method="knn", n_neighbors=5, miss_array=None, ordinal_list=["gender"])

# To change other parameters of the base estimator
rffc.get_params(deep=True)
param = {"max_depth":10,"base_estimator__ccp_alpha":0.5}
rffc.set_params(**param)
rffc.get_params(deep=True)

```
     
     
