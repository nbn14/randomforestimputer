### randomforestimputer - version 0.1
-------------------------------------
Imputation of missing data based on random forest's clustering property. This technique measures data points' proximity by counting the number of times they appear in the same leaf node. By building a decision tree in the forest one by one and updating the proximity matrix at the end of each tree, the algorithm estimates the value of missing data points based on a weighted average of existing samples appearing in the same leaf.

Simple imputation methods such as mean or mode fill in tends to provide biased results for large amount of missing data. RandomForestImputer can deal with various types and large volume of missing data. The process of building a random forest is repeated several times (usually <10) until convergence is reached. 

1. Note:    
    * Algorithm followed steps proposed in [link] https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#missing1
    * A practical definition of convergence has been derived - not part of the original algorithm
    * Refer to sklearn documentation for an in-depth explanation of base estimators' parameters
     
     
