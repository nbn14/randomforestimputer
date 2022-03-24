import pandas as pd
import numpy as np
from feature_transformation import *
from data_processing import *
import warnings

from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as im_pipeline  
from sklearn.base import BaseEstimator,clone
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


class RandomForestImputer(BaseEstimator):
    """
    Random Forest algorithm to fill in missing data
    Algorithm followed steps proposed in https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#missing1
    Note: 
        - A practical definition of convergence has been derived - not part of the original algorithm
        - Refer to sklearn documentation for in-depth explanation of base parameters. A selection of base estimator's parameters - decision tree is
        included at class initialisation for convenience. Setting base estimator's other parameters can be done using method set_params(**param)

    Parameters
    ----------
    base_estimator: sklearn tree estimator, default = tree.DecisionTreeClassifier or any tree type algorithms with apply function
    classification: bool, default = True
        Classifier flag. Indicate whether the target variable is continuous or categorical. False means regression
    miss_val_type: {"categorical_nominal", "categorical_ordinal", "continuous}, default="categorical-nominal"
        dtype of the feature containing missing data to be filled
    max_depth: int or None, default = None
        Set maximum depth of individual tree
    max_features: int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split 
    criterion: {"gini", "entropy}, default="gini"
        The function to measure the quality of a split
    min_samples_split: int or float, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leafint or float, default=1
        The minimum number of samples required to be at a leaf node.
    resampling: {"bootstrap", "random_over", "random_under"}, default="random_over
        Resampling method used to build trees in the forest
    n_estimators: int, default=10
        Number of trees in the forest
    n_iter: int, default=3
        Number of times (i.e. forests) the algorithm will be repeated. Imputed results from previous iteration are used as initial guess for the next.

    Attributes:
    -----------
    miss_val_: dtype
        Missing value to be filled
    miss_col_name_: dtype
        Name of column with missing data to be filled
    miss_col_: ndarray of shape (# of total data points,)
        Original array with missing data to be filled
    target_name_: dtype
        Name of target column recorded in original input dataframe
    y_: ndarray of shape (# of total data points,)
        Target variable. Format compatile with base_estimator  
    df_miss_target_: pd DataFrame with 2 columns (feature with missing values and target variable)
    x_rest_: ndarray of shape(total # of data points,# of columns after encoded)
        Encoded/transformed array of other input features (excluding the feature with missing values to be filled)
    index_known_: ndarray of shape(# of known data points,)
        List of indices of known data points in the original dataframe
    index_missing_: ndarray of shape(# of unknown data points,)
        List of indices of unknown data points in the original dataframe
    mod_proximity_mat_: dict of float arrays of shape (n_iter,)
        Contain the modified proximity matrix for each random forest run before normalisation
    weighted_freq_mat_: dict of float arrays of shape (n_iter,)
        Contain transformed proximity matrix after weighing and normalisation are done for each random forest run
    updated_miss_col_: dict of float arrays of shape (n_iter,)
        Contain updated guess values for the entire miss_col_ at every iteration

    Methods:
    -------
    fit_transform(df,target_name,miss_col_name, miss_val,simple_fill,method, n_neighbors, miss_array, ordinal_list): return an array with known and imputed missing data
    """
    
    def __init__(self,base_estimator=tree.DecisionTreeClassifier(),n_iter=6,
                n_estimators=2,max_depth=None, criterion='gini',
                max_features=5,min_samples_split=2,min_samples_leaf=1,
                classification=True,resampling ="random_over", miss_val_type="categorical_nominal",
                class_weight = "balanced"):

        # Base estimator attributes
        self.classification = classification
        self.miss_val_type = miss_val_type
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.criterion = criterion
        
        # Forest attributes
        self.resampling = resampling
        self.n_estimators = n_estimators
        
        # Multiple forest runs
        self.n_iter = n_iter

    
        
    def generate_estimators(self,random_state_rf=1):
        """Generate a random forest using the defined attributes in __init__"""
        forest_est = []
        for i in range(self.n_estimators):
            # Pass on parameters to define each base estimator
            estimator = clone(self.base_estimator)   
            estimator.max_depth = self.max_depth
            estimator.max_features = self.max_features
            estimator.min_samples_split = self.min_samples_split
            estimator.min_samples_leaf = self.min_samples_leaf
            estimator.class_weight = self.class_weight
            estimator.criterion = self.criterion
            estimator.random_state = i*123 + random_state_rf*533
            
            # Create a pipeline for each estimator which includes resampling method of choice
            random_state = i*577 + random_state_rf*1013
            if self.resampling =="random_over":
                pipe = im_pipeline([("res",RandomOverSampler(random_state=random_state,sampling_strategy="auto")),
                                    ("est",estimator)])
            elif self.resampling == "random_under":
                pipe = im_pipeline([("res",RandomUnderSampler(random_state=random_state,sampling_strategy="auto")),
                                    ("est",estimator)])
            elif self.resampling == "bootstrap":
                pipe = im_pipeline([("res",RandomOverSampler(random_state=random_state,sampling_strategy="all")),
                                    ("est",estimator)])
              
            forest_est.append((f"estimator_{i}", pipe))

        return forest_est
    


    def define_problem(self,df,target_name,miss_col_name,miss_val=np.nan,ordinal_list=[]):
        """Generate attributes describing the problem to be used in other functions of the class
        Parameters:
        -----------
        df: pd DataFrame of input and target variables
        target_name: name of target variable as recorded in df
        miss_col_name: name of feature with missing values to be filled, same as recorded in df
        miss_val = dtype, default=np.nan
            Value denoted as missing, same as recorded in df[miss_col_name]
        ordinal_list: list, default=[]
            List of ordinal input features, same as recorded in df
        Returns:
        --------
        self
        """
        # Define attributes for the missing column
        self.miss_val_ = miss_val
        self.miss_col_name_ = miss_col_name
        self.miss_col_ = df[miss_col_name].values
        
        # Target: Encode target into appropriate format -> return a ndarray (y_row,)
        if self.classification:
            self.y_ = LabelEncoder().fit_transform(df[target_name]) # Not suitable for estimators requiring sparse matrix as input
        else:
            self.y_= df[target_name].values
        self.target_name_ = target_name
        
        self.df_miss_target_ = df[[target_name,miss_col_name]]
        
        # Other columns: Separate other inputs from missing columns and target variable
        df_restX = df.loc[:,~(df.columns.isin([target_name,miss_col_name]))]
        # Transform other inputs into appropriate encoding format -> x_rest is an array
        self.x_rest_ = transform_cat_column(df_restX,ordinal_list=ordinal_list)
        
        # Generate original indices of missing and known data points 
        if self.miss_val_ != self.miss_val_:   # This is true only when the value is na
            self.index_known_ = df[~(df[miss_col_name].isna())].index
            self.index_missing_ = df[df[miss_col_name].isna()].index
        else:
            self.index_known_ = df[df[miss_col_name]!=self.miss_val_].index
            self.index_missing_ = df[df[miss_col_name]==self.miss_val_].index
    
        return self
    

    
    def generate_modified_proximity(self):
        """
        Return: a DataFrame of [n*(m+1)]
        n: # of known data points
        m: # of missing data points + 
            1 column is used to store classes of known values in miss_col_
        """
        # Initiate an empty dataframe for modified proximity matrix to track only 
        # dependence of unknown samples on known ones
        mod_proximity_mat = pd.DataFrame(index=self.index_known_,columns=self.index_missing_) 
        # Assign zero values to all columns
        for col in mod_proximity_mat.columns:
            mod_proximity_mat[col].values[:] = 0 
        # Add actual values of known_val to indices of known data points    
        mod_proximity_mat["known_val"] = self.miss_col_[self.index_known_]
        
        return mod_proximity_mat


    
    def initialise_miss_guess(self,simple_fill="most_frequent", method="simple", n_neighbors=1, miss_array=None):
        """Initialise the first guess of the of the missing values.
        Return ndarray with the same dimension as original missing array
        
        Parameters:
        -----------
        method: {"simple", "simple_stratified", "knn", "manual"}, default="simple"
            Imputation method for initial rough guess of missing values
            - "simple": Use SimpleImputer(). "constant" option is omitted in this version. Require simple_fill value.
            - "simple_stratified": Group data by target variable before calculating the initial guess values. Require simple_fill value
            Only available for categorical target variables (i.e. classification problems).
            - "knn": Use KNNImputer(). Require n_neighbors value
            - "manual": manual input an array with same indices and size as original missing array. Require miss_array.
        simple_fill: {"most_frequent", "mean", "median"}, default="most frequent"
            - simple_fill option needs to be compatible with miss_val_type_, e.g., "mean" can only be used for miss_val_type_=continuous 
        n_neighbors: int, default=1 
            Used in KNNImputer() option.
        miss_array: 1D array, default=None
            Manually input array to be used as initial guess. Must have the same dimension as original miss_col_
        """

        if method == "simple":
            simple = SimpleImputer(missing_values=self.miss_val_, strategy=simple_fill) # For continuous variable simple_fill= mean/mode/constant
            imputed_missing = simple.fit_transform(self.miss_col_.reshape(-1,1))

        elif method == "simple_stratified": # assigning mode and mean according within each target class, only applicable to categorical targets
            assert self.classification, "Only classification problem is eligible for simple_stratified imputation method"
            if self.miss_val_type == "continuous":
                strat_fill = self.df_miss_target_.groupby(self.target_name_).agg({self.miss_col_name_:simple_fill}).reset_index()
            else: # Apply to all categorical
                strat_fill = self.df_miss_target_.groupby(self.target_name_).agg({self.miss_col_name_:pd.Series.mode}).reset_index()

            # Merge to get mode then filter
            df0 = self.df_miss_target_.merge(strat_fill,on=self.target_name_,suffixes=("","_1"),how="left")
            # Replace original missing values with stratified initial guess value
            df0.loc[self.index_missing_,self.miss_col_name_] = df0.loc[self.index_missing_,self.miss_col_name_ + "_1"] # Note indices are preserved so loc and iloc of columns gives the same answer
            imputed_missing = df0[self.miss_col_name_].values.reshape(-1, 1)

        elif method == "knn":
            assert self.x_rest_ is not None, "Knn imputation requires other input features to be included in df_restX"
            
            # Append encoded missing column to the rest of X
            if self.miss_col_.dtype == "object":
                adj_miss_col = encode_arr(arr0=self.miss_col_,index_missing=self.index_missing_,
                                          index_known=self.index_known_,miss_replace=np.nan) # Encode into numerical with np.nan filled 
            else:
                assert (np.isnan(self.miss_col_).any() & (self.miss_col_.dtype != "object")), "Numeric missing columns must already have nan encoded"
                adj_miss_col = self.miss_col_

            # Regenerate entire X space for knnimputer
            arr_X = np.append(self.x_rest_,adj_miss_col.reshape(-1,1),axis=1)
            # Use Knnimputer to get initial guess of X
            imputer = KNNImputer(n_neighbors = n_neighbors, weights = "uniform")
            imputed_dataset = imputer.fit_transform(arr_X)
            if self.miss_val_type == "continuous":
                imputed_missing = imputed_dataset[:,-1].reshape(-1, 1)
            else: 
                imputed_missing = np.round(imputed_dataset[:,-1]).reshape(-1, 1) # Rounding required in the case of n_neighbour>1 for categorical values

        elif method == "manual":
            assert miss_array.size, "An input array required for filling method == manual"
            assert len(miss_array) == len(self.miss_col_), "Input array must have the same length as original array"
            imputed_missing = miss_array.reshape(-1, 1)
         
        # Transform filled initial missing col into numerical if necessary, or correct type of nominal / ordinal
        if self.miss_val_type == "continuous":
             imputed_missing_encoded = imputed_missing
        else:
            if self.miss_val_type == "categorical_nominal":
                imputed_missing_encoded = OneHotEncoder().fit_transform(imputed_missing).toarray()
            elif self.miss_val_type == "categorical_ordinal":
                imputed_missing_encoded = OneHotEncoder(drop="if_binary").fit_transform(imputed_missing).toarray()
#            
        return imputed_missing_encoded


    
    def process_input(self,simple_fill="most_frequent", method="simple", n_neighbors=1, miss_array=None):
        """Return transformed array of input features (including the feature with missing values to be filled)"""
        # Initialise missing column value
        initial_imputed_col= self.initialise_miss_guess(simple_fill=simple_fill, 
                                                        method=method, n_neighbors=n_neighbors, miss_array=miss_array)
        
        # Combine x_rest_ and imputed column for the next iteration of random forest:
        x_all = np.append(self.x_rest_,initial_imputed_col,axis=1)
        
        return x_all, initial_imputed_col


    
    def fit_forest(self,X,random_state_rf=1):
            """Run the dataset through the clustering algorithm using a single random forest 
            and return an updated proximity matrix

            Parameters:
            ----------
            X: array of input variables
                Processed data for input into the model
            random_state_rf: int, default=1
                Multiple random forests can be built. 
                Setting random_state_rf ensures randomisation of multiple forests.
            Returns:
            --------
            mod_proximity_mat: updated modified_proximity_matrix for a single random forest
            weighted_freq_mat: updated weighted modified_proximity_matrix
            """

            # Generate an empty proximity matrix
            mod_proximity_mat = self.generate_modified_proximity()

            # Generate a forest of estimators
            forest = self.generate_estimators(random_state_rf=random_state_rf)

            # Fit individual decision tree in the forest
            for est in forest:
                est[1].fit(X,self.y_)
                x_res,y_res = est[1]["res"].fit_resample(X,self.y_)  # Refit to get x_resample
                leave_id = est[1]["est"].apply(x_res)   # Get leave id for each sample
                sample_index = est[1]["res"].sample_indices_

                # Get status of only samples appeared in this tree by filtering for only missing columns values at index of sample in tree
                sample_status = []  
                for i in sample_index:
                    sample_status.append(self.miss_col_[i]) 

                # Create a dataframe to store a temporary proximity matrix from each tree
                indi_tree = pd.DataFrame(data={"sample_index":sample_index,"leave_id":leave_id,"missing_status":sample_status})
                # In the case of repeated points in the dataset due to resampling procedure,
                # each point should only be considered once in the similarity matrix
                indi_tree.drop_duplicates(inplace =True)  

                # Generate a list of unique indices of unknown data appearing in the resampled dataset unique_unknown_index  
                unique_unknown_index = indi_tree[indi_tree["sample_index"].isin(self.index_missing_)]["sample_index"]            

                # Update the proximity matrix
                for i in unique_unknown_index:
                    # Find the leave node for each missing data
                    leave_node = indi_tree[indi_tree["sample_index"]==i]["leave_id"].values[0] # Get the leave node id that the unknown index belongs to
                    # Find all known samples at this node
                    samples_at_node = indi_tree[(indi_tree["leave_id"]==leave_node) & (indi_tree["sample_index"].isin(self.index_known_))]["sample_index"] # Get all known samples at the node
                    mod_proximity_mat.loc[samples_at_node,i] = mod_proximity_mat.loc[samples_at_node,i] + 1 # add one to location [known,unknown] if known data is in the same node as unknown i
            
            # Calculate the weighted modified mod_proximity_mat
            weighted_freq_mat = self.get_weighted_proximity_matrix(mod_proximity_mat)
            
            return mod_proximity_mat, weighted_freq_mat
        
        
        
    def get_weighted_proximity_matrix(self,mod_proximity_mat):
            """Return a weighted frequency DataFrame of shape *m
                n: # of missing data points
                m: m=1 in the case self.miss_val_type == "continuous"
                   m=n_classes otherwise where n_classes is the number of classes in miss_col_
            """
            # Divide all weights by total number of trees
            # Note the last column of proximity matrix contains "known_val", any operation on weight update is on [:,:-1] only
            mod_proximity_mat.iloc[:,:-1] = mod_proximity_mat.iloc[:,:-1]/self.n_estimators
            weighted_freq_mat = pd.DataFrame()

            if self.miss_val_type == "continuous":
                for i in mod_proximity_mat.columns[:-1]: # Ignore the last column where the known value is recorded
                    if mod_proximity_mat[i].sum()>0:     # Present known values are in the same leave as the missing value
                        weighted_freq_mat[i] = mod_proximity_mat[i]/mod_proximity_mat[i].sum()
                        weighted_freq_mat[i] = weighted_freq_mat[i]*mod_proximity_mat.iloc[:,-1]
                    else:
                        weighted_freq_mat[i] = 0   
            else:
                # Sum all the weights of known data by each class of the missing variable for each missing data point
                weight_mat = mod_proximity_mat.pivot_table(
                    values=mod_proximity_mat.columns.to_list()[:-1],   
                    index="known_val",aggfunc="sum").T

                # Calculate the frequency of each value of the missing variable for known data only
                # This frequency calculation is the same for all missing data points
                freq_mat = mod_proximity_mat.pivot_table(
                    values=mod_proximity_mat.columns.to_list()[:-1],
                    index="known_val",aggfunc="count").T

                # Normalise the weight and frequency dataframe with total weight and number of known data points
                for i in weight_mat.columns:
                    weight_mat["norm_" + i] = weight_mat[i]/weight_mat.sum(axis=1) # Generate np.nan if sum=0
                    freq_mat["norm_" + i] = freq_mat[i]/freq_mat.sum(axis=1) # Generate np.nan if sum=0
                    # Multiply the frequency of response with weight to get the weighted frequency
                    weighted_freq_mat[i] = weight_mat["norm_" + i]*freq_mat["norm_" + i]
                    
            # Warning if there are missing data not yet imputed        
            if weighted_freq_mat.sum(axis=1).isin([0,np.nan]).any():
                warnings.warn("For some missing data points: Either no known datapoints are present in the same leaf or missing data points not selected in the resampling process of forest building \
                Increase tree complexity might help.")

            return weighted_freq_mat
        
        
        
    def transform_forest(self,weighted_freq_mat):
        """Return updated missing values
        
        Returns:
        --------
        updated_miss_col: array of updated missing column (inluding known values and newly imputed missing values)
        pred_val: array of updated missing values only
        pred_dis: str, distribution of updated missing values
            - For continuous miss_val_type_: the range of filled values is reported.
            - For others: the distribution of categorical values of filled values is repored.
        check_na: int, # of missing data points yet to be imputed. There are 2 possible reasons for this: 
            (1) missing data points not selected in resampling processing of tree building for the entire forest.
                Increase the number of estimators in the forest might help.
            (2) No known data points present in the same leave as the unknown data points.
                Increase tree complexity might help, e.g., increasing max_depth.
        """
        # Calculate the weighted frequency promixity matrix
        updated_miss_col = self.miss_col_.copy()

        if self.miss_val_type == "continuous":
            pred_val = weighted_freq_mat.sum()
            check_na = (pred_val==0.0).sum()
            
            updated_miss_col[self.index_missing_] = pred_val.values
            updated_miss_col[updated_miss_col==0.0] = np.median(updated_miss_col)
            
            pred_dis = "min-max fill range: {:.2f}-{:.2f}".format(np.min(pred_val[pred_val>0]),np.max(pred_val))

        else:
            # Find index of column with maximum weighted frequency
            pred_val = weighted_freq_mat.idxmax(axis=1)
            # Check the distribution of predicted values
            pred_dis = pred_val.value_counts()
            # Update the prediction of missing variable
            updated_miss_col[self.index_missing_] = pred_val.values
            # sum will have na values if the unknown datapoint was not selected in any of the resampled trees 
            check_na = pred_val.isna().sum()
            updated_miss_col[updated_miss_col!=updated_miss_col] = pd.DataFrame(updated_miss_col).mode()[0] # fill na with mode

        return updated_miss_col, pred_val, pred_dis, check_na
    
    
    
    def fit_transform(self,df,target_name,miss_col_name, miss_val=np.nan,simple_fill="most_frequent", 
            method="simple", n_neighbors=1, miss_array=None, ordinal_list=[]):
        """
        Run the dataset through multiple iteration of clustering algorithm using random forest
        
        Parameters:
        -----------
        df: pd DataFrame of input and target variables
        target_name: name of target variable as recorded in df
        miss_col_name: name of feature with missing values to be filled, same as recorded in df
        miss_val = dtype, default=np.nan
            Value denoted as missing, same as recorded in df[miss_col_name]
        ordinal_list: list, default=[]
            List of ordinal input features, same as recorded in df
        method: {"simple", "simple_stratified", "knn", "manual"}, default="simple"
            Imputation method for initial rough guess of missing values before running Random Forest clustering algorithm
            - "simple": Use SimpleImputer(). "constant" option is omitted in this version. Require simple_fill value.
            - "simple_stratified": Group data by target variable before calculating the initial guess values. Require simple_fill value
            Only available for categorical target variables (classification=True).
            - "knn": Use KNNImputer(). Require n_neighbors value.
            - "manual": manual input an array with same indices and size as original missing array. Require miss_array.
        simple_fill: {"most_frequent", "mean", "median"}, default="most frequent"
            - simple_fill option needs to be compatible with miss_val_type_, e.g., "mean" can only be used for miss_val_type_=continuous 
        n_neighbors: int, default=1 
            Used in KNNImputer() option.
        miss_array: 1D array, default=None
            Manually input array to be used as initial guess. Must have the same dimension as original miss_col_.   
        
        Returns:
        --------
        updated_imputed_col: ndarray of updated features with filled missing value (same dimension as miss_col_)
        
        """
                
        # Define the problem - Generate attributes
        self.define_problem(df=df,target_name=target_name,miss_col_name=miss_col_name,miss_val=miss_val,ordinal_list=ordinal_list)
        
        # Initialise input for first forest run
        x_all, initial_imputed_col = self.process_input(simple_fill=simple_fill, 
                                                        method=method, n_neighbors=n_neighbors, miss_array=miss_array)
        self.mod_proximity_mat_ = {}
        self.weighted_freq_mat_ = {}
        self.updated_miss_col_ = {}
        
        convergence = np.nan
        pred_val0 = [0]
        check_shape = x_all.shape
        update_shape = check_shape
        prev_imputed = initial_imputed_col
        
        for j in range(self.n_iter):
            # Fit each forest
            assert check_shape == update_shape, "Shape of input array must stay consistent at every iteration"
            mod_proximity_mat,weighted_freq_mat = self.fit_forest(x_all,random_state_rf=j)
            
            # Get predicted value for the missing column
            updated_miss_col, pred_val, pred_dis, check_na = self.transform_forest(weighted_freq_mat)
            # Save results in dictionary
            self.mod_proximity_mat_[f"iteration_{j}"] = mod_proximity_mat
            self.weighted_freq_mat_[f"iteration_{j}"] = weighted_freq_mat
            self.updated_miss_col_[f"iteration_{j}"] = updated_miss_col

            # Update convergence
            if j>0:
                convergence = self.calculate_convergence(pred_val0,pred_val)
            
            print("Iteration {}: Number of missing samples not yet imputed: {}".format(j,check_na))
            print(pred_dis)
            print("Convergence: {:.2f}%".format(convergence))
            print("------------------------------------------------------------")

            # Restart a new iteration with updated guess for miss_col_
            updated_imputed_col= self.initialise_miss_guess(method="manual", miss_array=updated_miss_col)
            x_all = np.append(self.x_rest_,updated_imputed_col,axis=1)
            
            # Update input variables with new predicted results from forest for convergence calculation
            pred_val0 = pred_val
            update_shape = x_all.shape
            prev_imputed = updated_imputed_col 
            
        return updated_imputed_col
    
    
    
    def calculate_convergence(self,pred_val0,pred_val1):
        """
        Calculate convergence after every random forest iteration. Return a convergence measurement
        Note the convergence definition here is not used in the originally proposed algorithm but derived here as a practical metric to measure iteration convergence    
        - Convergence = (accuracy_measurement)*(% of imputed values)
        - accuracy_measurement:
            - continuous: changes in estimated values of missing data after each iteration
            - categorical: number of similar guesses out of the total missing data population
        - % of imputed values: total missing value with a found filled value after clustering algorithm
            - Note: a legitimate guess of a missing value might not appear in the final weighted_freq_mat_ for 2 possible reasons:
            (1) Missing data points not selected in resampling processing of tree building for the entire forest.
            (2) No known data points present in the same leave as the unknown data points.
        """
        # For continuous missing data
        if self.miss_val_type == "continuous":
            rows_with_nan0 = pred_val0[pred_val0==0.0].index
            rows_with_nan1 = pred_val1[pred_val1==0.0].index
            
            na_tot = len(set(rows_with_nan0).union(rows_with_nan1))
            data_tot = min(pred_val0.shape[0],pred_val1.shape[0])
            
            pred_val0 = pred_val0[pred_val0>0]
            pred_val1 = pred_val1[pred_val1>0]
            
            check = pred_val0.to_frame().join(pred_val1.to_frame(),how="outer",lsuffix='_0', rsuffix='_1')
            same_pred = (1-((check.iloc[:,0] - check.iloc[:,1])/check.iloc[:,0])).sum()/min(pred_val0.shape[0],pred_val1.shape[0])
        # For categorical missing data
        else:    
            rows_with_nan0 = [index for index, row in pred_val0.to_frame().iterrows() if row.isnull().any()]
            rows_with_nan1 = [index for index, row in pred_val1.to_frame().iterrows() if row.isnull().any()]

            na_tot = len(set(rows_with_nan0).union(rows_with_nan1))
            data_tot = min(pred_val0.shape[0],pred_val1.shape[0])

            pred_val0.dropna(inplace=True)
            pred_val1.dropna(inplace=True)
            check = pred_val0.to_frame().join(pred_val1.to_frame(),how="outer",lsuffix='_0', rsuffix='_1')
            same_pred = ((check.iloc[:,0] == check.iloc[:,1]).sum())/min(pred_val0.shape[0],pred_val1.shape[0])

        convergence = same_pred*100*(1-na_tot/data_tot)
        
        return convergence
