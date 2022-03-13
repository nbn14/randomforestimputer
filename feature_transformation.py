import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer

def transform_cat_column(df,ordinal_list=[]):
    """This function takes in a DataFrame of categorical and continuous features.
    Parameters:
    ----------
    df: pd.DataFrame
    ordinal_list: list
        List of ordinal catergorical column names in df 
    Return: 
    -------
    array_transformed: array
        encoded array
    """
  
    nominal_list = []
    binary_list = []
    for col,val in zip(df.dtypes.index,df.dtypes):
        # Nomial features with more than 2 unique values -> onehot encoded, with 2 values = binary encoded
        if (val == "object") and (ordinal_list.count(col)<1): # Check whether the object is ordinal
            if df[col].nunique() < 3:     # Search categorical features with binary values  
                binary_list.append(col)
            elif df[col].nunique() > 2:
                nominal_list.append(col)
    transformers = [("label",OrdinalEncoder(),ordinal_list),("ohen",OneHotEncoder(),nominal_list),("oheb",OneHotEncoder(drop="if_binary"),binary_list)] # OneHotEncoder(drop='if_binary').fit(X) or using this for binary onehotencoder
    col_trans = ColumnTransformer(transformers = transformers, remainder="passthrough")
    array_transformed = col_trans.fit_transform(df)
    
    return array_transformed



def encode_arr(arr0,index_missing,index_known,miss_replace=np.nan):
    '''This function encode missing value and encode non-missing values as numerical assuming ordinal categorical values
    Parameters:
    ----------
    arr0: 1D array
    index_missing: list 
        List of indices of missing-data
    index_known: list
        List of indices of known data (non-missing)
    miss_replace: np.nan or any dtype
        Replace all missing data with this value
    
    Return:
    ------- 
    arr: encoded 1D array  
    '''
    
    arr = arr0.copy()  # Preserve the original dataset
    assert arr.dtype == "O", "Input data might already be numerical - no need the transformation"
    # Keep only non-missing data. Note miss_val
    known = arr[index_known].reshape(-1,1)
    # Encode non-missing data and replace in the original
    encode_known = OrdinalEncoder().fit_transform(known)
    arr[index_known]= np.squeeze(encode_known)
    # Replace missing value with chosen value
    arr[index_missing] = miss_replace
    
    return arr


