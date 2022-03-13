from pandas.api.types import is_numeric_dtype,is_string_dtype

def is_numeric(array):
    """Return False if any value in the array or list is not numeric"""
    for i in array:
        try:
            float(i)
        except ValueError:
            return False
        else:
            return True

def dtype_standardise(df):
    for col in df.columns:
        if ~is_numeric_dtype(df[col]):
            df[col].astype(object)
    return df     