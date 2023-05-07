import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesRegressor

# Set figure size
plt.rcParams["figure.figsize"] = (14, 8) 

### EXPLORATORY ANALYSIS FUNCTIONS

def combine_columns(df, col1, col2, combined_col):
    """
    Returns dataframe with a new column that combines the values of two columns 
    (If one column has an NA value, the other column will replace the spot with its value)

    Parameters:
    df (pandas.DataFrame): The data containing the features.
    col1 and col2: Two columns you want to combine
    combined_col: New column name

    Returns:
    pandas.DataFrame with new column
    """
    new_col = []
    for i, row in df.iterrows():
        if pd.isna(row[col1]) and pd.isna(row[col2]):
            new_col.append(np.nan)
        elif pd.isna(row[col1]):
            new_col.append(row[col2])
        else:
            new_col.append(row[col1])
    df[combined_col] = new_col
    return df


def compute_feature_importance(df, target_column):
    """
    Computes feature importance using the Extra Trees Regressor algorithm.

    Parameters:
    df (pandas.DataFrame): The data containing the features.

    Returns:
    pandas.DataFrame: The feature importance scores.
    """
    # Split the data into X (features) and y (target)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train an Extra Trees Regressor model to compute feature importance
    model = ExtraTreesRegressor()
    model.fit(X, y)

    # Create a DataFrame with feature importance scores
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df


def standard_units(col):
    """
    Convert any column to standard units.
    
    Parameters: 
    col : dataframe column
    
    Returns:
    column
    """
    stand_unit = (col - np.mean(col))/np.std(col)
    return stand_unit