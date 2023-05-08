import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

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


### PREDICTION ANALYSIS FUNCTIONS

def filtered_df_two_columns(df, col1, col2):
    """
    Returns a new dataframe that contains only values in both columns (no NA values)

    Parameters:
    the dataframe and 2 variable column names

    Returns: 
    pandas.Dataframe that only contains values in both columns (no NA)
    """
    # Check that the input DataFrame contains both column names
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Input DataFrame does not contain both specified column names")

    # Select only the specified columns
    selected_cols = [col1, col2]
    new_df = df[selected_cols]

    # Use pandas' built-in methods to check which cells have numbers
    is_numeric = pd.to_numeric(new_df.iloc[:, 0], errors="coerce").notnull() & pd.to_numeric(new_df.iloc[:, 1], errors="coerce").notnull()

    # Filter the DataFrame to keep only the rows with numbers in both columns
    filtered_df = new_df.loc[is_numeric]

    return filtered_df

def pearson_corr_coef(x, y):
    """
    Computes the Pearson correlation coefficient

    Parameters: two lists of equal length

    Returns: an array with the Pearson correlation coefficient
    """
    # Checking if the values in the list are the same
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length")

    # Find Covariance
    covariance = np.cov(x, y)

    # Standard deviation of x and y
    stdx = np.std(x)
    stdy = np.std(y)

    # Returning Correlation coefficient
    return covariance / (stdx * stdy)

def prediction_analysis(filtered_data):
    """
    Performs a simple linear regression analysis on the input data

    Parameters: Filtered Dataframe that only has two columns

    Returns:
    Generates a list of prediction values based on the input variables
    """
    # Check that the input DataFrame contains only two columns
    if len(filtered_data.columns) < 2:
        raise ValueError("Input DataFrame contains less than 2 columns (Must only include 2)")
    if len(filtered_data.columns) > 2:
        raise ValueError("Input DataFrame contains more than 2 columns (Must only include 2)")

    # Fit a simple linear regression model to the input data
    model = LinearRegression()
    X = filtered_data.iloc[:, 0].values.reshape(-1, 1)
    y = filtered_data.iloc[:, 1].values.reshape(-1, 1)
    model.fit(X, y)

    # Generate predictions for the output column based on the input variables
    pred_col = model.predict(X)

    return pred_col

def regression_analysis_results(y_col, pred_col, filename):
    """
    Performs a regression analysis to determine how well the model predicts the actual values.

    Parameters:
    Columns containing actual values and predicted values
    filename that you want the figure to be saved as

    Returns:
    Calculates and returns the R-squared value and generates a scatter plot of the predicted values 
    versus the actual values.
    """
    # Check that the input DataFrame contains both columns
    if len(y_col) != len(pred_col):
        raise ValueError("Input columns are not the same length")

    # Calculate the R-squared value
    r_squared = r2_score(y_col, pred_col)

    # Generates a scatter plot of the observed vs. fitted values
    plt.scatter(y_col, pred_col)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Analysis Results')
    plt.show()

    filepath = os.path.join('figures', filename)
    plt.savefig(filepath)

    return r_squared

def calculate_MSE(y_col, pred_col):
    """
    Evaluates a set of predictive values given the actual values using the Mean Squared Error (MSE) metric

    Parameters:
    predicted_values: A list or numpy array of predicted values.
    actual_values: A list or numpy array of actual values.

    Returns:
    The Mean Squared Error (MSE) between the predicted and actual values.
    """
    # Check if the inputs are of equal length
    if len(y_col) != len(pred_col):
        raise ValueError("The predicted and actual values must have the same length.")

    # Calculate the Mean Squared Error (MSE) between the predicted and actual values
    mse = np.mean((np.array(pred_col) - np.array(y_col))**2)

    return mse