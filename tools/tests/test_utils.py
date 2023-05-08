from tools import utils as ul
import pytest
import numpy as np
import pandas as pd
import os

data = pd.read_csv("data/Most-Recent-Cohorts-Institution-filtered.csv")

### Exploratory Analysis tests

def test_combinecolumns():
    df = pd.DataFrame({'A': [np.nan, 2, 3], 'B': [4, np.nan, np.nan]})
    df_test = ul.combine_columns(df, 'A', 'B', 'C')

    # Check the values of column C
    assert all(df_test.C == [4, 2, 3])

def test_featureimportance():
    df = pd.DataFrame(np.random.rand(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'target'])

    # Compute feature importance
    importance_df = ul.compute_feature_importance(df, 'target')

    # Check that the output is a pandas DataFrame
    assert isinstance(importance_df, pd.DataFrame)

    # Check that the sum of the feature importances equals 1
    assert np.isclose(importance_df['Importance'].sum(), 1.0)

def test_standard_units():
    assert type(ul.standard_units(data.RET_FT4)) is not None


### Prediction Analysis tests

def test_filtered_df():
    assert ul.filtered_df_two_columns(data, 'AVGFACSAL', 'RET_FT4').shape[1] == 2

# Testing dataframe for the next couple functions
df_test = ul.filtered_df_two_columns(data, 'AVGFACSAL', 'RET_FT4')

def test_pearson_coeff():
    assert type(ul.pearson_corr_coef(df_test.AVGFACSAL, df_test.RET_FT4)) == np.ndarray

def test_prediction_analysis():
    assert ul.prediction_analysis(df_test) is not None

x = [1, 2, 3]
y = [4, 5, 6]

def test_regression_analysis():
    regression = ul.regression_analysis_results(x, y, 'test_graph.png')
    assert os.path.exists('figures/test_graph.png')

def test_mse():
    assert type(ul.calculate_MSE(x, y)) == np.float64