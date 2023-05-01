from tools import utils as ul
import pytest
import numpy as np
import pandas as pd


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

    # Check that the output DataFrame has two columns ('Feature' and 'Importance')
    assert set(importance_df.columns) == set(['Feature', 'Importance'])

    # Check that the sum of the feature importances equals 1
    assert np.isclose(importance_df['Importance'].sum(), 1.0)
