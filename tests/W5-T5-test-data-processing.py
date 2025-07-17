import pandas as pd
from data_loading import load_data  # Now loading from flat module
import pytest

def test_data_loading_from_csv(tmp_path):
    """
    Test that load_data correctly loads a CSV file into a DataFrame.
    """
    # Create a dummy DataFrame
    test_data = pd.DataFrame({
        'Amount': [100, 200],
        'Value': [90, 180],
        'is_high_risk': [0, 1],
    })

    # Save to temporary file
    file_path = tmp_path / "dummy.csv"
    test_data.to_csv(file_path, index=False)

    # Load using the function
    df = load_data(str(file_path))

    # Assertions
    assert not df.empty
    assert df.shape == (2, 3)
    assert 'Amount' in df.columns
