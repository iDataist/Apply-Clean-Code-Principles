import pytest

def df_plugin():
    return None

def scaled_df_plugin():
    return None

def encoded_df_plugin():
    return None

def split_plugin():
    return None

def pytest_configure():
    pytest.df = df_plugin()
    pytest.scaled_df = scaled_df_plugin()
    pytest.encoded_df = encoded_df_plugin()
    pytest.split = split_plugin()
