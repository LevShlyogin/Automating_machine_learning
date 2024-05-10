import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

@pytest.fixture()
def load_ys():
    ys = np.loadtxt("data/ys2.csv", delimiter=";")
    return ys

@pytest.fixture()
def load_pred():
    pred = np.loadtxt("data/pred2.csv", delimiter=";")
    return pred

def test_mse(load_ys, load_pred):
    assert mean_squared_error(load_ys, load_pred) < 1
    
def test_r2(load_ys, load_pred):
    assert r2_score(load_ys, load_pred) > 0.9
