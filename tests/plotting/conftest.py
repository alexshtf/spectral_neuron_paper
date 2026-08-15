import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
