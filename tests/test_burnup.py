from . import *
from main.burnup import Burnup


@pytest.fixture
def b():
    return Burnup(df=pd.read_excel(file))

def test_parse_energy(b):
    res = b._parse_energy("2.0833d")
    print(res)
    assert 0

def test_simulate_burnup(b):
    b.simulate_burnup("100h", 0)