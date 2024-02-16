from itertools import chain
from . import *
from main.permutations import Permutations

@pytest.fixture
def p():
    return Permutations(
        df=pd.read_excel(file),
        ind=0
    )

def test_quick_examination_processing(p):
    
    p._quick_examination_processing()

    assert 0


def test_make_permutations(p):
    p.make_permutations(
        cells_from=["7-6", "7-5"], 
        cells_to=["2-3", "2-4"],
        chain_like=True
    )

    assert 0

def test_add_fresh_fuel(p):
    dfs = p.add_fresh_fuel(
        cells=["2-3", "2-5"],
        chain_like=True
    )
    assert len(dfs) == 2 and dfs[0]["2-5_l"].to_numpy() != dfs[1]["2-5_l"].to_numpy()

def test_random_permutations(p):
    p._random_permutations(5)
    assert 0

def test_cells_to_permutate_finder(p):
    res = p._cells_to_permutate_finder()
    print(res)
    assert 0