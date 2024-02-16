from . import *
from itertools import permutations
from main.core import Cartogram



@pytest.fixture
def c():
    c = Cartogram(
        df=pd.read_excel(file)
    )
    return c


def test_make_static_data(c):
    data = c._make_static_data()
    print(data)
    assert 0

def test_quick_processing():
    c = Cartogram(
        df=pd.read_excel(file)
    )
    c.quick_processing(process="all")
    
    assert 0

def test_from_df_to_core(c):

    print(c.df.loc[0, c.CORE_PARTS["burnup"]["ALL_CELLS"]])
    core = c.from_df_to_core(c.df.loc[0, c.CORE_PARTS["burnup"]["ALL_CELLS"]])

    kernel = np.asarray([
        [0,1,0],
        [1,5,1],
        [0,1,0],
    ])

    cv = scipy.signal.convolve2d(core, kernel, mode="same")
    print(core)
    print(cv)
    print( core / cv)

    assert 0

def test_from_arr_to_df():
    
    df = Cartogram().from_arr_to_df(arr, data_type="burnup")
    print(df)
    assert 0

def test_permutations():
    cells = [
        "7-6",	"7-5",	"7-4",	"7-3",		
        "5-6",	"5-3",	"4-6",	"4-3",
        "2-6",	"2-5",	"2-4",	"2-3"
    ]

    p = permutations(cells)
    i = 0
    for _ in p:
        i+=1
    print(i)

    assert 0