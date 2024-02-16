from . import *
from main.display import Display



def test_display():
    # arr = np.arange(1,21,1)
    arr = np.array([
        0.00, 48.07, 27.07,	26.84,
        28.6, 24.07, 58.76, 7.34,	
        40.77, 15.33, 29.07, 34.91,
        26.03, 54.61, 25.97, 6.78,
        48.18, 3.61, 61.32, 32.69
    ])
    # print(arr)
    print(Display().display(arr))
    assert 0


