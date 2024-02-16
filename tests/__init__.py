import pytest
import os
import numpy as np
import pandas as pd
import scipy


inp_str = "61.43	45.74	23.95	23.66	26.20	20.48	56.21	3.51\
    	38.06	10.55	25.90	30.92	23.40	52.10	22.16	3.12	46.07	0.00	59.31	29.84"
arr = inp_str.split()

file = os.path.join(
    os.path.split(
        os.path.dirname(__file__)
    )[0],
    "main",
    "input.xlsx"
)