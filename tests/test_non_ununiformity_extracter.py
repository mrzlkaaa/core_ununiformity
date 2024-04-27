import pytest
from main.pdc_handlers.non_ununiformity_extracter import *

def test_extract_fin():
    df = extract_fin(
        path="/mnt/c/Users/Nikita/Desktop/codes/ML/IRTT_reactor/core_uniniformity/main/pdc_handlers/burn.FIN",
        df=None
    )

    df = extract_fin(
        path="/mnt/c/Users/Nikita/Desktop/codes/ML/IRTT_reactor/core_uniniformity/main/pdc_handlers/burn2.FIN",
        df=df
    )
    print(df)
    assert 0