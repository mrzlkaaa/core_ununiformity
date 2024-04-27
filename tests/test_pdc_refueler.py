import pytest
from main.pdc_handlers.pdc_refueler import PDCHandler

core = [
            0.4221356 , 0.46820898, 0.00828161, 0.12626734, 0.29356008,
            0.26331917, 0.50193121, 0.26949621, 0.23336462, 0.39914299,
            0.32394683, 0.29006118, 0.39914299, 0.31522509, 0.3037365 ,
            0.08657985, 0.63050121, 0.11054461, 0.3883517 , 0.4426671
        ]

@pytest.fixture
def pdc():
    return PDCHandler(
        
        pdc_file="127_burn.PDC"
    )

def test_get_cells_data(pdc):
    res = pdc._get_cells_data()
    print(res)
    assert 0

def test_collect_cells_data(pdc):
    pdc.collect_cells_data()
    assert 0

def test_persist_cells_data(pdc):
    df = pdc.collect_cells_data()
    pdc.persist_cells_data(df)
    assert 0

def test_core_augmentation(pdc):
    pdc.core_augmentation(
        core_burnup=core
    )
    assert 0

def test_make_mat_pattern(pdc):
    pdc._make_mat_pattern(
        54,
        0.001731538
    )
    assert 0

def test_core_map_constructor(pdc):
    pdc._core_map_constructor()
    assert 0

def test_make_augmented_pdc(pdc):
    densities = pdc.core_augmentation(core_burnup=core)
    pdc.make_augmented_pdc(
        densities,
        output_file_name="test_random_burn.PDC",
        to_folder="random_test_0",
        core_map_export=True,
        core_burnup=core
    )
    assert 0