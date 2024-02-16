import unittest
import pandas as pd
import numpy as np
import re

try:
    from .core import Cartogram
except ImportError:
    from core import Cartogram


class Burnup(Cartogram):
    U_RATE = 1.23
    FUEL_CONS = {
        "MWd": U_RATE / 20,  #* grams for 1MWd
        "MWh": U_RATE / 24  / 20 #* grams for 1MWh
    }
    
    ENERGY = {
        "d": 24,
        "h": 1,
        "MWd": 1 / 24,
        "MWh": 1
    }

    def __init__(
        self,
        df:  pd.core.frame.DataFrame
    ) -> None:
        super().__init__(df)


    def _get_fuel_consumption(
        self, 
        energy_produced: float
    ) -> float:
        
        return
    
    def _parse_energy(
        self,
        string:str, 
        power: float = 7.19 #* units of MW 
    ) -> float:

        _pattern = r'(\d+\.\d+|\d+)(\w*)'
        
        val, unit = re.findall(_pattern, string)[0]

        try:
            val = float(val)
        except ValueError:
            raise ValueError("Cannot parse given time on power")

        if unit == "":
            raise ValueError("The unit of time on power is not specified")
        
        
        _energy_coef = self.ENERGY[unit] * power
        
        return val * _energy_coef #* units of MWh

    def simulate_burnup(
        self,
        # time_onpower: str | None = None,
        en_produced: str,
        index_: int | None = None
    ):

        produced = self._parse_energy(en_produced)
        #* consumed by each cell if non-uniformity coef is equal to 1
        fuel_consumed = self.FUEL_CONS["MWh"] * produced 

        if index_ is None:
            st = 0
            fn = len(self.df) - 1
        
        else:
            st = index_
            fn = index_

        df_to_simulate = self.df.loc[
            st: fn, 
            [
                *self.CORE_PARTS["burnup"]["ALL_CELLS"],
                *self.CORE_PARTS["left"]["ALL_CELLS"],
                *self.CORE_PARTS["fuel_type"]["ALL_CELLS"],
                
            ]
        ]


        consumed_by_cells = self.df.loc[
            st: fn, 
            [
                *self.CORE_PARTS["coef"]["ALL_CELLS"]
                
            ]
        ].to_numpy() * fuel_consumed
        
        df_to_simulate.loc[:, self.CORE_PARTS["burnup"]["ALL_CELLS"]] += consumed_by_cells
        df_to_simulate.loc[:, self.CORE_PARTS["left"]["ALL_CELLS"]] -= consumed_by_cells

        return df_to_simulate
