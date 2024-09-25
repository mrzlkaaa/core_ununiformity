from collections import defaultdict
import os
from statistics import mean
from textwrap import indent
import numpy as np
import pandas as pd
from pprint import pprint 

from distutils import core

from . import init_nuclides as nuclides_list


class PDCHandler:
    CELLS_6LAYERS_HEIGHT = {
        "7-6": [
            1,
            21,
            41,
            61,
            81,
            101
        ],
        "7-5": [
            2,
            22,
            42,
            62,
            82,
            102
        ],
        "7-4": [
            3,
            23,
            43,
            63,
            83,
            103
        ],
        "7-3": [
            4,
            24,
            44,
            64,
            84,
            104
        ],
        "6-6": [
            5,
            25,
            45,
            65,
            85,
            105
        ],
        "6-5": [
            6,
            26,
            46,
            66,
            86,
            106
        ],
        "6-4": [
            7,
            27,
            47,
            67,
            87,
            107
        ],
        "6-3": [
            8,
            28,
            48,
            68,
            88,
            108
        ],
        "5-6": [
            9,
            29,
            49,
            69,
            89,
            109
        ],
        "5-3": [
            10,
            30,
            50,
            70,
            90,
            110
        ],
        "4-6": [
            11,
            31,
            51,
            71,
            91,
            111
        ],
        "4-3": [
            12,
            32,
            52,
            72,
            92,
            112
        ],
        "3-6": [
            13,
            33,
            53,
            73,
            93,
            113
        ],
        "3-5": [
            14,
            34,
            54,
            74,
            94,
            114
        ],
        "3-4": [
            15,
            35,
            55,
            75,
            95,
            115
        ],
        "3-3": [
            16,
            36,
            56,
            76,
            96,
            116
        ],
        "2-6": [
            17,
            37,
            57,
            77,
            97,
            117
        ],
        "2-5": [
            18,
            38,
            58,
            78,
            98,
            118
        ],
        "2-4": [
            19,
            39,
            59,
            79,
            99,
            119
        ],
        "2-3": [
            20,
            40,
            60,
            80,
            100,
            120
        ]
    }

    #* fresh fuel
    FRESH_FUEL: dict = {
        "U235": 2.4600E-03, #* only this part is about to be modified
        "AL":   5.3180E-02,
        "U238": 2.4600E-04,
        "U234": 2.7330E-05,
        "O16":  5.4660E-03
    }
    CELLS_DATA_STORAGE = "cells_data.xlsx"

    def __init__(
        self,
        pdc_file: str | None = None
    ):
        self.nuclides_list = nuclides_list()
        self.nuclides_list.sort()
        self.pdc_file = pdc_file
    
    def _core_burnup_processing(
        self,
        core_burnup
    ):
        return abs(np.asarray(core_burnup))


    def _read_pdc(
        self,
        path: str
    ):
        with open(path, "r") as f:
            content = f.readlines()
        return content

    def _get_cells_data(
        self
    ):
        '''
        #* Iterates over fuel materials and collects
        #* U235 densities for each cell
        #* From collected cells get axial distribution
        #* and burnup in each cell
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*  u235 densities in each cell,
        #*  axial distribution for every cell, 
        #*  burnup in each cell
        '''
        axial_distr = {k: {} for k in self.CELLS_6LAYERS_HEIGHT.keys()}
        
        nuclides_storage = {
            nuc: 0.0 for nuc in self.nuclides_list
        }
        
        path = os.path.join(
            os.path.dirname(__file__),
            "pdcs_collection",
            self.pdc_file
        )
        density_search = False
        current_matr = 0
        for i in self._read_pdc(path):
            if "MATR" in i and ~density_search:
                density_search = not density_search
                current_matr = int(i.split()[1])

                #* storage reset to prevent duplicates
                nuclides_storage = {
                    nuc: 0.0 for nuc in self.nuclides_list
                }
            
            elif "stop" in i and density_search:
                density_search = not density_search
                nuclides_storage =  dict(sorted(nuclides_storage.items()))

                for k,v in self.CELLS_6LAYERS_HEIGHT.items():
                    if current_matr in v:
                        
                        if axial_distr[k].get("keys") is None and axial_distr[k].get("values") is None:
                            axial_distr[k] = defaultdict(list)
                            axial_distr[k]["keys"] = list(nuclides_storage.copy().keys())
                        
                        axial_distr[k]["values"].append(
                            list(nuclides_storage.copy().values())
                        )
                        nuclides_storage.clear()
                        break
            elif density_search:
                for nuclide in self.nuclides_list:
                    if nuclide in i:
                        
                        nuclide_density = float(i.split()[-1])
                        nuclides_storage[nuclide] = nuclide_density
                        break
                
        
        burnup_map = []
        for cell in axial_distr.keys():
            axial_distr[cell]["means"] = np.asarray(axial_distr[cell]["values"]).mean(axis=0)
            u5_dens = []
            u5_pos = axial_distr[cell]["keys"].index("U235")
            for layer in axial_distr[cell]["values"]:
                u5_dens.append(layer[u5_pos])
            axial_distr[cell]["u5_dens"] = np.asarray(u5_dens)
            axial_distr[cell]["u5_aver"] = ( 1 - np.asarray(u5_dens).mean() / self.FRESH_FUEL["U235"] ) * 100
            axial_distr[cell]["u5_axial"] = np.asarray(u5_dens) / np.asarray(u5_dens).mean()

            burnup_map.append(
                axial_distr[cell]["u5_aver"]
            )
            
        return axial_distr, burnup_map
    

    def collect_cells_data(
        self
    ):
        '''
        #* Method description
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        if self.pdc_file is None:
            print("< pdc_file > variable is None")
            return
        axial_dist_data, burnup_map = self._get_cells_data()
        d = {}
        df = pd.DataFrame()

        for n, k in enumerate(axial_dist_data.keys()):
            d["u5_aver"] = float(
                "{:.2f}".format(axial_dist_data[k]["u5_aver"])
            )
            
            for i in range(len(axial_dist_data[k]["u5_dens"])):

                d[f"u5_dens_{i}"] = axial_dist_data[k]["u5_dens"][i]
                d[f"u5_axial_{i}"] = axial_dist_data[k]["u5_axial"][i]
            d = { 
                **d, 
                **dict(zip(axial_dist_data[k]["keys"], axial_dist_data[k]["means"]))
            }

            df = pd.concat([
                df,
                pd.DataFrame(data=d, index=[n])
            ])
            
        #* drop U235 column
        df = df.drop("U235", axis=1)

        df.set_index("u5_aver", inplace=True)
        # df.to_excel(self.CELLS_DATA_STORAGE)
        return df

    def persist_cells_data(
        self,
        df
    ):
        '''
        #* Writes to .csv cells data
        #* generated in self._get_cells_data and
        #* refactored to df in self.collect_cells_data
        #* Row in .csv file consist all required data
        #* to generate a core with random burnup
        #* Persist method do not allows < name > duplicates
        #* to write
        #* If duplicate finds it removes from df
        #* Parameters
        #* ----------
        #* df
        #*  DataFrame of cells data
        #* Raises
        #* ----------
        #*  None
        #* Returns
        #* ----------
        #*  None
        '''
        os.chdir(os.path.dirname(__file__))
        stored_data = pd.read_excel(self.CELLS_DATA_STORAGE, index_col="u5_aver")
        
        stored_burnups = stored_data.index
        burnups = df.index
        
        matched = list(set(list(stored_burnups)).intersection(set(list(burnups))))
        
        if len(matched) > 0:
            df = df.drop(index=matched)

        df = pd.concat(
            [
                stored_data, 
                df
            ],
            axis=0
        )
        # df = df.reset_index()
        # df = df.drop(columns="index", axis=1)
        
        df.to_excel(
            os.path.join(
                os.path.dirname(__file__),
                self.CELLS_DATA_STORAGE
            )
        )
        

    def core_augmentation(
        self,
        core_burnup: list | None = None,
        percentage_norm: bool = True

    ):
        core_burnup = self._core_burnup_processing(core_burnup)
        if percentage_norm:
            core_burnup = core_burnup * 100

        stored_data = pd.read_excel(
            os.path.join(
                os.path.dirname(__file__),
                self.CELLS_DATA_STORAGE
            ), 
            index_col=0
        )
        
        #* matr is key, values dict of nucl - dens pairs
        materials = {}
        axial_cols = [c for c in stored_data.columns if "axial" in c]
        dens_cols = [c for c in stored_data.columns if "dens" in c]

        aver_burnup = stored_data.index
        df_axial_shapes = stored_data.loc[:, axial_cols]
        df_nuclides = stored_data.drop(columns=[*axial_cols, *dens_cols], axis=1)


        densities = {}
        for n, i in enumerate(core_burnup):
            #* Ind of closest burnup found in storage
            diff = list(abs((i - aver_burnup)))
            
            sort_diff = diff.copy()
            sort_diff.sort()
            
            ind = diff.index(
                sort_diff[0]
            )
            #* prevent from taking densitities for a fresh fuel
            #* IN CASE  fresh fuel is closest one (for FAs with burnup < 1%)
            if i == sort_diff[0] and i != 0.0:
                ind = diff.index(
                    sort_diff[1]
                )   
            # .sort_values()[0]
            # print(ind)
            # print(stored_data.iloc[ind])
            # sorted(np.absolute((i*100 - storage["aver_burnup"])))
            cell_mats = list(self.CELLS_6LAYERS_HEIGHT.values())[n]
            # print(cell_mats)
            for itr, mat_num in enumerate(cell_mats):
                #* u235 density normalization
                densities["U235"] = self.FRESH_FUEL["U235"]\
                    * ( 1 - i / 100 )\
                    * df_axial_shapes.iloc[ind][itr]

                densities["AL"] = self.FRESH_FUEL["AL"]
                densities["U238"] = self.FRESH_FUEL["U238"]
                densities["U234"] = self.FRESH_FUEL["U234"]
                densities["O16"] = self.FRESH_FUEL["O16"]

                
                for nucl in df_nuclides.columns:
                    densities[nucl] = df_nuclides.iloc[ind][nucl] / df_axial_shapes.iloc[ind][itr]

                materials[mat_num] = densities.copy()
                densities.clear()
                # pprint(materials, indent=6)
        materials = dict(sorted(materials.items()))
        return materials

    def _make_mat_pattern(
        self,
        mat_num,
        densities
    ):
        
        d = list(map(lambda x: f"{x[0]} {'{:.4e}'.format(x[1])}", densities.items()))
        if mat_num < 9:
            mat_pattern = f"MATR          {mat_num}          0   -10.0\n"\
            + "\n".join(d) + "\nstop\n"
        elif mat_num > 9 and mat_num < 99:
            mat_pattern = f"MATR         {mat_num}          0   -10.0\n"\
                + "\n".join(d) + "\nstop\n"
        elif mat_num > 99 and mat_num < 999:
            mat_pattern = f"MATR        {mat_num}          0   -10.0\n"\
                + "\n".join(d) + "\nstop\n"
        else:
            mat_pattern = f"MATR       {mat_num}          0   -10.0\n"\
                + "\n".join(d) + "\nstop\n"
        return mat_pattern
        # return


    def _core_map_constructor(
        self,
        core_burnup
    ):
        core_map = list(map(
                lambda x: "{:.2f}".format(x),
                list(core_burnup)
            )
        )
        core_map = np.asarray(
            [
                *core_map[:9], 
                "Be", 
                "Be",
                *core_map[9:11],
                "Be", 
                "Be",
                *core_map[11:]
            ]
            
        ).reshape(6,4)
        
        return core_map.tolist()

    def make_augmented_pdc(
        self,
        matetials: list | np.ndarray,
        output_file_name: str = "random_burn.PDC",
        to_folder: str | None = None,
        core_map_export: bool = False,
        core_burnup: list | None = None,
        percentage_norm: bool = True
    ):
        '''
        #* Creates .PDC file using augmentation approach
        #* descrobed in methods above
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        if core_burnup is not None and percentage_norm:
            core_burnup = self._core_burnup_processing(core_burnup) * 100
        elif core_burnup is not None and not percentage_norm:
            core_burnup = self._core_burnup_processing(core_burnup)

        if core_burnup is None and core_map_export:
            raise TypeError("'core_burnup' variable is None, but 'core_map_export' is set to True")
    
        file_save_path = os.path.join(
            os.path.dirname(__file__),
            "random_burns",
            output_file_name
        )
        map_save_path = os.path.join(
            os.path.dirname(__file__),
            "random_burns",
            f"!_map_{output_file_name[:-4]}.txt"
        )

        if to_folder:
            folder_path = os.path.join(
                os.path.dirname(__file__),
                "random_burns",
                to_folder
            )
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            
            file_save_path = os.path.join(
                folder_path,
                output_file_name
            )

            map_save_path = os.path.join(
                folder_path,
                f"!_map_{output_file_name[:-4]}.txt"
            )


        mats = []
        for mat, nucls in matetials.items():
            mats.append(
                self._make_mat_pattern(
                    mat, nucls
                )
            )
        # print(mats)
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "!template_burn.PDC"
            ),
            "r"
        ) as fr:
            content = fr.readlines()

        content = [content[0], *mats, *content[1:]]
        
        with open(
            file_save_path,
            "w"
        ) as fw:
            fw.writelines(content)

        if core_map_export:
            core_map = self._core_map_constructor(core_burnup)
            print(core_map)
            with open(
                map_save_path,
                "w"
            ) as f:
                for line in core_map:
                    f.writelines(
                        "\t".join(line)
                    )
                    f.write("\n")
                    

        return

    def extract_data(
        self
    ):
        return