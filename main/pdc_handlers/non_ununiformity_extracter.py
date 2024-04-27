
import argparse
from importlib.resources import contents
import sys
import os
import pandas as pd

__all__ = [
    "extract_fin"   
]

parser = argparse.ArgumentParser(
    description='Extract fission reaction rates from .FIN files'
)
parser.add_argument(

    "--files", 
    action="store_true",
    help="Specify search algorithm. If passed search for .FIN files in a current folder and search over folders otherwise")

def extract_fin(
    path: str,
    df: pd.core.frame.DataFrame | None = None
):
    if df is None:
        df = pd.DataFrame()

    keyword = "NUCLIDE:          U235, REACTION:            3, ENERGY:    0.00000E+00"
    
    with open(path, "r", errors="ignore") as f:
        content = f.readlines()
    p, file = os.path.split(path)
    folder = os.path.split(p)[-1]

    column = f"{folder}_{file}"
    rates = []
    extracting = False    
    for i in content:
        if keyword in i and not extracting:
            extracting = True
        elif extracting and len(i) > 2:
            line = i.split()
            try:
                #* check to float conversion
                
                rates.append(float(line[1]))
            except ValueError:
                #* stick table header
                continue
        else:
            extracting = False
    df[column] = rates

    return df

def check_dir_files(
    ldir,
    path
):
    files = [
        os.path.join(path, i)
        for i in ldir if not os.path.isdir(
            os.path.join(
                cwd,
                i
            )) and ".FIN" in i
    ]
    return files

def folders_walk(
    folders: list
):
    df = pd.DataFrame()
    for folder in folders:
        ldir = os.listdir(folder)
        files = check_dir_files(ldir, folder)
        if not len(files) > 0:
            continue
        df = pd.concat(
            [
                df,
                files_walk(files)
            ],
            axis=1
        )
        
    return df

def files_walk(
    files: list
):
    df = None
    for file_path in files:
        df = extract_fin(file_path, df)
    
    return df



if __name__ == "__main__":
    args = parser.parse_args()
    cwd = os.getcwd()
    ldir = os.listdir()
    print(args)
    if args.files:
        files = check_dir_files(ldir, cwd)
        df = files_walk(files)
    else:
        folders = [
            os.path.join(cwd, i)
            for i in ldir if os.path.isdir(
                os.path.join(
                    cwd,
                    i
                ))
        ]

        df = folders_walk(folders)
    df.to_excel(
        f"{os.path.split(os.getcwd())[-1]}.xlsx"
        
    )
