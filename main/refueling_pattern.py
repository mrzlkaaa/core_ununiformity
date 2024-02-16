from importlib.util import spec_from_file_location
from itertools import count
import random
import numpy as np
from . import init_core_parts

__all__ = [
    "pairwised_cells_swap",
    "find_pattern",
    "decode_pattern"

]

core_parts = init_core_parts()


#* call this method before asking to find a refueling pattern
def pairwised_cells_swap(
    before: np.ndarray,
    after: np.ndarray
):
    #* to prevent overwritting
    after = after.copy()
    
    if isinstance(before, list):
        before = np.asarray(before)

    if isinstance(after, list):
        after = np.asarray(after)

    for pair in core_parts["index"]["PAIRWISED_SYMM"]["PAIRS"]:
        pair = np.asarray(pair)

        before_cells = before[pair]
        after_cells = after[pair]
        
    
        if before_cells[0] == after_cells[0] and\
            before_cells[1] == after_cells[1]:
            #* it's ok they are already equal
            continue
        elif before_cells[0] == after_cells[1] and\
            before_cells[1] == after_cells[0]:
            #* swap pairwised cells in after core
            after[pair] = before_cells
    
    return after
         

def find_pattern(
    before: list,
    after: list,
    cycles: list | None = None
):

    '''
    #* Builds pattern to make core <after>
    #* from core <before>
    #* pattern is a list of indexes
    #* describes the sequence of permutations
    #* the first and last items in a pattern is 
    #* a bucket step (FA moves to / from bucket) 
    #* Returning pattern is always 2D array
    #* It is because pattern may include few refueling cycles
    #* By a cycle means the sequence bucket - bucket
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

    if not isinstance(before, list):
        before = list(before)

    if not isinstance(after, list):
        after = list(after)

    before = before.copy()
    after = after.copy()

    if cycles is None:
        cycles = []

    #* finds the cells that were refueled
    fresh_spent_fuel = list(set(before).difference(set(after)))
    # print("spent fuel cells: ", fresh_spent_fuel)
    
    for i in range(len(before)):
        if before[i] in fresh_spent_fuel:
            before[i] = 0.0

    before_mask = before.copy()
    after_mask = after.copy()

    #* masking to handle cells with same burnup (duplicates)
    taken_ind = []
    for i in range(len(before)):
        before_mask[i] = f"{before[i]}_{i}"
        
        #* index of before value in after list 
        lookup_after = after.index(before[i])
        if lookup_after in taken_ind:
            #* so duplicate is found
            #* iterate over list and take all duplicates
            #* then iterate over duplicates indexes
            #* and take unbound
            for dup_ind in [ind for ind, j in enumerate(after) if before[i] == j]:
                
                if dup_ind not in taken_ind:
                    # print(dup_ind, f"{before[i]}_{i}")
                    after_mask[dup_ind] = f"{before[i]}_{i}"
                    taken_ind.append(dup_ind)
                    break
        else:
            after_mask[lookup_after] = f"{before[i]}_{i}"
            taken_ind.append(lookup_after)
    
    diff = []
    path = []
    for i in range(len(before)):
        if before_mask[i] == after_mask[i]:
            continue
        diff.append(i)
    # print(diff)

    #* cell with index placed in bucket
    #* so in < before > there is an empty cell
    bucket = random.choice(diff)
    fa = before_mask[bucket]
    cell_taken_from = before_mask.index(fa)
    cell_to_place = after_mask.index(fa)

    path.append(cell_taken_from)
    path.append(cell_to_place)
    # print(f" the FA {fa} from cell {cell_taken_from} moved to a bucket")
    # print(f" the FA {fa} from bucket moved to a {cell_to_place}")

    chain = True
    counter = 1
    freezing_counter = 0
    while chain:

        if set(path) == set(diff):
            # print(set(path).difference(set(diff)))
            # print("all step are made")
            path.append(path[0])
            # print(path)
            cycles.append(path[::-1])
            # print(cycles)
            break
        elif cell_to_place == bucket:
            # print("all cells are allocated, so bucket loop is finished. But not all steps are completed")
            # print(path)
            before = np.asarray(before)
            before[path[:-1]] = np.asarray(after)[path[:-1]]
            before = list(before)
            # print(before_mask, after_mask)
            cycles.append(path[::-1])
            return find_pattern(
                before=before, 
                after=after,
                cycles=cycles)
        
        fa = before_mask[cell_to_place]
        cell_taken_from = before_mask.index(fa)
        cell_to_place = after_mask.index(fa)
        path.append(cell_to_place)
        # print(f" the FA {fa} from cell {cell_taken_from} about to move")
        # print(f" the FA {fa} moved to a {cell_to_place}")

        counter += 1

    return cycles

def decode_pattern(
    before:list,
    cycles: list
):
    decoded_cycles = {}
    cell_names = core_parts["default"]["ALL_CELLS"]

    counter = 0
    #* unpacking cycles
    for cn, cycle in enumerate(cycles):
        decoded_cycle = []
        for n, cell in enumerate(cycle):
            counter += 1
            if n == 0:
                decoded_cycle.append(
                    f'FA {before[cell]} from cell {cell_names[cell]} moves to a bucket'
                )
                continue
            elif n == len(cycle)-1:
                decoded_cycle.append(
                    f'FA {before[cell]} from bucket places in a cell {cell_names[cycle[n-1]]}'
                )
                continue
            
            decoded_cycle.append(
                f'FA {before[cell]} from cell {cell_names[cell]} moves to a cell {cell_names[cycle[n-1]]}'
            )
        decoded_cycles[cn] = decoded_cycle
    decoded_cycles["total_permutations"] = counter
            
    return decoded_cycles
