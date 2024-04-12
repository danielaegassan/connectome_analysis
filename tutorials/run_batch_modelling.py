# SPDX-FileCopyrightText: 2024 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Example python file needed to run model building from sbatch script.
#
# Author(s): C. Pokorny
# Last modified: 02/2023


import sys
from connalysis.modelling import modelling


def main_wrapper():
    """Main function wrapper for model building when called from command line.
       Command line arguments:
         adj_file (.npz), nrn_file (.h5/.feather), cfg_file (.json), N_split (optional), part_nr (optional)
    """

    # Parse inputs
    args = sys.argv[1:]
    if len(args) < 3:
        print(f'Usage: {__file__} <adj_file.npz> <nrn_file.h5/.feather> <cfg_file.json> [N_split] [part_nr]')
        sys.exit(2)

    adj_file = args[0]
    nrn_file = args[1]
    cfg_file = args[2]

    if len(args) > 3:
        N_split = int(args[3])
    else:
        N_split = None

    if len(args) > 4:
        part_nr = int(args[4])
        if part_nr >= 1 and part_nr <= N_split:
            part_idx = part_nr - 1
        elif part_nr == -1:
            part_idx = part_nr
        else:
            assert False, 'ERROR: Part number out of range (must be between 1 and N_split, or -1 for merging)!'
    else:
        part_idx = None

    # Run model building
    modelling.run_batch_model_building(adj_file, nrn_file, cfg_file, N_split, part_idx)


if __name__ == "__main__":
    main_wrapper()
