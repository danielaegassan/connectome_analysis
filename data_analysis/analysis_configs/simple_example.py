from conntility import ConnectivityMatrix
from conntility.connectivity import LOCAL_CONNECTOME
import bluepy

circ_fn = "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC_WM"
circ = bluepy.Circuit(circ_fn)
analysis_config = "connection_prob_for_pathways.json" # File should exist along this example script

loader_config = {
    "loading": {
        "base_target": "central_column_4_region_700um", # Remove this line to load the entire model!
        "properties": ["x", "y", "z", "etype", "mtype", "layer", "synapse_class", "ss_flat_x", "ss_flat_y"] # Adapt as needed
    },
    "filtering":[ # For speed we load connectivity of only inhibitory neurons. Remove to load all (or set to "EXC" for excitatory)
        {
            "column": "synapse_class",
            "value": "INH"
        }
    ]
}

M = ConnectivityMatrix.from_bluepy(circ, loader_config, connectome=LOCAL_CONNECTOME) # connectome="intra_SSCX_midrange_wm" for long-range!
res = M.analyze(analysis_config)
p_200 = res["con_prob_within_200"]
p_all = res["overall_con_prob"]
print(p_200.unstack("idx-mtype_post"))
print(p_all.unstack("idx-mtype_post"))
