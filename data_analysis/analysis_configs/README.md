## Analysis configs
In this location I gather .json formatted analysis config files that turn the atomic analysis in this repository into more in-depth analyses that apply meaningful controls or break up analyses by m-type, etc.

For a documentation of this functionality:
https://bbpgitlab.epfl.ch/conn/structural/Connectome-utilities/-/blob/master/configuration_files.md#analyzing-connectivity-matrices


Here, I provide configs for six use cases:
  - cn_counts: Analyzes the number of common neighbors, i.e. whether in a connection matrix there is an unexpected large number of common
  neighbors compared to expectation (expectation based on ER with average in-/ out-degree).
  - cn_counts_per_mtype: Same, but the analysis is performed separately for (submatrices corresponding to) individual mtypes.
    - Required columns in neuron dataframe: "mtype".
  - cn_bias_for_unweighted_matrix: Calculates common neighbor connectivity biases. That is, how much the probability of a connection is
  affected by the number of common neighbors between the neurons. For use with unweighted (binary) matrices. Explicitly factors out the
  impact of distance on the number of common neighbors.
    - Required columns in neuron dataframe: "x", "y", "z" (for calcuating distances)
  - cn_bias_per_mtype_for_unweighted_matrix: Same, separately per mtype
    - Required columns in neuron dataframe: "x", "y", "z", "mtype"
  - cn_bias_for_weighted_matrix: Similar to the above, but includes analyses specific to weighted matrices. That is, how much the weight 
  of a connection depends on the number of common neighbors
    - Requied columns in neuron dataframe: "x", "y", "z"
  - cn_bias_per_mtype_for_weighted_matrix: Same, but per mtype
    - Requied columns in neuron dataframe: "x", "y", "z", "mtype"

Other use cases can be readily adapted from these.
TODO: Include control by shuffled matrices
