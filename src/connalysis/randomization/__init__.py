""""Functions for getting parameters for modelling of random graphs."""

# TODO: Import all functions that the user might need
from .randomization import (run_ER,
                            run_SBM,
                            run_DD2,
                            run_DD3,
                            run_DD2_block_pre,
                            configuration_model)

__all__ = ["run_ER","run_SBM","run_DD2","run_DD3","run_DD2_block_pre", "configuration_model"]
