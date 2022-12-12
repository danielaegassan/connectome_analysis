""""Functions for getting parameters for modelling of random graphs."""

# TODO: Import all functions that the user might need
from .modelling import (
    conn_prob_2nd_order_model,
    conn_prob_2nd_order_pathway_model,
)

__all__ = [
    "conn_prob_2nd_order_model",
    "conn_prob_2nd_order_pathway_model",
]