"""Finite independent-source triangle models. No learned fit proves nonlocality."""
from .model import Model, Parameterization, apply_uniform_output_channel
from .optimize import fit
from .exact import (rationalize, verify, apply_uniform_output_channel as
                    apply_uniform_output_channel_exact, trim_zero_weight_symbols)

__all__ = ["Model", "Parameterization", "fit", "rationalize", "verify",
           "apply_uniform_output_channel", "apply_uniform_output_channel_exact",
           "trim_zero_weight_symbols"]
