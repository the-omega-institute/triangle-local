"""Finite independent-source triangle models. No learned fit proves nonlocality."""
from .model import Model, Parameterization
from .optimize import fit
from .exact import rationalize, verify

__all__ = ["Model", "Parameterization", "fit", "rationalize", "verify"]
