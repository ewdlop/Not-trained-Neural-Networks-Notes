"""
Statistical Neural Networks Package

This package implements neural networks based on statistical distributions
from quantum and classical mechanics:
- Fermi-Dirac Brain: Based on fermionic statistics
- Maxwell-Boltzmann Brain: Based on classical statistics
"""

__version__ = "0.1.0"
__author__ = "Not-trained Neural Networks Notes"

from .fermi_dirac_brain import FermiDiracBrain
from .maxwell_boltzmann_brain import MaxwellBoltzmannBrain

__all__ = ['FermiDiracBrain', 'MaxwellBoltzmannBrain']