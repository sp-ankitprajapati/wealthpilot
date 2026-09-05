"""
WealthPilot — an AI agent for SME loan underwriting and credit research.

Built incrementally, week by week, as a mentor-guided project. See the
README for the current build status and the design decisions behind each
week's code — most of them came from a real bug caught by actually running
the code, not from following a tutorial.
"""

from .schemas import FinancialStatement
from .ratios import compute_ratios

__all__ = ["FinancialStatement", "compute_ratios"]
__version__ = "0.1.0"
