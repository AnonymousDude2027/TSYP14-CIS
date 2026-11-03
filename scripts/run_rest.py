"""The remaining experiment steps, in priority order.

Ordered so that an interrupted run still leaves the tables that carry the report's
claims: false alarms and the standard profile first, the expensive evasion study
last.  The generalisation grid is run separately (scripts/run_grid.py) because it
is split across two processes.
"""
import sys
sys.path.insert(0, "src")
from tsfdi.eval import experiments as E

STEPS = ["standard_profile", "false_alarms", "diagnosis", "baselines", "ablation",
         "efficiency", "evasion", "evasion_curve"]
E.run_all(quick=False, steps=STEPS)
