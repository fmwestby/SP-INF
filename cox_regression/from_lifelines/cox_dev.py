import pandas as pd

"""
from fitters.cox_time_varying_fitter import CoxTimeVaryingFitter


base_df = pd.DataFrame([
    {'start': 0, 'var1': 0.1, 'var2': 0, 'stop': 5, 'id': 1, 'event': False},
    {'start': 5, 'var1': 0.1, 'var2': 1.4, 'stop': 9, 'id': 1, 'event': False},
    {'start': 9, 'var1': 0.1, 'var2': 1.2, 'stop': 10, 'id': 1, 'event': True},
    {'start': 0, 'var1': 0.5, 'var2': 0, 'stop': 5, 'id': 2, 'event': False},
    {'start': 5, 'var1': 0.5, 'var2': 1.6, 'stop': 12, 'id': 2, 'event': True},
])

ctv = CoxTimeVaryingFitter(penalizer=0.1)
ctv.fit(base_df, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=True)
ctv.print_summary()
#ctv.plot()
"""

from cox_func import CoxTimeVaryingFitter

base_df = pd.DataFrame([
    {'start': 0, 'var1': 0.1, 'var2': 0, 'stop': 5, 'id': 1, 'event': False},
    {'start': 5, 'var1': 0.1, 'var2': 1.4, 'stop': 9, 'id': 1, 'event': False},
    {'start': 9, 'var1': 0.1, 'var2': 1.2, 'stop': 10, 'id': 1, 'event': True},
    {'start': 0, 'var1': 0.5, 'var2': 0, 'stop': 5, 'id': 2, 'event': False},
    {'start': 5, 'var1': 0.5, 'var2': 1.6, 'stop': 12, 'id': 2, 'event': True},
])

ctv = CoxTimeVaryingFitter(penalizer=0.1)
ctv.fit(base_df, id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=True)
ctv.print_summary()