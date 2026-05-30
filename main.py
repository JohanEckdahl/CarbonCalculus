#!/usr/bin/env python3
import numpy as np
from .landscape import Landscape



BorealFireResisting = dict(
                            disturbance_interval     = 100,
                            disturbance_probability  = lambda self, t, C=None: 0.10 / (1 + np.exp(-(t - 100) / 20)),
                            disturbance              = lambda self, t, C: C * 0.3,
                            growth                   = lambda self, t, C: 100 * (2 / np.pi) * np.arctan(0.05 * t),
                            decay                    = lambda self, t, C: C * 0.02,
                        )



e = Landscape(BorealFireResisting, (10,10))
e.run(500)
#display(e.df)
