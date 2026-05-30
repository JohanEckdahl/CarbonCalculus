#!/usr/bin/env python3

import numpy as np
from .plotmixin import *



class Regime(RegimePlotMixin): 
    disturbance_interval = 50
    def disturbance_probability(self, t, C=None): return np.where(t >= self.disturbance_interval, 1.0, 0.0)
    def disturbance(self, t, C): return C * 0.5
    def growth(self, t, C): return C * 0.05
    def decay(self, t, C): return C * 0.01

    def reorganization_disturbance(self, t, C): return self.disturbance(t,C)
    def reorganization_growth(self, t, C): return self.growth(t,C)
    def reorganization_decay(self, t, C): return self.growth(t,C)


class BorealFireResisting(Regime):

    disturbance_interval = 100

    def disturbance_probability(self, t, C):
        midpoint = 100
        spread = 20
        max_prob = 0.10
        #return np.where(t >= self.disturbance_interval, 1.0, 0.0)
        return max_prob / (1 + np.exp(-(t - midpoint) / spread))

    def disturbance(self, t, C): return C * 0.3

    def growth(self, t, C): return 100 * (2 / np.pi) * np.arctan(0.05 * t)

    def decay(self, t, C): return C * 0.02