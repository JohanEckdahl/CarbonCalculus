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

    def SCurve(self, t, midpoint, spread, peak):
        return peak / (1 + np.exp(-(t - midpoint) / spread))



class BorealFireResisting(Regime):

    disturbance_interval = 100

    def disturbance_probability(self, t, C):
        midpoint, spread, peak = self.disturbance_interval, 15 , 0.02
        return self.SCurve(t, midpoint, spread, peak)

    def disturbance(self, t, C):
        return C * 0.3

    def growth(self, t, C):
        midpoint, spread, peak = 25, 10, 1
        return self.SCurve(t, midpoint, spread, peak)

    def decay(self, t, C):
        midpoint = 60
        spread = 20
        peak = 1
        return self.SCurve(t, midpoint, spread, peak)


class BorealBroadLeaf(Regime):

    disturbance_interval = 100

    def disturbance_probability(self, t, C):
        midpoint = self.disturbance_interval
        spread = 15
        max_prob = 0.02
        return self.SCurve(t, midpoint, spread, max_prob)

    def disturbance(self, t, C):
        return C * 0.75

    def growth(self, t, C):
        midpoint = 15
        spread = 15
        peak = 1.5
        return self.SCurve(t, midpoint, spread, peak)

    def decay(self, t, C):
        midpoint = 60
        spread = 20
        peak = 1
        return self.SCurve(t, midpoint, spread, peak)

    def reorganization_disturbance(self, t, C):
        return C *0.9
    
    def reorganization_growth(self, t, C):
        midpoint = 50
        spread = 15
        peak = 1.5
        return self.SCurve(t, midpoint, spread, peak)
    
    def reorganization_decay(self, t, C):
        midpoint = 60
        spread = 20
        peak = 1
        return self.SCurve(t, midpoint, spread, peak)