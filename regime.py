#!/usr/bin/env python3

import numpy as np
from .plotmixin import *
from dataclasses import dataclass

@dataclass
class Regime(RegimePlotMixin): 
    probDistInt : bool = False
    probDist    : bool = False
    probGrowth  : bool = False
    probDecay   : bool = False
    disturbance_interval: int = 100
    
    #Deterministic functions (used if probDist=False, probGrowth=False, or probDecay=False)
    def interval(self, t, C): return np.where(t >= self.disturbance_interval, 1.0, 0.0)
    def disturbance(self, t, C): return C * 0.5
    def growth(self, t, C): return C * 0.05
    def decay(self, t, C): return C * 0.01

    # Probabilistic functions (used if probDist=True, probGrowth=True, or probDecay=True)
    def prob_interval(self, t, C): return np.where(t >= self.disturbance_interval, 1.0, 0.0)
    def prob_disturbance(self, t, C): return self.disturbance(t,C)
    def prob_growth(self, t, C): return self.growth(t,C)
    def prob_decay(self, t, C): return self.decay(t,C)

    # Deterministic reorganization functions (used for patches in reorganization state, if shifted_regime is not None)
    def reorg_interval(self, t, C): return self.interval(t, C)
    def reorg_disturbance(self, t, C): return self.disturbance(t,C)
    def reorg_growth(self, t, C): return self.growth(t,C)
    def reorg_decay(self, t, C): return self.growth(t,C)

    # Probabilistic reorganization functions (used for patches in reorganization state, if shifted_regime is not None)
    def prob_reorg_interval(self, t, C): return self.prob_interval(t, C)
    def prob_reorg_disturbance(self, t, C): return self.disturbance(t,C)
    def prob_reorg_growth(self, t, C): return self.growth(t,C)
    def prob_reorg_decay(self, t, C): return self.growth(t,C)


    # Methods to be used in Landscape.step(), overwrite with caution
    def disturbance_probability(self, t, C, reorg=False):
        if reorg:
            return self.prob_reorg_interval(t, C) if self.probDist else self.reorg_interval(t, C)
        return self.prob_interval(t, C) if self.probDist else self.interval(t, C)

    def disturb(self, t, C, reorg=False):
        if reorg:
            return self.prob_reorg_disturbance(t, C) if self.probDist else self.reorg_disturbance(t, C)
        return self.prob_disturbance(t, C) if self.probDist else self.disturbance(t, C)

    def capture(self, t, C, reorg=False):
        if reorg:
            return self.prob_reorg_growth(t, C) if self.probGrowth else self.reorg_growth(t, C)
        return self.prob_growth(t, C) if self.probGrowth else self.growth(t, C)
    
    def release(self, t, C, reorg=False):
        if reorg:
            return self.prob_reorg_decay(t, C) if self.probDecay else self.reorg_decay(t, C)
        return self.prob_decay(t, C) if self.probDecay else self.decay(t, C)
    
    # Helpers
    def SCurve(self, t, midpoint, spread, peak):
        return peak / (1 + np.exp(-(t - midpoint) / spread))

class BorealFireResisting(Regime):

    def prob_interval(self, t, C):
        midpoint, spread, peak = self.disturbance_interval, 15 , 0.02
        return self.SCurve(t, midpoint, spread, peak)

    def disturbance(self, t, C):
        return C * 0.3

    def growth(self, t, C):
        midpoint, spread, peak = 25, 10, 1
        return self.SCurve(t, midpoint, spread, peak)

    def decay(self, t, C):
        midpoint, spread, peak = 60, 20, 1
        return self.SCurve(t, midpoint, spread, peak)

class BorealBroadLeaf(Regime):

    def prob_interval(self, t, C):
        midpoint, spread, max_prob = self.disturbance_interval, 15, 0.02
        return self.SCurve(t, midpoint, spread, max_prob)

    def disturbance(self, t, C):
        return C * 0.75

    def growth(self, t, C):
        midpoint, spread, peak = 15, 15, 1.5
        return self.SCurve(t, midpoint, spread, peak)

    def decay(self, t, C):
        midpoint, spread, peak = 60, 20, 1
        return self.SCurve(t, midpoint, spread, peak)

    def reorg_disturbance(self, t, C):
        return C *0.9
    
    def reorg_growth(self, t, C):
        midpoint, spread, peak = 50, 15, 1.5
        return self.SCurve(t, midpoint, spread, peak)
    
    def reorg_decay(self, t, C):
        midpoint, spread, peak = 60, 20, 1
        return self.SCurve(t, midpoint, spread, peak)