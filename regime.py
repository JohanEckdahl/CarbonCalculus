#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod



BorealFireResisting = { disturbance_interval: 100,
                        disturbance_probability: lambda t, C=None: 0.10 / (1 + np.exp(-(t - 100) / 20)),
                        disturbance: lambda t, C: C * 0.3,
                        growth: lambda t, C: 100 * (2 / np.pi) * np.arctan(0.05 * t),
                        decay: lambda t, C: C * 0.02,
                        }



@dataclass
class Regime(ABC, RegimePlotMixin):
    disturbance_interval = 100
    
    def disturbance_probability(self,t,C=None):
        return t >= self.disturbance_interval
    
    @abstractmethod
    def disturbance(self, t, C): return 0.0
    @abstractmethod
    def growth(self, t, C): return 0.0
    @abstractmethod
    def decay(self, t, C): return 0.0


    def reorganization_disturbance(self,t,C): return self.disturbance(t,C)
    def reorganization_growth(self,t,C): return self.growth(t,C)
    def reorganization_decay(self,t,C): return self.decay(t,C)   


class RegimeTemplate(Regime):

    def disturbance(self, t, C):
        rate = 1 * C
        return rate

    def growth(self, t, C):
        rate = 1 * C
        return rate

    def decay(self, t, C):
        rate = 1 * C
        return rate


    # Delete these three methods if wanting to reuse Regime class implementation
    def reorganization_disturbance(self,t,C):
        rate = 1 * C
        return rate
    def reorganization_growth(self,t,C):
        rate = 1 * C
        return rate
    def reorganization_decay(self,t,C):
        rate = 1 * C
        return rate



class BorealFireResisting(Regime):

    disturbance_interval = 100
    
    def disturbance_probability(self, t, C = None):
        # Parameters
        midpoint = self.disturbance_interval   # age where probability = 50% of max_prob
        spread = 20      # controls steepness
        max_prob = 0.10  # maximum annual disturbance probability
        return max_prob / (1 + np.exp(-(t - midpoint) / spread))

    def disturbance(self, t, C):
        rate = C * 0.3
        return rate
    
    def growth(self, t, C):
        L, k = 100, 0.05
        rate = L * (2 / np.pi) * np.arctan(k * t)
        return rate

    def decay(self, t, C):
        rate = C *  0.02
        return rate


        

class BorealBroadLeaf(Regime):
    
    def disturbance(self, t, C): return 0.0

    def growth(self, t, C): return 0.0

    def decay(self, t, C): return 0.0
