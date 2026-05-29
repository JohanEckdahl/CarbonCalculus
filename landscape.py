#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Landscape(LandscapePlotMixin):
    regime_class: type = Regime #Type should be child of abstract Regime class
    size: tuple[int, int] = (10, 10)
    initial_carbon: float = 100
    log_spinup: bool = False

    def __post_init__(self):
        #Instantiate initial regime class
        self.regime = self.regime_class()
        # The regime shift must happen after a certain amount of time has passed
        self.shifted_regime = None
        # Initialize Carbon matrix as nan values. Values will be filled in with initial carbon values when first disturbed.
        self.carbon = np.full(self.size, np.nan)
        # Initialize time-since-last-disturbance matrix. Random integers from 1 to disturbance interval value in matrix
        self.last_disturbance = np.random.randint(1, self.regime.disturbance_interval + 1, size=self.size)
        #self.last_disturbance = np.random.permutation(np.arange(self.regime.disturbance_interval - np.prod(self.size), self.regime.disturbance_interval)).reshape(self.size) +1
        # Extract position of first-distrubed patch
        self.patch = np.unravel_index(np.argmax(self.last_disturbance), self.last_disturbance.shape)
        # Empty container to store yearly data
        self.legacy = []
        # Execute spinup
        self.spin_up()

    # -------------------------
    # SPIN-UP
    # -------------------------
    def spin_up(self):
        # Loop until all nan values are replaced with numbers in the carbon matrix
        while np.isnan(self.carbon).any():
            # Get nan patchs ready for disturbance 
            mask = ((self.last_disturbance == self.regime.disturbance_interval) & np.isnan(self.carbon))
            # Apply initial carbon value to retrieved patchs
            self.carbon[mask] = self.initial_carbon
            # Execute yearly step
            self.step(log=self.log_spinup)

    # -------------------------
    # STEP
    # -------------------------
    def step(self, log = True):
        
        # Define equation matrices
        t, C = self.last_disturbance, self.carbon

        # Disturb
        D = np.where(np.random.random(t.shape) < self.regime.disturbance_probability(t), self.regime.disturbance(t, C), 0)
        C -= D
        t[D > 0] = 0
        
        # Grow
        G = self.regime.growth(t, C)
        C += G

        # Decay
        L = self.regime.decay(t, C)
        C -= L


        if log:
            # Store
            self.legacy.append({
                "Landscape Carbon": np.nansum(C),
                "Average Growth": np.nanmean(G),
                "Average Decay": np.nanmean(L),
                "Average Disturbance": np.nanmean(D),
                "Average Age": np.nanmean(t),
    
                "patch Carbon": C[self.patch],
                "patch Growth": G[self.patch],
                "patch Decay": L[self.patch],
                "patch Disturbance": D[self.patch],
                "patch Age": t[self.patch],
              })
    
        # Advance
        t += 1 

            
    def regime_shift(self,regime_class):
        self.regime = regime_class()


    def run(self, years):
        for _ in range(years):
            self.step()

    
    @property
    def df(self):
        df = pd.DataFrame(self.legacy)
        df.index.name = 'Year'
        return df





