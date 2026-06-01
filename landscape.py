#!/usr/bin/env python3

import numpy as np
import pandas as pd
from dataclasses import dataclass
from .plotmixin import *
from .regime import Regime
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__qualname__} was completed in {elapsed:.4f} seconds.")
        return result
    return wrapper


@dataclass
class Landscape(LandscapePlotMixin):
    regime: type = Regime() # Regime subclass
    size: tuple[int, int] = (10, 10)
    initial_carbon: float = 100
    log_spinup: bool = False
    probSpinup  : bool = False
    probDistInt : bool = False
    probDist    : bool = False
    probGrowth  : bool = False
    probDecay   : bool = False
    

    def __post_init__(self):
        # The regime shift must happen after a certain amount of time has passed
        self.shifted_regime = None
        # Initialize Carbon matrix as nan values. Values will be filled in with initial carbon values when first disturbed.
        self.carbon = np.full(self.size, np.nan)
        
        # Spinup Set Up
        if self.probSpinup:
            # Initialize time-since-last-disturbance matrix. Random integers from 1 to disturbance interval value in matrix
            self.last_disturbance = np.random.randint(1, self.regime.disturbance_interval + 1, size=self.size)
        else:
            if np.prod(self.size) % self.regime.disturbance_interval == 0:
                self.last_disturbance = np.random.permutation(np.arange(self.regime.disturbance_interval - np.prod(self.size), self.regime.disturbance_interval)).reshape(self.size) + 1
            else:
                raise ValueError("Matrix size must be a multiple of the disturbance interval for non-random spinup.")
       
        # Extract position of first-distrubed patch
        self.patch = np.unravel_index(np.argmax(self.last_disturbance), self.last_disturbance.shape)

        # 0 = old regime, 1 = reorganizing, 2 = new regime
        self.patch_state = np.zeros(self.size, dtype=np.uint8)

        # Empty container to store yearly data
        self.legacy = []
        
        # Execute spinup
        self.spin_up()

    # -------------------------
    # SPIN-UP
    # -------------------------
    @timer
    def spin_up(self):
        # Loop until all nan values are replaced with numbers in the carbon matrix
        while np.isnan(self.carbon).any():
            # Get nan patches ready for disturbance 
            mask = ((self.last_disturbance == self.regime.disturbance_interval) & np.isnan(self.carbon))
            # Apply initial carbon value to retrieved patches
            self.carbon[mask] = self.initial_carbon
            # Execute yearly step
            self.step(log=self.log_spinup)

    # -------------------------
    # STEP
    # -------------------------
    def step(self, log=True):
        t, C = self.last_disturbance, self.carbon
        S = self.patch_state
        new = self.shifted_regime

        # --- Disturbance mask ---
        if new is not None:
            prob = np.where(S == 0, self.regime.disturbance_probability(t, C),
                                    new.disturbance_probability(t, C))
        else:
            prob = self.regime.disturbance_probability(t, C)

        if self.probDistInt:
            DMask = np.random.random(t.shape) < prob
        else:
            DMask = t >= self.regime.disturbance_interval

        # --- Disturb (using S before any state advance) ---
        if new is not None:
            D = np.where(DMask,
                    np.where(S == 1, new.reorganization_disturbance(t, C),
                    np.where(S == 2, new.disturbance(t, C),
                                    self.regime.disturbance(t, C))), 0)
        else:
            D = np.where(DMask, self.regime.disturbance(t, C), 0)

        C -= D
        t[DMask] = 0

        # --- Advance state AFTER disturbance, BEFORE growth/decay ---
        if new is not None:
            S_prev = S.copy()
            S[DMask & (S == 0)] = 1   # just disturbed → reorganizing
            S[S_prev == 1] = 2        # was reorganizing → new regime

        # --- Growth & Decay ---
        if new is not None:
            G = np.where(S == 0, self.regime.growth(t, C),
                np.where(S == 1, new.reorganization_growth(t, C),
                                new.growth(t, C)))
            L = np.where(S == 0, self.regime.decay(t, C),
                np.where(S == 1, new.reorganization_decay(t, C),
                                new.decay(t, C)))
        else:
            G = self.regime.growth(t, C)
            L = self.regime.decay(t, C)

        C += G
        C -= L

        # --- Log ---
        if log:
            fields = {"Carbon": C, "Growth": G, "Decay": L, "Disturbance": D, "Age": t}
            self.legacy.append(
                {f"Average {k}": np.nanmean(v) for k, v in fields.items()} |
                {f"Patch {k}": v[self.patch] for k, v in fields.items()}
            )

        t += 1

        # --- Swap regime once all patches have transitioned ---
        if self.shifted_regime is not None and np.all(self.patch_state == 2):
            self.regime = self.shifted_regime
            self.shifted_regime = None
            self.patch_state = None
            
    
    def regime_shift(self, regime):
        self.shifted_regime = regime


    @timer
    def run(self, years):
        for _ in range(years):
            self.step()

    
    @property
    def df(self):
        df = pd.DataFrame(self.legacy)
        df.index.name = 'Year'
        return df





