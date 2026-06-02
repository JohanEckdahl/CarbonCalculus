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
    store_spinup  : bool = False
    probSpinup  : bool = False
   
    

    def __post_init__(self):
        # Set parameters, initialize matrices, etc.
        self.initialize()

        # Execute spinup
        self.spin_up()

    
    def initialize(self):
        # The regime shift must happen after the Spinup
        self.shifted_regime = None
        
        # 0 = old regime, 1 = reorganizing, 2 = new regime
        self.patch_state = None

        # Empty container to store yearly data
        self.legacy = []

        # Initialize Carbon matrix as nan values. Values will be filled in with initial carbon values when first disturbed.
        self.carbon = np.full(self.size, np.nan)

        # Spinup Set Up
        if self.probSpinup:
            # Initialize time-since-last-disturbance matrix. Random integers from 1 to disturbance interval value in matrix
            self.last_disturbance = np.random.randint(1, self.regime.disturbance_interval + 1, size=self.size)
        else: # Otherwise assign sequential ages
            if np.prod(self.size) % self.regime.disturbance_interval == 0:
                self.last_disturbance = np.random.permutation(np.arange(self.regime.disturbance_interval - np.prod(self.size), self.regime.disturbance_interval)).reshape(self.size) + 1
            else:
                raise ValueError("Matrix size must be a multiple of the disturbance interval for non-random spinup.")

        # Extract position of first-distrubed patch
        self.patch = np.unravel_index(np.argmax(self.last_disturbance), self.last_disturbance.shape)

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
            self.step(log=self.store_spinup)


    def log(self, t, C, G, L, D):
            fields = {"Carbon": C, "Growth": G, "Decay": L, "Disturbance": D, "Age": t}
            self.legacy.append(
                {f"Average {k}": np.nanmean(v) for k, v in fields.items()} |
                {f"Patch {k}": v[self.patch] for k, v in fields.items()}
            )


    # -------------------------
    # STEP
    # -------------------------
    def step(self, log=True):
        t, C, S, reg1, reg2 = self.last_disturbance, self.carbon, self.patch_state, self.regime, self.shifted_regime
        
        # Build disturbance probability
        if reg2 is None:
            prob = reg1.disturbance_probability(t, C)   
        else:
            prob = np.where(S == 0, reg1.disturbance_probability(t, C), reg2.disturbance_probability(t, C))

        #Create disturbance mask
        DMask = np.random.random(t.shape) < prob
        
        
        if S is not None:
            # Add to S
            S[DMask] += 1


        # --- Disturb (using S before any state advance) ---
        if reg2 is None:
            D = np.where(DMask, self.regime.disturb(t, C), 0)

        else:

            D = np.where(DMask,
            np.where(S == 1, reg2.disturb(t, C, reorg=True),
            np.where(S >= 2, reg2.disturb(t, C),
                            reg1.disturb(t, C))), 0)

            
        C -= D
        t[DMask] = 0
        
        #DMask = t >= self.regime.disturbance_interval    


        # --- Growth & Decay ---
        if reg2 is not None:
            G = np.where(S == 0, reg1.capture(t, C),
                np.where(S == 1, reg2.capture(t, C,reorg=True),
                                reg2.capture(t, C)))
            
            C += G

            L = np.where(S == 0, reg1.release(t, C),
                np.where(S == 1, reg2.release(t, C, reorg = True),
                                reg2.release(t, C)))
            
            C -= L
        else:
            G = reg1.capture(t, C)
            C += G
            L = reg1.release(t, C)
            C -= L

        # --- Log ---
        if log: self.log(t, C, G, L, D)

        # --- Advance time ---
        t += 1

        # --- Swap regime once all patches have transitioned ---
        if self.shifted_regime is not None and np.all(self.patch_state >= 2):
            self.regime = self.shifted_regime
            self.shifted_regime = None
            self.patch_state = None
            
    
    def regime_shift(self, regime):
        self.shifted_regime = regime
        self.patch_state = np.zeros(self.size, dtype=np.uint8)


    @timer
    def run(self, years):
        for _ in range(years):
            self.step()

    
    @property
    def df(self):
        df = pd.DataFrame(self.legacy)
        df.index.name = 'Year'
        return df





