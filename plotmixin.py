#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt




class RegimePlotMixin:
    def plot(self):
        ages = np.arange(0, 300)
        p = self.disturbance_probability(ages,None)
        plt.plot(ages, p)
        plt.xlabel("Age")
        plt.ylabel("Annual disturbance probability")
        plt.ylim(0, p.max() * 1.1)
        plt.grid(True)
        plt.show()


class LandscapePlotMixin:

    def plot(self):
        """The code in this method was auto-generated"""
        
        df = self.df

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # -------------------------
        # TOP: Patch dynamics
        # -------------------------
        ax1 = axes[0]

        ax1.plot(df.index, df["Patch Carbon"], label="Patch Carbon")

        ax1.set_ylabel("Patch Carbon")

        ax2 = ax1.twinx()
        ax2.plot(df.index, df["Patch Growth"], linestyle="--", label="Patch Growth", color ='green')
        ax2.plot(df.index, df["Patch Decay"], linestyle="--", label="Patch Decay", color='brown')

        ax2.set_ylabel("Patch Fluxes")
        
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        ax1.set_title("Patch-Level Dynamics")

        # combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # -------------------------
        # BOTTOM: Landscape totals
        # -------------------------
        ax3 = axes[1]

        ax3.plot(df.index, df["Average Carbon"], label="Average Carbon")
        ax3.set_ylabel("Landscape Carbon")
        ax3.set_ylim(bottom=0)

        ax4 = ax3.twinx()

        ax4.plot(
            df.index,
            df["Average Growth"],
            linestyle="--",
            label="Avg Growth",
            color="green"
        )

        ax4.plot(
            df.index,
            df["Average Decay"],
            linestyle="--",
            label="Avg Decay",
            color="brown"
        )

        # Disturbance rug marks
        dist = df["Average Disturbance"]
        mask = dist > 0

        ax4.vlines(
            df.index[mask],
            0,
            dist[mask],
            color="lightgray",
            linewidth=1,
            alpha=0.8,
            label="Avg Disturbance"
        )

        ax4.set_ylabel("Average Fluxes")
        ax4.set_ylim(bottom=0)

        ax3.set_title("Landscape-Level Dynamics")
        ax3.set_xlabel("Year")

        # combined legend
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper left")

        plt.tight_layout()
        plt.show()
