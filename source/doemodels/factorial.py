from dataclasses import dataclass, field
from contextlib import contextmanager
from itertools import combinations
import pyDOE2 as doe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings(action="ignore", category=PerformanceWarning)
import statsmodels.formula.api as smf
import statsmodels.api as sm

@contextmanager
def plotting(size: tuple[int] = (15, 5)):
    try:
        plt.style.use("ggplot")
        plt.rcParams["font.weight"] = "normal"
        plt.rcParams["axes.labelweight"] = "bold"
        fig, axes = plt.subplots(figsize=size)
        yield fig, axes
    finally:
        plt.show()

@dataclass
class Factorial2k:

    columns_input: list[str] = field(default_factory=list)
    columns_multiplication: list[str] = field(default_factory=list, init=False, repr=False)
    DataFrame_Full: object = field(default_factory=list, init=False, repr=False)
    aliases: dict = field(default_factory=list, init=False, repr=False)
    aliases_filtered: dict = field(default_factory=list, init=False, repr=False)
    DataFrame_AntiAliased: object = field(default_factory=list, init=False, repr=False)
    effects: object = field(default_factory=list, init=False, repr=False)

    def multiplications(self):
        for i in range(1, len(self.columns_input) + 1):
            for j in combinations(self.columns_input, i):
                result = ''.join(j)
                self.columns_multiplication.append(result)

    def create_DataFrame(self):
        df = pd.DataFrame(doe.fracfact(" ".join(self.columns_input)), columns=self.columns_input, dtype=int)
        for i in self.columns_multiplication:
            mult_columns = list(i)
            df[i] = df[mult_columns].prod(axis=1)
        df.index.name = "Experiments"
        df.index += 1
        self.DataFrame_Full = df

    def fractional(self, principal_fraction: Union[list[str], str]):
        if isinstance(principal_fraction, str):
            self.DataFrame_Full = self.DataFrame_Full.query(f"{principal_fraction} == 1")
            self.DataFrame_Full = self.DataFrame_Full.drop(principal_fraction, axis=1)
        else:
            for i in principal_fraction:
                self.DataFrame_Full = self.DataFrame_Full.query(f"{i} == 1")
                self.DataFrame_Full = self.DataFrame_Full.drop(i, axis=1)
        self.DataFrame_Full.reset_index(drop=True, inplace=True)
        self.DataFrame_Full.index.name = "Experiments"
        self.DataFrame_Full.index += 1

        # Aliases
        alias_dict = {}
        for checking_column in self.DataFrame_Full.columns:
            aliases = []
            for column in self.DataFrame_Full.columns:
                if checking_column != column and (self.DataFrame_Full[checking_column] == self.DataFrame_Full[column]).all():
                    aliases.append(column)
            alias_dict[checking_column] = aliases
        self.aliases = alias_dict

        # Cleaning
        cols_to_drop = []
        cols_to_keep = []
        for key, values in self.aliases.items():
            for alias in values:
                if alias in self.DataFrame_Full.columns and alias not in cols_to_keep:
                    cols_to_drop.append(alias)
            cols_to_keep.append(key)
        self.DataFrame_AntiAliased = self.DataFrame_Full.drop(cols_to_drop, axis=1)
        self.aliases_filtered = {key: value for key, value in self.aliases.items() if key not in cols_to_drop}

    def effect(self, response: list[float], n: int = 1, graph: bool = False):
        if self.aliases:
            labels = []
            for key, value in self.aliases_filtered.items():
                aliases = value
                labels.append(f"{key} -> {aliases}")
            df = self.DataFrame_AntiAliased
            division = df["A"].value_counts().values[1]
            df["Y"] = response
            df_multiplied = df.drop(columns=['Y']).mul(df['Y'], axis=0)
            self.effects = pd.DataFrame(df_multiplied.sum(), columns=["Response"]).T / division / n
            min = self.effects.min().values.min()
            max = self.effects.max().values.max()
            self.effects.columns = labels
            if graph:
                if len(self.DataFrame_AntiAliased.columns) > 15:
                    with plotting() as (fig, ax):
                        ax.barh(y=self.effects.columns, width=self.effects.loc["Response"], color="turquoise", edgecolor="black", linewidth=0.5)
                        ax.axvline(0, color="black")
                        ax.set_xlabel("Factors")
                        ax.set_ylabel("Effect")
                        ax.set_title("Effects of factors")
                        ax.set_xticks(np.linspace(min, max, 10))
                else:
                    with plotting() as (fig, ax):
                        ax.bar(x=self.effects.columns, height=self.effects.loc["Response"], color="turquoise", edgecolor="black", linewidth=0.5)
                        ax.axhline(0, color="black")
                        ax.set_xlabel("Factors")
                        ax.set_ylabel("Effect")
                        ax.set_title("Effects of factors")
                        ax.set_yticks(np.linspace(min, max, 10))
        else:
            df = self.DataFrame_Full
            division = df["A"].value_counts().values[1]
            df["Y"] = response
            df_multiplied = df.drop(columns=['Y']).mul(df['Y'], axis=0)
            self.effects = pd.DataFrame(df_multiplied.sum(), columns=["Response"]).T / division / n
            min = self.effects.min().values.min()
            max = self.effects.max().values.max()
            if graph:
                if len(self.DataFrame_Full.columns) > 15:
                    with plotting() as (fig, ax):
                        ax.barh(y=self.effects.columns, width=self.effects.loc["Response"], color="turquoise", edgecolor="black", linewidth=0.5)
                        ax.axvline(0, color="black")
                        ax.set_xlabel("Factors")
                        ax.set_ylabel("Effect")
                        ax.set_title("Effects of factors")
                        ax.set_xticks(np.linspace(min, max, 10))
                else:
                    with plotting() as (fig, ax):
                        ax.bar(x=self.effects.columns, height=self.effects.loc["Response"], color="turquoise", edgecolor="black", linewidth=0.5)
                        ax.axhline(0, color="black")
                        ax.set_xlabel("Factors")
                        ax.set_ylabel("Effect")
                        ax.set_title("Effects of factors")
                        ax.set_yticks(np.linspace(min, max, 10))

    def pareto(self):
        df = self.effects.copy()
        df = df.T
        df["%"] = (np.absolute(df["Response"]) / np.absolute(df["Response"]).sum() * 100).round(2)
        df = df.sort_values(by="%", ascending=False)
        df["cum_sum_%"] = df["%"].cumsum().round(2)

        with plotting() as (fig, ax):
            # Effect
            min = np.min(df["%"])
            max = np.max(df["%"])
            ax.bar(df.index, df["%"], color="blue", edgecolor="black", linewidth=0.5, alpha=0.5)
            ax.tick_params(axis='y', colors='blue')
            ax.set_ylabel("Effect [%]", fontdict={"fontweight":"bold"}, color="blue")
            ax.set_yticks(np.linspace(min, max, 10).round(0))
            ax.set_title("Pareto chart")
            ax.set_xlabel("Factors", fontdict={"fontweight":"bold"})
            plt.grid(False)

            # Cumulative effect
            min = np.min(df["cum_sum_%"])
            max = np.max(df["cum_sum_%"])
            ax2 = plt.twinx()
            ax2.plot(df.index, df["cum_sum_%"], marker="o", linestyle="--", color="red")
            ax2.set_ylabel("Cumulative effect [%]", fontdict={"fontweight":"bold"}, color="red", rotation=-90, labelpad=15)
            ax2.tick_params(axis='y', colors='red')
            ax2.spines["right"].set_color('red')
            ax2.spines["left"].set_color('blue')
            ax2.set_yticks(np.linspace(min, max, 10).round(0))
            ax2.axhline(90)
            plt.grid(False)

        return df

    def __post_init__(self):
        self.multiplications()
        self.create_DataFrame()