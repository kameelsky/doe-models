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
        plt.style.use("default")

@dataclass
class Factorial2k:

    columns_input: list[str] = field(default_factory=list)
    columns_multiplication: list[str] = field(default_factory=list, init=False, repr=False)
    DataFrame: object = field(default_factory=list, init=False, repr=False)
    aliases: dict = field(default_factory=list, init=False, repr=False)
    aliases_filtered: dict = field(default_factory=list, init=False, repr=False)
    DataFrame_AntiAliased: object = field(default_factory=list, init=False, repr=False)
    effects_table: object = field(default_factory=list, init=False, repr=False)
    pareto_table: object = field(default_factory=object, init=False, repr=False)

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
        self.DataFrame = df

    def fractional(self, principal_fraction: Union[list[str], str]):
        if isinstance(principal_fraction, str):
            self.DataFrame = self.DataFrame.query(f"{principal_fraction} == 1")
            self.DataFrame = self.DataFrame.drop(principal_fraction, axis=1)
        else:
            for i in principal_fraction:
                self.DataFrame = self.DataFrame.query(f"{i} == 1")
                self.DataFrame = self.DataFrame.drop(i, axis=1)
        self.DataFrame.reset_index(drop=True, inplace=True)
        self.DataFrame.index.name = "Experiments"
        self.DataFrame.index += 1

        # Aliases
        alias_dict = {}
        for checking_column in self.DataFrame.columns:
            aliases = []
            for column in self.DataFrame.columns:
                if checking_column != column and (self.DataFrame[checking_column] == self.DataFrame[column]).all():
                    aliases.append(column)
            alias_dict[checking_column] = aliases
        self.aliases = alias_dict

        # Cleaning
        cols_to_drop = []
        cols_to_keep = []
        for key, values in self.aliases.items():
            for alias in values:
                if alias in self.DataFrame.columns and alias not in cols_to_keep:
                    cols_to_drop.append(alias)
            cols_to_keep.append(key)
        self.DataFrame_AntiAliased = self.DataFrame.drop(cols_to_drop, axis=1)
        self.aliases_filtered = {key: value for key, value in self.aliases.items() if key not in cols_to_drop}

    def effects(self, response: list[float], n: int = 1, graph: bool = False):
        if self.aliases:
            labels = []
            for key, value in self.aliases_filtered.items():
                aliases = value
                labels.append(f"{key} -> {aliases}")
            df = self.DataFrame_AntiAliased
            division = df["A"].value_counts().values[1]
            df["Y"] = response
            df_multiplied = df.drop(columns=['Y']).mul(df['Y'], axis=0)
            self.effects_table = pd.DataFrame(df_multiplied.sum(), columns=["Response"]).T / division / n
            min = self.effects_table.min().values.min()
            max = self.effects_table.max().values.max()
            self.effects_table.columns = labels
            if graph:
                if len(self.DataFrame_AntiAliased.columns) > 15:
                    with plotting() as (fig, ax):
                        ax.barh(y=self.effects_table.columns, width=self.effects_table.loc["Response"], color="turquoise", edgecolor="black", linewidth=1.5)
                        ax.axvline(0, color="black")
                        ax.set_xlabel("Effect")
                        ax.set_ylabel("Factors")
                        ax.set_title("Effects of factors")
                        ax.set_xticks(np.linspace(min, max, 10))
                else:
                    with plotting() as (fig, ax):
                        ax.bar(x=self.effects_table.columns, height=self.effects_table.loc["Response"], color="turquoise", edgecolor="black", linewidth=1.5)
                        ax.axhline(0, color="black")
                        ax.set_xlabel("Factors")
                        ax.set_ylabel("Effect")
                        ax.set_title("Effects of factors")
                        ax.set_yticks(np.linspace(min, max, 10))
        else:
            df = self.DataFrame
            division = df["A"].value_counts().values[1]
            df["Y"] = response
            df_multiplied = df.drop(columns=['Y']).mul(df['Y'], axis=0)
            self.effects_table = pd.DataFrame(df_multiplied.sum(), columns=["Response"]).T / division / n
            min = self.effects_table.min().values.min()
            max = self.effects_table.max().values.max()
            if graph:
                if len(self.DataFrame.columns) > 15:
                    with plotting() as (fig, ax):
                        ax.barh(y=self.effects_table.columns, width=self.effects_table.loc["Response"], color="turquoise", edgecolor="black", linewidth=1.5)
                        ax.axvline(0, color="black")
                        ax.set_xlabel("Effect")
                        ax.set_ylabel("Factors")
                        ax.set_title("Effects of factors")
                        ax.set_xticks(np.linspace(min, max, 10))
                else:
                    with plotting() as (fig, ax):
                        ax.bar(x=self.effects_table.columns, height=self.effects_table.loc["Response"], color="turquoise", edgecolor="black", linewidth=1.5)
                        ax.axhline(0, color="black")
                        ax.set_xlabel("Factors")
                        ax.set_ylabel("Effect")
                        ax.set_title("Effects of factors")
                        ax.set_yticks(np.linspace(min, max, 10))

    def pareto(self, graph: bool):
        df = self.effects_table.copy()
        df = df.T
        df["%"] = (np.absolute(df["Response"]) / np.absolute(df["Response"]).sum() * 100).round(2)
        df = df.sort_values(by="%", ascending=False)
        df["cum_sum_%"] = df["%"].cumsum().round(2)
        self.pareto_table = df

        if graph:
            with plotting() as (fig, ax):
                # Effect
                min = np.min(df["%"])
                max = np.max(df["%"])
                ax.bar(df.index, df["%"], color="turquoise", edgecolor="black", linewidth=1.5, alpha=0.5)
                ax.tick_params(axis='y', colors='teal')
                ax.set_ylabel("Effect [%]", fontdict={"fontweight":"bold"}, color="teal")
                ax.set_xlabel("Factors", fontdict={"fontweight":"bold"})
                ax.set_yticks(np.linspace(min, max, 10).round(0))
                ax.set_title("Pareto chart")
                plt.grid(False)

                # Cumulative effect
                ax2 = plt.twinx()
                ax2.plot(df.index, df["cum_sum_%"], marker="o", linestyle="--", color="red")
                ax2.set_ylabel("Cumulative effect [%]", fontdict={"fontweight":"bold"}, color="red", rotation=-90, labelpad=15)
                ax2.tick_params(axis='y', colors='red')
                ax2.spines["right"].set_color('red')
                ax2.spines["left"].set_color('teal')
                ax2.set_yticks(np.arange(0, 110, 10))
                ax2.grid(axis='y', color='red', alpha=0.1)

    def __post_init__(self):
        self.multiplications()
        self.create_DataFrame()