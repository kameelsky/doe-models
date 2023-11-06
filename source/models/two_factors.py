from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as smp
from scipy.optimize import curve_fit
from scipy.stats import f, shapiro, t
from statsmodels.api import qqplot
from tabulate import tabulate

sns.set_style("ticks")
sns.set_context("notebook")

@dataclass
class Model:
    ### Atributes
    Data: pd.DataFrame = field(repr=False) # Experimental DataFrame
    x_1: str = field(repr=False) # Name of independent variable X1
    x_2: str = field(repr=False) # Name of independent variable X2
    y_i: str = field(repr=False) # Name of dependent variable

    n: int = field(init=False, repr=False) # Number of observations
    k: int = field(init=False, repr=False) # Number of coefficients
    residuals: np.ndarray = field(init=False, repr=False) # Residuals
    shapiro: pd.DataFrame = field(init=False, repr=False) # p-value of Shapiro normality test for residues

    SST: float = field(init=False, repr=False) # Sum of squares total
    SSE: float = field(init=False, repr=False) # Sum of squares error
    SSR: float = field(init=False, repr=False) # Sum of squares regression

    r_squared: float = field(init=False, repr=False) # R squared
    r_squared_adj: float = field(init=False, repr=False) # Adjusted R squared

    anova: pd.DataFrame = field(init=False, repr=False) # Table of ANOVA regression analysis
    RMSE: float = field(init=False, repr=False) # Root Mean Standard Error

    coefficients_stats: pd.DataFrame = field(init=False, repr=False) # Table of coefficients analysis based on t Student distribution
    stats: pd.DataFrame = field(init=False, repr=False) # Table of R squared, Adjusted R squared and RMSE

    DataFramePredicted: pd.DataFrame = field(init=False, repr=False) # Data with additional column of predicted values of the response

    SSLF: float = field(init=False, repr=False) # Sum of squares lack of fit
    SSPE: float = field(init=False, repr=False) # Sum of squares pure error
    anova_error: pd.DataFrame = field(init=False, repr=False) # Table of ANOVA regression analysis of error

    def _n(self) -> n: 
        self.n = len(self.Data[self.y_i])
    
    def _k(self) -> k:
        self.k = len(self.coefficients_symbols)

    def _residuals(self) -> residuals:
        self.residuals = self.Data[self.y_i] - self.y_hat

    def _shapiro(self) -> shapiro:
        
        pvalue = shapiro((self.residuals))[1]
            
        self.shapiro = pd.DataFrame({
            "Shapiro":[pvalue]
        }, index=["p-value"])

    def _SST(self) -> SST:
        diff = self.Data[self.y_i] - np.mean(self.Data[self.y_i])
        self.SST = np.sum(diff**2)

    def _SSE(self) -> SSE:
        diff = self.Data[self.y_i] - self.y_hat
        self.SSE = np.sum(diff**2)
    
    def _SSR(self) -> SSR:
        diff  = self.y_hat - np.mean(self.Data[self.y_i])
        self.SSR = np.sum(diff**2)

    def _r_squared(self) -> (r_squared, r_squared_adj):
        self.r_squared = self.SSR / self.SST
        self.r_squared_adj =  1 - (1 - self.r_squared)*(self.n - 1)/(self.n - self.k)

    def _anova(self) -> (anova, RMSE):

        # Degrees of freedom
        df_total = self.n - 1
        df_regression = self.k - 1
        df_error = self.n - self.k

        # Mean squared
        msr = self.SSR / df_regression
        mse = self.SSE / df_error
        self.RMSE = np.sqrt(mse)

        # F-statistic
        f_stat = msr / mse

        # p-value
        pvalue = 1 - f.cdf(f_stat, df_regression, df_error)

        # Significance
        if pvalue <= 0.0001:
            significance = "****"
        elif pvalue <= 0.001:
            significance = "***"
        elif pvalue <= 0.01:
            significance = "**"
        elif pvalue <= 0.05:
            significance = "*"
        else:
            significance = "ns"

        # DataFrame of ANOVA
        self.anova = pd.DataFrame({
            "df": [df_regression, df_error, df_total],
            "SS": [self.SSR, self.SSE, self.SST],
            "MS": [msr, mse, ""],
            "F": [f_stat, "", ""],
            "p-value": [pvalue, "", ""],
            "Significance": [significance, "", ""]
        }, index=["Regression", "Error", "Total"])

    def _error(self) -> (SSLF, SSPE, anova_error):

        if True in self.Data.duplicated(subset=[self.x_1, self.x_2]).values:

            grp = self.DataFramePredicted.groupby(by=[self.x_1, self.x_2]).mean()
            df = self.DataFramePredicted.copy()

            df[f"{self.y_i} mean"] = [round(grp.loc[df[self.x_1][i]].loc[df[self.x_2][i]][self.y_i], ndigits=2) for i in df.index]
            df["Lack of fit"] = df[f"{self.y_i} mean"] - self.y_hat
            df["Pure Error"] = df[self.y_i] - df[f"{self.y_i} mean"]

            self.SSLF = round(np.sum((df["Lack of fit"])**2), ndigits=2)
            self.SSPE = round(np.sum((df["Pure Error"])**2), ndigits=2)
            
            # Degrees of freedom
            unique = df.loc[-df.duplicated(subset=[self.x_1, self.x_2])].shape[0]
            total = df.shape[0]
            df_error = self.n - self.k
            df_lack_of_fit = df_error - total + unique
            df_pure_error = total - unique
            
            # Mean squared
            mslf = self.SSLF / df_lack_of_fit
            mspe = self.SSPE / df_pure_error

            # f-statistic
            f_stat = mslf / mspe

            # p-value
            pvalue = 1 - f.cdf(f_stat, df_lack_of_fit, df_pure_error)
            
            # Significance
            if pvalue <= 0.0001:
                significance = "****"
            elif pvalue <= 0.001:
                significance = "***"
            elif pvalue <= 0.01:
                significance = "**"
            elif pvalue <= 0.05:
                significance = "*"
            else:
                significance = "ns"


        # DataFrame of ANOVA
        self.anova_error = pd.DataFrame({
            "df": [df_lack_of_fit, df_pure_error, df_error],
            "SS": [self.SSLF, self.SSPE, self.SSE],
            "MS": [mslf, mspe, ""],
            "F": [f_stat, "", ""],
            "p-value": [pvalue, "", ""],
            "Significance": [significance, "", ""]
        }, index=["Lack of fit", "Pure error", "Error"])

    def _coefficients_pvalues(self) -> coefficients_stats:

        # Errors from the covariance matrix
        variances = np.diagonal(self.covariance)
        errors = np.sqrt(variances)

        # Degrees of freedom
        df = self.n - self.k

        # t-values
        tvalues = self.coefficients / errors

        # p-values
        pvalues = (1 - t.cdf(np.abs(tvalues), df)) * 2

        # Significance
        significance_list = []
        for pvalue in pvalues:
            if pvalue <= 0.0001:
                significance_list.append("****")
            elif pvalue <= 0.001:
                significance_list.append("***")
            elif pvalue <= 0.01:
                significance_list.append("**")
            elif pvalue <= 0.05:
                significance_list.append("*")
            else:
                significance_list.append("ns")

        self.coefficients_stats = pd.DataFrame({
            coefficient: [value, error, t_stat, pvalue, sig] 
            for coefficient, value, error, t_stat, pvalue, sig in zip(
            self.coefficients_symbols,
            self.coefficients,
            errors,
            tvalues,
            pvalues,
            significance_list)
        }, index=["Value", "Standard error", "t Statistic", "p-value", "Significance"])

    def _stats(self) -> stats: # Stats of the model
        self.stats = pd.DataFrame({
            "Model":[self.r_squared, self.r_squared_adj, self.RMSE]
        }, index=["R squared", "R squared adjusted", "RMSE"])

    def _DataFramePredicted(self) -> DataFramePredicted: # Adds a column with predicted valueas to DataFrame

            # Calculation of predicted values
            df = self.Data.copy()

            predicted = self.y_hat
            predicted = [round(i, ndigits=2) for i in predicted]
            df[self.y_i + "^"] = predicted

            self.DataFramePredicted = df

    def summary(self) -> None:

        print("ANOVA") 
        print(tabulate(self.anova, headers=self.anova.columns, tablefmt="fancy_grid"))
        print("\nCoefficients") 
        print(tabulate(self.coefficients_stats, headers=self.coefficients_stats.columns, tablefmt="fancy_grid"))
        print("\nStatistics") 
        print(tabulate(self.stats, headers=self.stats.columns, tablefmt="fancy_grid"))

    # Plots

    def replicate_plot(self) -> None: # Returns plot of experiments and responses.

        replicate_index = self.Data.index.to_list()

        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        stem = ax.stem(replicate_index, self.Data[self.y_i], linefmt="-", basefmt = "k:")
        ax.tick_params(axis="both", width=1, labelsize=10, pad=8)
        ax.set_xticks(np.arange(min(replicate_index), max(replicate_index) + 1, step=1))
        stem[1].set_linewidth(1)
        stem[1].set_linestyles("solid")
        ax.set_xlabel('Replicate index', fontsize=13)
        ax.set_ylabel(self.y_i, fontsize=13)
        plt.show()

    def qq_plot(self) -> None: # Plots a Q-Q plot of residuals.
        fig, ax = plt.subplots(figsize=(5, 3))
        fig = qqplot(data=(self.residuals), 
                        ax=ax)
        ax.tick_params(axis="both", width=1, labelsize=10, pad=4)
        ax.set_title('QQ-plot of Residuals', fontsize=14)
        ax.set_xlabel("Theoretical quantiles", fontsize=14)
        ax.set_ylabel("Sample quantiles", fontsize=14)
        ax.grid(True)
        plt.show()

    def residuals_plot(self) -> None: # Plot residuals vs variables and index.

        replicate_index = self.Data.index.to_list()
        
        fig, axes = plt.subplots(1, 3, figsize=(12,3), sharey=True)
        fig.suptitle("Residuals plots", fontsize=14)

        ax = axes[0]
        ax.scatter(replicate_index, self.residuals)
        ax.set_xlabel('Replicate index', fontsize=13)
        ax.set_ylabel('Residuals', fontsize=13)

        ax = axes[1]
        ax.scatter(self.Data[self.x_1], self.residuals)
        ax.set_xlabel(self.x_1, fontsize=13)
        ax.tick_params(left=False)
        for i, txt in enumerate(self.Data[self.x_1], start=1):
            ax.text(self.Data[self.x_1][i], self.residuals[i], i, fontsize=10, color="red")

        ax = axes[2]
        ax.scatter(self.Data[self.x_2], self.residuals)
        ax.set_xlabel(self.x_2, fontsize=13)
        ax.tick_params(left=False)
        for i, txt in enumerate(self.Data[self.x_2], start=1):
            ax.text(self.Data[self.x_2][i], self.residuals[i], i, fontsize=10, color="red")
        
        for ax in axes:
            ax.tick_params(axis="both", width=1, labelsize=10, pad=8)
            ax.axhline(y=0, linestyle="--")
            ax.grid(True)

        plt.show()

    def rsp(self, figure_size: tuple = (7,5), dpi: int = 300, step_x: float = 1, step_y: float = 1,
            contours_number: int = 15, contour_color: str = "black", contour_font_size: float = 10, 
            color_map: str = "RdYlGn") -> None: # Returns RSP
    
        plt.figure(figsize=figure_size, dpi=dpi)

        cf = plt.contourf(self.x1_mesh, self.x2_mesh, self.y_mesh, cmap=color_map, levels=contours_number)
        cp = plt.contour(self.x1_mesh, self.x2_mesh, self.y_mesh, colors=contour_color, levels=contours_number)
        plt.colorbar(cf)
        plt.clabel(cp, fontsize=contour_font_size)
        
        plt.xlabel(self.x_1)
        plt.xticks(ticks=(np.arange(min(self.Data[self.x_1]), max(self.Data[self.x_1] + step_x), step_x)))
        
        plt.yticks(ticks=(np.arange(min(self.Data[self.x_2]), max(self.Data[self.x_2] + step_y), step_y)))
        plt.ylabel(self.x_2)

        plt.title(f"{self.y_i}({self.x_1}, {self.x_2})")
        plt.show()

    def rsp3d(self, figure_size: tuple = (7,5), dpi: int = 300, step_x: float = 1, step_y: float = 1,
              color_map: str = "RdYlGn") -> None: # Returns RSP in 3D

        plt.figure(figsize=figure_size, dpi=dpi)

        ax = plt.axes(projection="3d")
        ax.plot_surface(self.x1_mesh, self.x2_mesh, self.y_mesh, cmap=color_map, linewidth=0)
        
        ax.set_xlabel(self.x_1)
        ax.set_xticks(ticks=(np.arange(min(self.Data[self.x_1]), max(self.Data[self.x_1] + step_x), step_x)))
        
        ax.set_yticks(ticks=(np.arange(min(self.Data[self.x_2]), max(self.Data[self.x_2] + step_y), step_y)))
        ax.set_ylabel(self.x_2)

        ax.set_title(f"{self.y_i}({self.x_1}, {self.x_2})")
        plt.show()

    def interaction_plot(self) -> None:

        fig, axes = plt.subplots(1,2, figsize=(11,5))

        ax = axes[0]
        sns.pointplot(data=self.Data, x=self.x_1, y=self.y_i, hue=self.x_2, capsize=0.1, palette='colorblind', errorbar="sd", errwidth=1, ax=ax)

        ax = axes[1]
        sns.pointplot(data=self.Data, x=self.x_2, y=self.y_i, hue=self.x_1, capsize=0.1, palette='colorblind', errorbar="sd", errwidth=1, ax=ax)

        for ax in axes:
            pass

        plt.show()

@dataclass
class LIQ(Model):

    equation: smp.core.add.Add = field(init=False, repr=False) # Equtation
    dfdx1: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X1
    dfdx2: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X2
    symbols: list[smp.core.symbol.Symbol] = field(init=False, repr=False) # All symbols in equation
    coefficients_symbols: list[smp.core.symbol.Symbol] = field(init=False, repr=False) # Symbols of coefficients in equation
    coefficients: np.ndarray = field(init=False, repr=False) # Coefficients values after curve fitting
    covariance: np.ndarray = field(init=False, repr=False) # Covariance matrix after curve fitting
    equation_subs: smp.core.add.Add = field(init=False, repr=False) # Equtation with coefficients substituted after curve fitting
    dfdx1_subs: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X1
    dfdx2_subs: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X2
    y_hat: pd.Series = field(init=False, repr=False) # Predicted values
    x1_mesh: np.ndarray = field(init=False, repr=False) # X1 Meshgrid 
    x2_mesh: np.ndarray = field(init=False, repr=False) # X2 Meshgrid 
    y_mesh: np.ndarray = field(init=False, repr=False) # Predicted values (meshgrid)
    substitutions: list[tuple] = field(init=False, repr=False) # Substitutions
    linear_equation: tuple = field(init=False, repr=False)

    def _model(self) -> (equation, symbols, coefficients_symbols, coefficients, dfdx1, dfdx2, covariance, equation_subs, dfdx1_subs, dfdx2_subs, y_hat, x1_mesh, x2_mesh, y_mesh, substitutions):

        # Symbols and equtation
        global b0, b1, b2, b12, b11, b22, x1, x2, function
        b0, b1, b2, b12, b11, b22, x1, x2 = smp.symbols("beta_0, beta_1, beta_2, beta_12, beta_11, beta_22, x_1, x_2")
        self.equation = b0 + b1 * x1 + b2 * x2 + b11 * x1**2 + b22 * x2**2 + b12 * x1 * x2
        self.symbols = b0, b1, b2, b12, b11, b22, x1, x2
        self.coefficients_symbols = b0, b1, b2, b12, b11, b22

        # Partial derivatives
        self.dfdx1 = smp.diff(self.equation, x1)
        self.dfdx2 = smp.diff(self.equation, x2)

        # Curve fit
        function = smp.lambdify(((x1, x2), b0, b1, b2, b12, b11, b22), self.equation)
        self.coefficients, self.covariance = curve_fit(f=function,
                                                       xdata=(self.Data[self.x_1], self.Data[self.x_2]),
                                                       ydata=self.Data[self.y_i],
                                                       p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), method="lm")
        
        # Substitute coefficients using a loop
        self.substitutions = [(symbol, coefficient) for symbol, coefficient in zip(self.coefficients_symbols, self.coefficients)]
        self.equation_subs = self.equation.subs(self.substitutions)
        self.dfdx1_subs = self.dfdx1.subs(self.substitutions)
        self.dfdx2_subs = self.dfdx2.subs(self.substitutions)

        # Meshgrid
        self.y_hat = function((self.Data[self.x_1], self.Data[self.x_2]), *self.coefficients)
        self.x1_mesh, self.x2_mesh = np.meshgrid(np.linspace(min(self.Data[self.x_1]), max(self.Data[self.x_1]), 100), 
                                                 np.linspace(min(self.Data[self.x_2]), max(self.Data[self.x_2]), 100))
        self.y_mesh = function((self.x1_mesh, self.x2_mesh), *self.coefficients)

    def solve(self) -> linear_equation:

        # Linear algebra and matrices
        variables = (x1, x2)
        A, B = smp.linear_eq_to_matrix([self.dfdx1, self.dfdx2], variables)
        self.linear_equation = (A, B)

        solution = (A.inv() @ B).subs(self.substitutions)
        eigenvalues = [round(eigenvalue, ndigits=2) for eigenvalue, multiplicity in dict(A.subs(self.substitutions).eigenvals()).items()]
        critical_value = function((solution[0], solution[1]), *self.coefficients)

        d = {self.x_1: round(solution[0], ndigits=2), self.x_2: round(solution[1], ndigits=2), "Critical value": round(critical_value, ndigits=2), "Hessian matrix eigenvalues": eigenvalues}
        for key, value in d.items():
            print(f"{key}: {value}")

    ### Specific Plots

    def constant_plot(self, value: tuple[str, float], loc: tuple[float, float] = (0.05, 0.9)) -> None:

        if value[0] == self.x_1:

            # Substitute X1 with custom value in the equtation and lambdify it using X2 as arguments
            f = self.equation_subs.subs(x1, value[1])
            f_lam = smp.lambdify(x2, f)

            # Create both X2 and Y domains
            x = np.linspace(min(self.Data[self.x_2]), max(self.Data[self.x_2]), 100)
            y = f_lam(x)

            # Solve the differential equtation
            f_dfdx = smp.diff(f, x2)
            solution = smp.solve(f_dfdx)[0]
            _max = f.subs(x2, solution)

            # Plot
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(self.x_2)
            ax.set_ylabel(self.y_i)
            text=f"{self.x_1}: {value[1]}\n{self.x_2}: {solution:.2f}\n{self.y_i}: {_max:.1f}"
            ax.text(loc[0], loc[1], text, fontsize=8, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"))
            ax.plot(x, y)
            plt.show()

        if value[0] == self.x_2:

            # Substitute X2 with custom value in the equtation and lambdify it using X1 as arguments
            f = self.equation_subs.subs(x2, value[1])
            f_lam = smp.lambdify(x1, f)
            
            # Create both X1 and Y domains:
            x = np.linspace(min(self.Data[self.x_1]), max(self.Data[self.x_1]), 100)
            y = f_lam(x)

            # Solve the differential equtation
            f_dfdx = smp.diff(f, x1)
            solution = smp.solve(f_dfdx)[0]
            _max = f.subs(x1, solution)

            # Plot
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(self.x_1)
            ax.set_ylabel(self.y_i)
            text=f"{self.x_2}: {value[1]}\n{self.x_1}: {solution:.2f}\n{self.y_i}: {_max:.1f}"
            ax.text(loc[0], loc[1], text, fontsize=8, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"))
            ax.plot(x, y)
            plt.show()

    # Post init

    def __post_init__(self) -> None:
        self._model()
        self._n(), self._k()
        self._residuals(), self._shapiro()
        self._SST(), self._SSE(), self._SSR()
        self._r_squared()
        self._anova()
        self._coefficients_pvalues()
        self._stats()
        self._DataFramePredicted()
        try:
            self._error()
        except UnboundLocalError:
            pass

@dataclass
class LIT(Model):
    
    equation: smp.core.add.Add = field(init=False, repr=False) # Equtation
    dfdx1: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X1
    dfdx2: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X2
    symbols: list[smp.core.symbol.Symbol] = field(init=False, repr=False) # All symbols in equation
    coefficients_symbols: list[smp.core.symbol.Symbol] = field(init=False, repr=False) # Symbols of coefficients in equation
    coefficients: np.ndarray = field(init=False, repr=False) # Coefficients values after curve fitting
    covariance: np.ndarray = field(init=False, repr=False) # Covariance matrix after curve fitting
    equation_subs: smp.core.add.Add = field(init=False, repr=False) # Equtation with coefficients substituted after curve fitting
    dfdx1_subs: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X1
    dfdx2_subs: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X2
    y_hat: pd.Series = field(init=False, repr=False) # Predicted values
    x1_mesh: np.ndarray = field(init=False, repr=False) # X1 Meshgrid 
    x2_mesh: np.ndarray = field(init=False, repr=False) # X2 Meshgrid 
    y_mesh: np.ndarray = field(init=False, repr=False) # Predicted values (meshgrid)
    substitutions: list[tuple] = field(init=False, repr=False) # Substitutions

    def _model(self) -> (equation, symbols, coefficients_symbols, coefficients, dfdx1, dfdx2, covariance, equation_subs, y_hat, x1_mesh, x2_mesh, y_mesh, substitutions):

        # Symbols and equtation
        global b0, b1, b2, b12, x1, x2, function
        b0, b1, b2, b12, x1, x2 = smp.symbols("beta_0, beta_1, beta_2, beta_12, x_1, x_2")
        self.equation = b0 + b1 * x1 + b2 * x2 + b12 * x1 * x2
        self.symbols = b0, b1, b2, b12, x1, x2
        self.coefficients_symbols = b0, b1, b2, b12

        # Partial derivatives
        self.dfdx1 = smp.diff(self.equation, x1)
        self.dfdx2 = smp.diff(self.equation, x2)

        # Curve fit
        function = smp.lambdify(((x1, x2), b0, b1, b2, b12), self.equation)
        self.coefficients, self.covariance = curve_fit(f=function,
                                                       xdata=(self.Data[self.x_1], self.Data[self.x_2]),
                                                       ydata=self.Data[self.y_i],
                                                       p0=(1.0, 1.0, 1.0, 1.0), method="lm")
        
        # Substitute coefficients using a loop
        self.substitutions = [(symbol, coefficient) for symbol, coefficient in zip(self.coefficients_symbols, self.coefficients)]
        self.equation_subs = self.equation.subs(self.substitutions)
        self.dfdx1_subs = self.dfdx1.subs(self.substitutions)
        self.dfdx2_subs = self.dfdx2.subs(self.substitutions)

        # Meshgrid
        self.y_hat = function((self.Data[self.x_1], self.Data[self.x_2]), *self.coefficients)
        self.x1_mesh, self.x2_mesh = np.meshgrid(np.linspace(min(self.Data[self.x_1]), max(self.Data[self.x_1]), 100), 
                                                 np.linspace(min(self.Data[self.x_2]), max(self.Data[self.x_2]), 100))
        self.y_mesh = function((self.x1_mesh, self.x2_mesh), *self.coefficients)

    ### Specific Plots

    def constant_plot(self, value: tuple[str, float], loc: tuple[float, float] = (0.05, 0.9)) -> None:

        if value[0] == self.x_1:

            # Substitute X1 with custom value in the equtation and lambdify it using X2 as arguments
            f = self.equation_subs.subs(x1, value[1])
            f_lam = smp.lambdify(x2, f)

            # Create both X2 and Y domains
            x = np.linspace(min(self.Data[self.x_2]), max(self.Data[self.x_2]), 100)
            y = f_lam(x)

            # Plot
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(self.x_2)
            ax.set_ylabel(self.y_i)
            text=f"{self.x_1}: {value[1]}"
            ax.text(loc[0], loc[1], text, fontsize=8, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"))
            ax.plot(x, y)
            plt.show()

        if value[0] == self.x_2:

            # Substitute X2 with custom value in the equtation and lambdify it using X1 as arguments
            f = self.equation_subs.subs(x2, value[1])
            f_lam = smp.lambdify(x1, f)
            
            # Create both X1 and Y domains:
            x = np.linspace(min(self.Data[self.x_1]), max(self.Data[self.x_1]), 100)
            y = f_lam(x)

            # Plot
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(self.x_1)
            ax.set_ylabel(self.y_i)
            text=f"{self.x_2}: {value[1]}"
            ax.text(loc[0], loc[1], text, fontsize=8, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"))
            ax.plot(x, y)
            plt.show()
    
    # Post init

    def __post_init__(self) -> None:
        self._model()
        self._n(), self._k()
        self._residuals(), self._shapiro()
        self._SST(), self._SSE(), self._SSR()
        self._r_squared()
        self._anova()
        self._coefficients_pvalues()
        self._stats()
        self._DataFramePredicted()
        try:
            self._error()
        except UnboundLocalError:
            pass

@dataclass
class LIN(Model):
    
    equation: smp.core.add.Add = field(init=False, repr=False) # Equtation
    dfdx1: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X1
    dfdx2: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X2
    symbols: list[smp.core.symbol.Symbol] = field(init=False, repr=False) # All symbols in equation
    coefficients_symbols: list[smp.core.symbol.Symbol] = field(init=False, repr=False) # Symbols of coefficients in equation
    coefficients: np.ndarray = field(init=False, repr=False) # Coefficients values after curve fitting
    covariance: np.ndarray = field(init=False, repr=False) # Covariance matrix after curve fitting
    equation_subs: smp.core.add.Add = field(init=False, repr=False) # Equtation with coefficients substituted after curve fitting
    dfdx1_subs: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X1
    dfdx2_subs: smp.core.add.Add = field(init=False, repr=False) # Partial derivative X2
    y_hat: pd.Series = field(init=False, repr=False) # Predicted values
    x1_mesh: np.ndarray = field(init=False, repr=False) # X1 Meshgrid 
    x2_mesh: np.ndarray = field(init=False, repr=False) # X2 Meshgrid 
    y_mesh: np.ndarray = field(init=False, repr=False) # Predicted values (meshgrid)
    substitutions: list[tuple] = field(init=False, repr=False) # Substitutions

    def _model(self) -> (equation, symbols, coefficients_symbols, coefficients, dfdx1, dfdx2, covariance, equation_subs, y_hat, x1_mesh, x2_mesh, y_mesh, substitutions):

        # Symbols and equtation
        global b0, b1, b2, x1, x2, function
        b0, b1, b2, x1, x2 = smp.symbols("beta_0, beta_1, beta_2, x_1, x_2")
        self.equation = b0 + b1 * x1 + b2 * x2
        self.symbols = b0, b1, b2, x1, x2
        self.coefficients_symbols = b0, b1, b2

        # Partial derivatives
        self.dfdx1 = smp.diff(self.equation, x1)
        self.dfdx2 = smp.diff(self.equation, x2)

        # Curve fit
        function = smp.lambdify(((x1, x2), b0, b1, b2), self.equation)
        self.coefficients, self.covariance = curve_fit(f=function,
                                                       xdata=(self.Data[self.x_1], self.Data[self.x_2]),
                                                       ydata=self.Data[self.y_i],
                                                       p0=(1.0, 1.0, 1.0), method="lm")
        
        # Substitute coefficients using a loop
        self.substitutions = [(symbol, coefficient) for symbol, coefficient in zip(self.coefficients_symbols, self.coefficients)]
        self.equation_subs = self.equation.subs(self.substitutions)
        self.dfdx1_subs = self.dfdx1.subs(self.substitutions)
        self.dfdx2_subs = self.dfdx2.subs(self.substitutions)

        # Meshgrid
        self.y_hat = function((self.Data[self.x_1], self.Data[self.x_2]), *self.coefficients)
        self.x1_mesh, self.x2_mesh = np.meshgrid(np.linspace(min(self.Data[self.x_1]), max(self.Data[self.x_1]), 100), 
                                                 np.linspace(min(self.Data[self.x_2]), max(self.Data[self.x_2]), 100))
        self.y_mesh = function((self.x1_mesh, self.x2_mesh), *self.coefficients)

    ### Specific Plots

    def constant_plot(self, value: tuple[str, float], loc: tuple[float, float] = (0.05, 0.9)) -> None:

        if value[0] == self.x_1:

            # Substitute X1 with custom value in the equtation and lambdify it using X2 as arguments
            f = self.equation_subs.subs(x1, value[1])
            f_lam = smp.lambdify(x2, f)

            # Create both X2 and Y domains
            x = np.linspace(min(self.Data[self.x_2]), max(self.Data[self.x_2]), 100)
            y = f_lam(x)

            # Plot
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(self.x_2)
            ax.set_ylabel(self.y_i)
            text=f"{self.x_1}: {value[1]}"
            ax.text(loc[0], loc[1], text, fontsize=8, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"))
            ax.plot(x, y)
            plt.show()

        if value[0] == self.x_2:

            # Substitute X2 with custom value in the equtation and lambdify it using X1 as arguments
            f = self.equation_subs.subs(x2, value[1])
            f_lam = smp.lambdify(x1, f)
            
            # Create both X1 and Y domains:
            x = np.linspace(min(self.Data[self.x_1]), max(self.Data[self.x_1]), 100)
            y = f_lam(x)

            # Plot
            fig, ax = plt.subplots(1,1)
            ax.set_xlabel(self.x_1)
            ax.set_ylabel(self.y_i)
            text=f"{self.x_2}: {value[1]}"
            ax.text(loc[0], loc[1], text, fontsize=8, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="black"))
            ax.plot(x, y)
            plt.show()
    
    # Post init

    def __post_init__(self) -> None:
        self._model()
        self._n(), self._k()
        self._residuals(), self._shapiro()
        self._SST(), self._SSE(), self._SSR()
        self._r_squared()
        self._anova()
        self._coefficients_pvalues()
        self._stats()
        self._DataFramePredicted()
        try:
            self._error()
        except UnboundLocalError:
            pass
