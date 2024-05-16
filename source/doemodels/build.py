from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f, shapiro, anderson, normaltest
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings

sns.set_style("ticks")

@dataclass
class Model:

    formula: str
    data: pd.DataFrame = field(repr=False)
    y_variable: str = field(repr=False, init=False)
    X_variables: list[str] = field(repr=False, init=False)
    statsmodel: statsmodels.regression.linear_model.RegressionResultsWrapper = field(init=False, repr=False)
    data_predict: pd.DataFrame = field(repr=False, init=False)
    data_aggregate: pd.DataFrame = field(repr=False, init=False)
    anova_table: pd.DataFrame = field(repr=False, init=False)
    coefficients_table: pd.DataFrame = field(repr=False, init=False)
    metrics_table: pd.DataFrame = field(repr=False, init=False)

    
    def fit(self):
         
        self.statsmodel = smf.ols(formula=self.formula, data=self.data).fit()
        return self.statsmodel

    def variables(self):
         
        self.y_variable = self.formula.split('~')[0].strip()
        self.X_variables = self.data.drop(columns=self.y_variable).columns.to_list()
        return self.y_variable, self.X_variables

    def prediction_table(self):

        self.data_predict = self.data.copy()
        self.data_predict['Predicted_Capacity'] = self.statsmodel.predict()
        return self.data_predict

    def aggregate(self):

        df = self.data.copy()
        y_variable = self.formula.split('~')[0].strip()
        y = df[y_variable]
        X = df.drop(columns=y_variable)
        X_variables = X.columns.to_list()
        self.data_aggregate = df.groupby(by=X_variables).agg(["mean", "std", "count"])
        return self.data_aggregate

    def coefficients(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables = self.statsmodel.summary2().tables
        self.coefficients_table = tables[1]
        self.coefficients_table.columns = ['Coefficient', 'Std. Error', 't', 'Prob(t)', '[0.025', '0.975]']
        return self.coefficients_table

    def anova(self):

        # init ANOVA table
        aov = sm.stats.anova_lm(self.statsmodel)
        residual = aov.loc["Residual"] # residuals
        aov = aov.drop(index="Residual")
        
        # Model calculations
        df = aov["df"].sum()
        ssreg = aov["sum_sq"].sum()
        mean_ssreg = ssreg / df
        mean_sserr = residual["mean_sq"]
        f_stat = mean_ssreg / mean_sserr
        model_series = pd.Series({
            "df": aov["df"].sum(),
            "sum_sq": aov["sum_sq"].sum(),
            "mean_sq": mean_ssreg,
            "F": f_stat,
            "PR(>F)": 1 - f.cdf(f_stat, df, residual["df"])})
        
        # Combine Model and Residual into previous ANOVA table
        new_rows = pd.DataFrame([model_series, residual], index=["Model", "Error"])
        self.anova_table = pd.concat([aov, new_rows])

        # Rename the columns
        self.anova_table.columns = ['DF', 'Sum of Squares', 'Mean square', 'F', 'Prob(F)']
        return self.anova_table

    def metrics(self):
        
        rmse = np.sqrt(np.mean(self.statsmodel.resid**2))
        metrics = [self.statsmodel.rsquared_adj, self.statsmodel.rsquared, self.statsmodel.bic, self.statsmodel.aic, rmse]
        self.metrics_table = pd.DataFrame({self.formula: metrics}, index=["R squared (adjusted)", "R squared", "BIC", "AIC", "RMSE"])
        return self.metrics_table

    def normality(self):

        residuals = self.statsmodel.resid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shapiro_test = shapiro(residuals)
            agostino_test = normaltest(residuals)
            anderson_test = anderson(residuals, dist='norm')

        tests = ["Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling"]
        statistics = [shapiro_test.statistic, agostino_test.statistic, anderson_test.statistic]
        pvalues = [shapiro_test.pvalue, agostino_test.pvalue, np.nan]
        assessments = []
        if shapiro_test.pvalue > 0.05:
            assessments.append("Yes")
        else:
            assessments.append("No")
        if agostino_test.pvalue > 0.05:
            assessments.append("Yes")
        else:
            assessments.append("No")
        if anderson_test.statistic < anderson(residuals, dist='norm').critical_values[2]:
            assessments.append("Yes")
        else:
            assessments.append("No")
        
        fig = sm.qqplot(residuals, line='45', fit=True)
        plt.show()

        return pd.DataFrame({test:[statistic, pvalue, assesment] for test, statistic, pvalue, assesment in zip(tests, statistics, pvalues, assessments)}, 
                            index=["Statistic", "pvalue", "Normal"])

    def homoscedasticity(self, groups: list[str]):

        residuals = self.statsmodel.resid
        bp_test = het_breuschpagan(residuals, self.statsmodel.model.exog)

        tests = ["Breusch-Pagan"]
        statistics = [bp_test[0]]
        pvalues = [bp_test[1]]
        assessments = []

        if bp_test[1] > 0.05:
            assessments.append("Yes")

        # Residuals plot
        df = self.data.copy()
        df["Residuals"] = self.statsmodel.resid
        if len(groups) > 1:
            fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(len(groups) * 6, 5))
            for number, group in enumerate(groups):
                sns.scatterplot(data=df, x=group, y=residuals, ax=axes[number])
            for ax in axes:
                ax.set_ylabel("Residuals")
                ax.axhline(0, linestyle="dashed", color="red", alpha=0.5)
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=df, x=groups[0], y=residuals)
            ax.axhline(0, linestyle="dashed", color="red", alpha=0.5)
        plt.show()

        
        return pd.DataFrame({test:[statistic, pvalue, assesment] for test, statistic, pvalue, assesment in zip(tests, statistics, pvalues, assessments)}, 
                            index=["Statistic", "pvalue", "Homoscedastic"])

    def __post_init__(self):
        self.fit()
        self.variables()
        self.prediction_table()
        self.aggregate()
        self.anova()
        self.coefficients()
        self.metrics()