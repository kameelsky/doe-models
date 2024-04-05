# Design of Experiments and empirical models

<div style="text-align:center">
    <img src="./data/animation.gif" alt="Contour Animation">
</div>

This python package has been developed to analyze experimental data obtained within the framework of 'Designs of Experiments'. 

Package can help with an analysis of screening experiemnts conducted with utilization of a commonly used design **factorial 2<sup>k</sup>**.
It is also capable of model validation, visualization of the outcome and finding the critical values in two factors optimization. Please check how to use it in [real case scenarios](./examples/).

## Dependencies
The application has a few which can be installed with python [pip module](./requirements.txt):

```shell
python -m pip install -r doe-models/requirements.txt
```
or [conda](./requirements.yml) package manager:
```shell
conda install -f doe-models/requirements.yml
```

## Downloading and installation
Download the repository and install the package with python 'pip' module.

```shell
git clone https://github.com/kameelsky/doe-models.git
python -m pip install doe-models/source
```

## License
[MIT License](./LICENSE.md)

## Examples

### Screen
```python
# Import the libraries
from doemodels.factorial import Factorial2k

# Create an instance of Factorial2k class for four factors
design = Factorial2k(["A", "B", "C", "D"])

# Create a fractional factorial
design.fractional("ABCD")

# Get a dictionary of aliased factors
design.aliases
```
{'A': ['BCD'],
 'B': ['ACD'],
 'C': ['ABD'],
 'D': ['ABC'],
 'AB': ['CD'],
 'AC': ['BD'],
 'AD': ['BC'],
 'BC': ['AD'],
 'BD': ['AC'],
 'CD': ['AB'],
 'ABC': ['D'],
 'ABD': ['C'],
 'ACD': ['B'],
 'BCD': ['A']}

```python
# Provide the responses
design.effect(response=[45, 65, 60, 80, 100, 45, 75, 96], n=1, graph=True)
```
![screen](./data/screen.png)

```python
# Plot a Pareto chart
design.pareto()
```

![pareto](./data/pareto.png)

More detailed analysis can be found in the [jupyter notebook](./examples/Screen.ipynb).

### Optimization
```python
# Import the libraries
from doemodels.two_factors import POL
from pandas import read_csv

# Create an instance of LIQ class
model = POL(x_1 = "Cond [mS/cm]",
            x_2 = "pH",
            y_i = "Binding capacity",
            Data = read_csv('./data/example_data.csv', index_col="No"))
```
```python
# Returns a response surface contour plot
model.rsp(figure_size = (7,5), step_x = 1, step_y = 0.1, dpi = 100,
          contours_number = 15, contour_color = "black", contour_font_size = 10,  color_map = "RdYlGn")
```

![rsp](./data/response_surface.png)

```python
# Returns a gradient plot
model.gradient(figure_size = (7,5), step_x = 1, step_y = 0.1, dpi = 100, contours_alpha=0.2)
```

![rsp](./data/gradient.png)

Magnitude of gradient vector $\left|\vec{R}\right|$ equals 0 at $x_1$ and $x_2$ coordinates for which function $f\left(x_1,x_2\right)$ returns the critical points. Therefore, to find $x_1$ and $x_2$ values which return gradient vector $\left[0\hat{i},0\hat{j}\right]$, partial derivatives were solved for respective scalars of $\hat{i}$ and $\hat{j}$ vectors:

$\vec{R} = \nabla f(x_{1}, x_{2}) = \left[\frac{\partial f(x_1, x_2)}{\partial x_1} \hat i, \frac{\partial f(x_1, x_2)}{\partial x_2} \hat j \\ \right] = \left[0 \hat i , 0 \hat j \\ \right]$

$\frac{\partial f}{\partial x_1} (x_1, x_2) \hat i = \beta_1 + 2\beta_{11}x_1 + \beta_{12}x_2 = 0 \hat i$

$\frac{\partial f}{\partial x_2} (x_1, x_2) \hat j  = \beta_2 + 2\beta_{22}x_2 + \beta_{12}x_1 = 0 \hat j$

After combining formulas $\frac{\partial f}{\partial x_1} (x_1, x_2) \hat i$ and $\frac{\partial f}{\partial x_2} (x_1, x_2) \hat j$, $x_1$ and $x_2$ coordinates for which function returns critical points were conveniently calculated using linear algebra. The coefficient matrix took form of the Hessian matrix and its eigenvalues were used to classify the critical point.

```python
# We solve the model
model.solve()
```

    Cond [mS/cm]: 3.30
    pH: 5.41
    Critical value: 142.51
    Hessian matrix eigenvalues: [-0.63, -152.44]

Since loading the protein in the buffer with such low conductivity is not sensible, we focus on particular value of the 'Cond [ms/cm]'.

```python
model.constant_plot(value=("Cond [mS/cm]", 5), loc=(0.05, 0.85))
```
![rsp](./data/quadratic_function.png)

More detailed analysis can be found in the [jupyter notebook](./examples/Optimization.ipynb).