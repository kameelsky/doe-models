# Design of Experiments models for data analysis

The software has been designed to analyze experimental data obtained within the framework of 'Designs of Experiments'. Application is capable of model validation, visualization of the outcome and finding the critical values.

## Dependencies
The application has a few [dependencies](./requirements.txt), which can be installed with python pip module.

```shell
python -m pip install -r doe-models/requirements.txt
```

## Downloading and installation
Download the repository and install the package with python 'pip' module.

```shell
git clone https://github.com/kameelsky/doe-models.git
python -m pip install doe-models/source
```

## License
[MIT License](./LICENSE.md)

## Usage

Example usage, usefull plots and functions can be found in the [examples](./examples/) directory.

### **Optimization methods**

#### **LIQ (Linear, Interaction, Quadratic) for two factors:**

$Y = f(x_1, x_2) = \beta_0 + \beta_1x_1 + \beta_2x_2 + \beta_{11}x_1^2 + \beta_{22}x_2^2 + \beta_{12}x_1x_2 + e$

```python
# Import the libraries
from models.two_factors import LIQ
from pandas import read_csv

# Create an instance of LIQ class
model = LIQ(x_1 = "Cond [mS/cm]",
            x_2 = "pH",
            y_i = "Binding capacity",
            Data = read_csv('./data/example_data.csv', index_col="No"))

# Outputs a response surface contour plot
model.rsp(figure_size = (7,5), step_x = 1, step_y = 0.1, dpi = 100,
          contours_number = 15, contour_color = "black", contour_font_size = 10,  color_map = "RdYlGn")

# Solves the equtation
model.solve()
```
![rsp](./data/response_surface.png)

    Cond [mS/cm]: 3.30
    pH: 5.41
    Critical value: 142.51
    Hessian matrix eigenvalues: [-0.63, -152.44]
