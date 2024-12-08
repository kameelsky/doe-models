import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define the range for X1, X2, and X3
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
x3 = np.linspace(-10, 10, 100)

# Create a meshgrid for X1 and X2
X1, X2 = np.meshgrid(x1, x2)

# Function to calculate Y
def func(X1, X2, X3):
    return X1**2 + X2**2 + X3**2 + 2*X1*X2 + 3*X2*X3

# Calculate Z for all X3 values
y = [func(X1, X2, X3_val) for X3_val in x3]

# Determine the contour levels based on all Z values
min_y = np.min(y)
max_y = np.max(y)
levels = np.linspace(min_y, max_y, 15)

# Create a figure
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)

# Function to update the contour plot for each frame of the animation
def update(frame):
    ax.clear()
    cf = ax.contourf(X1, X2, y[frame], levels=levels, cmap='ocean')
    ax.set_title(f"f(X1, X2, X3 = {x3[frame]:.2f})")
    ax.set_xlabel("X1", fontweight="bold")
    ax.set_ylabel("X2", fontweight="bold")
    ax.text(0, 0, 'Design of Experiments\nModels for data analysis', fontweight="bold", ha='center', va='center', fontsize=20, color='white')
    fig.colorbar(cf, cax=cax, ticks=levels)

   
# Create the animation with adjusted duration and FPS
ani = animation.FuncAnimation(fig, update, frames=len(x3), interval=100, repeat=True)

# Save the animation as a GIF
ani.save('animation.gif', writer='pillow')