import numpy as np

def cubic_spline_coeffs(x, y):
    n = len(x) - 1
    h = np.diff(x)
    alpha = [0] + [3/h[i] * (y[i+1] - y[i]) - 3/h[i-1] * (y[i] - y[i-1]) for i in range(1, n)]

    # Solve the tridiagonal system for coefficients c
    l = [1] + [2 * (x[i+1] - x[i-1]) for i in range(1, n)] + [1]
    mu = [0] + [h[i] / (h[i] + h[i+1]) for i in range(1, n)]
    z = [0] * (n + 1)

    # Forward elimination
    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        alpha[i] = (alpha[i] - h[i-1] * alpha[i-1]) / l[i]
    
    # Back substitution
    c = [0] * (n + 1)
    for j in range(n-1, 0, -1):
        c[j] = alpha[j] - mu[j] * c[j+1]

    # Solve for b and d
    b = [(y[i+1] - y[i])/h[i] - h[i] * (c[i+1] + 2 * c[i])/3 for i in range(n)]
    d = [(c[i+1] - c[i]) / (3 * h[i]) for i in range(n)]
    a = y[:-1]
    
    return a, b, c[:-1], d

def eval_spline(x, coeffs, t):
    a, b, c, d = coeffs
    # Find the right interval for t
    i = np.searchsorted(x, t) - 1
    i = max(min(i, len(a) - 1), 0)
    dx = t - x[i]
    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

# Example data
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])  # y = x^2

# Compute spline coefficients
coeffs = cubic_spline_coeffs(x, y)

# Plot the spline
import matplotlib.pyplot as plt

x_fine = np.linspace(0, 5, 100)
y_fine = [eval_spline(x, coeffs, xi) for xi in x_fine]

plt.figure(figsize=(8, 4))
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_fine, y_fine, label='Cubic Spline')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Manual Cubic Spline Interpolation')
plt.legend()
plt.show()
