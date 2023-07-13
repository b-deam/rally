import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit(x=None, y=None):
    def usl_equation(n, gamma, alpha, beta):
        n = np.array(n)
        return gamma * n / (1 + alpha * (n - 1) + beta * n * (n - 1))

    # usl_equation can be quite pessimistic given it's quadratic, so this offers a less pessimistic model
    def log_usl_equation(n, gamma, alpha, beta):
        n = np.array(n)
        return gamma * n / (1 + alpha * (n - 1) + beta * np.log(n) * (n - 1))

    if not x or not y:
        # dummy test data if none is provided
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 40, 50, 100])
        y = np.array([1000, 2000, 3000, 11000, 11500, 11200, 11200, 11200, 11000, 11200, 11000, 12000, 13100, 14000, 14000, 600, 900, 1000])

    x_min = np.min(x)
    x_max = np.max(x)
    x_extrapolation_max = round(x_max * 2)

    # TODO: if we have some inuition of alpha or beta, or the curve fit is wildly inaccurate for our measurements
    # we can provide them to `curve_fit`
    initial_guess = [1, 0, 0]

    # perform curve fitting to estimate gamma, alpha, and beta
    usl_popt, _ = curve_fit(usl_equation, x, y, p0=initial_guess, bounds=(0, np.inf))
    gamma_fit, alpha_fit, beta_fit = usl_popt
    print(f"gamma: {gamma_fit}")
    print(f"alpha: {alpha_fit}")
    print(f"beta: {beta_fit}")

    # perform curve fitting to estimate gamma, alpha, and beta for log usl
    log_usl_popt, _ = curve_fit(log_usl_equation, x, y, p0=initial_guess, bounds=(0, np.inf))
    log_gamma_fit, log_alpha_fit, log_beta_fit = log_usl_popt
    print(f"gamma: {log_gamma_fit}")
    print(f"alpha: {log_alpha_fit}")
    print(f"beta: {log_beta_fit}")

    # plot observed data
    plt.plot(x, y, "bo", label="Observed Data", alpha=0.5, markerfacecolor="white")

    # calculate and plot USL curve
    x_usl = np.linspace(x_min, x_extrapolation_max, 100)
    y_usl = usl_equation(x_usl, gamma_fit, alpha_fit, beta_fit)
    plt.plot(x_usl, y_usl, "r-", label=f"USL Curve - contention (α)={alpha_fit:.4f}, coherency (β)={beta_fit:.4f}")
    plt.plot([], [], " ", label=f"Predicted maximum throughput of {round(np.max(y_usl))} at concurrency {round(x_usl[np.argmax(y_usl)])}")
    plt.plot([], [], " ", label=f"Predicted lowest throughput of {round(np.min(y_usl))} at concurrency {round(x_usl[np.argmin(y_usl)])}")

    # calculate and plot logarithmic USL curve (less pessimistic)
    x_log_usl = np.linspace(x_min, x_extrapolation_max, 100)
    y_log_usl = log_usl_equation(x_log_usl, log_gamma_fit, log_alpha_fit, log_beta_fit)
    plt.plot(x_log_usl, y_log_usl, "g--", label=f"Log USL Curve - contention (α)={log_alpha_fit:.4f}, coherency (β)={log_beta_fit:.4f}")
    plt.plot(
        [],
        [],
        " ",
        label=f"Predicted (log usl) maximum throughput of {round(np.max(y_log_usl))} at concurrency {round(x_usl[np.argmax(y_log_usl)])}",
    )
    plt.plot(
        [],
        [],
        " ",
        label=f"Predicted (log usl) lowest throughput of {round(np.min(y_log_usl))} at concurrency {round(x_usl[np.argmin(y_log_usl)])}",
    )

    plt.xlabel("nodes")
    plt.ylabel("throughput")
    plt.legend()
    plt.show()
