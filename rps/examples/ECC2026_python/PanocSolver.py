from __future__ import annotations

import numpy as np


class PanocSolver:
    """PANOC implementation ported from the MATLAB handle class."""

    def __init__(self, gamma, sigma, max_iter, tol, u_min, u_max, lbfgs_size):
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.u_min = np.asarray(u_min, dtype=float).reshape(-1)
        self.u_max = np.asarray(u_max, dtype=float).reshape(-1)
        self.lbfgs_size = int(lbfgs_size)

    def solve(self, u_init, params, field):
        u_k = np.asarray(u_init, dtype=float).reshape(-1).copy()
        dimension = u_k.size
        S = np.zeros((dimension, self.lbfgs_size))
        Y = np.zeros((dimension, self.lbfgs_size))
        lbfgs_index = 0
        lbfgs_count = 0
        grad_k = params["gradient_function"](u_k, params, field)
        residual_norm = float("inf")
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            cost_k = params["cost_function"](u_k, params, field)
            u_bar = self.project_U(u_k - self.gamma * grad_k)
            diff_u = u_bar - u_k
            gamma_changed = False
            while params["cost_function"](u_bar, params, field) > (
                cost_k + np.dot(grad_k, diff_u)
                + np.dot(diff_u, diff_u) / (2.0 * self.gamma)
            ):
                self.gamma *= 0.5
                self.sigma *= 0.5
                u_bar = self.project_U(u_k - self.gamma * grad_k)
                diff_u = u_bar - u_k
                gamma_changed = True
                if self.gamma < 1e-16:
                    break
            if gamma_changed:
                lbfgs_count = 0

            residual = -diff_u / self.gamma
            residual_norm = np.linalg.norm(residual, ord=np.inf)
            if residual_norm < self.tol:
                break

            direction = self.compute_lbfgs_direction(-residual, S, Y, lbfgs_count)
            fbe_k = (
                cost_k - self.gamma * np.dot(grad_k, grad_k) / 2.0
                + np.linalg.norm(u_k - self.gamma * grad_k - u_bar) ** 2
                / (2.0 * self.gamma)
            )
            tau = 1.0
            while tau > 1e-6:
                u_next = u_k - (1.0 - tau) * self.gamma * residual + tau * direction
                cost_next = params["cost_function"](u_next, params, field)
                grad_next = params["gradient_function"](u_next, params, field)
                u_bar_next = self.project_U(u_next - self.gamma * grad_next)
                fbe_next = (
                    cost_next - self.gamma * np.dot(grad_next, grad_next) / 2.0
                    + np.linalg.norm(u_next - self.gamma * grad_next - u_bar_next) ** 2
                    / (2.0 * self.gamma)
                )
                if fbe_next <= fbe_k - self.sigma * np.dot(residual, residual):
                    break
                tau *= 0.5

            if tau <= 1e-6:
                u_next = u_k - self.gamma * residual
                grad_next = params["gradient_function"](u_next, params, field)
                u_bar_next = self.project_U(u_next - self.gamma * grad_next)
            residual_next = (u_next - u_bar_next) / self.gamma
            s_k = u_next - u_k
            y_k = residual_next - residual
            if np.dot(s_k, y_k) > 1e-10:
                S[:, lbfgs_index] = s_k
                Y[:, lbfgs_index] = y_k
                lbfgs_index = (lbfgs_index + 1) % self.lbfgs_size
                lbfgs_count = min(lbfgs_count + 1, self.lbfgs_size)
            u_k = u_next
            grad_k = grad_next

        return u_k, residual_norm, iteration

    @staticmethod
    def compute_lbfgs_direction(q, S, Y, lbfgs_count):
        q = np.asarray(q, dtype=float).copy()
        if lbfgs_count == 0:
            return q
        alpha = np.zeros(lbfgs_count)
        rho = np.zeros(lbfgs_count)
        for index in range(lbfgs_count - 1, -1, -1):
            rho[index] = 1.0 / np.dot(Y[:, index], S[:, index])
            alpha[index] = rho[index] * np.dot(S[:, index], q)
            q -= alpha[index] * Y[:, index]
        H0 = np.dot(S[:, lbfgs_count - 1], Y[:, lbfgs_count - 1]) / np.dot(
            Y[:, lbfgs_count - 1], Y[:, lbfgs_count - 1]
        )
        direction = H0 * q
        for index in range(lbfgs_count):
            beta = rho[index] * np.dot(Y[:, index], direction)
            direction += S[:, index] * (alpha[index] - beta)
        return direction

    def project_U(self, u_in):
        return np.maximum(self.u_min, np.minimum(self.u_max, u_in))
