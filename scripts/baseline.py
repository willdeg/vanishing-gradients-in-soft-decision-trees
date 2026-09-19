import numpy as np


class LogisticRegressionGD:
    """
    Binary logistic regression trained with batch gradient descent.

    """

    def __init__(self):
        self.coefficients = None

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, x, y, learning_rate=0.1, n_iter=1000, tol=1e-4):
        # Add a bias column
        x_bias = np.hstack((np.ones((x.shape[0], 1)), x))
        n_samples, n_features = x_bias.shape
        self.coefficients = np.zeros(n_features, dtype=np.float64)

        for _ in range(n_iter):
            previous = self.coefficients.copy()
            # Compute the current predicted probabilities
            y_prob = self.sigmoid(x_bias @ self.coefficients)
            # The average gradient of the log-loss tells us how to update w
            gradient = x_bias.T @ (y_prob - y) / n_samples

            self.coefficients -= learning_rate * gradient

            # Stop early once the coefficients are barely changing
            if np.max(np.abs(self.coefficients - previous)) < tol:
                break

    def predict_proba(self, x):
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before prediction.")

        x_bias = np.hstack((np.ones((x.shape[0], 1)), x))
        return self.sigmoid(x_bias @ self.coefficients)
        

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(np.int64)
