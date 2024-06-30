import numpy as np
import inspect

class Metrics(): 

    def __init__(self, metrics_to_run, args=None):
        """
        Initialize a Metrics object.

        Args:
            metrics_to_run (list): List of metrics to run.
            args (dict):           Additional arguments required by some of the metrics
        """
        # Store the list of metrics to run
        self.metrics_to_run = metrics_to_run

        # Store the additional arguments
        self.args = args

    def run(self, y_true, y_pred):
        """
        Run the metrics on the given true and predicted values.

        Args:
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.

        Yields:
            tuple: A tuple containing the name of the metric and its corresponding value.
        """
        # Iterate over all the functions in the Metrics class
        for name, func in inspect.getmembers(Metrics, inspect.isfunction):
            if name in self.metrics_to_run:
                yield (name, func(y_true, y_pred, **self.args))

    def r2(y_true, y_pred, **kwargs):
        """
        Calculate the R^2 score.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mean_y = np.mean(y_true)
        ss_tot = np.sum((y_true - mean_y)**2)
        ss_res = np.sum((y_true - y_pred)**2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def r2_adj(y_true, y_pred, n_features, **kwargs): 
        """
        Calculate the adjusted R^2 score.
        """
        r2 = Metrics.r2(y_true, y_pred)
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adj_r2

    def rms(y_true, y_pred, **kwargs):
        """
        Calculate the root mean squared error.
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        error = y_pred - y_true
        error = np.nanmean(error**2)
        return np.sqrt(error)
    
    def mape(y_true, y_pred, **kwargs): 
        """
        Calculate the mean absolute percentage error. 
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        error = y_pred - y_true
        error = np.abs(error)
        error = np.nanmean(error)
        return error






