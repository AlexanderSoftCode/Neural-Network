import numpy as np
import aether.config as config
class Accuracy:
    #Givers the accuracy of the prediction sand truth values
    def calculate(self, predictions, y):
        xp = config.get_array_module(predictions)
        comparisons = self.compare(predictions, y)

        accuracy = xp.mean(comparisons) 
    
        self.accumulated_sum += xp.sum(comparisons)
        self.accumulated_count += len(comparisons)

    
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def predict(self, X, *, batch_size = None):
        prediction_steps = 1
class RegressionAccuracy(Accuracy):

    def __init__(self):
        self.precision = None
    
    #Now we are getting the precision value
    def init(self, y, reinit = False):
        xp = config.get_array_module(y)
        if self.precision is None or reinit:
            self.precision = xp.std(y) / 250 
    
    def compare(self, predictions, y):
        xp = config.get_array_module(predictions)
        return xp.absolute(predictions - y) < self.precision

class CategoricalAccuracy(Accuracy):
    def init(self, y):
        pass
    
    def compare(self, predictions, y):
        xp = config.get_array_module(predictions)

        if len(predictions.shape) == 2:
            predictions = xp.argmax(predictions, axis=1)
        if len(y.shape) == 2:
            y = xp.argmax(y, axis = 1)
        return predictions == y
