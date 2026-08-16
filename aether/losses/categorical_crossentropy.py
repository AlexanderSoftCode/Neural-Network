import numpy as np
import aether.config as config
# Using Python 3.15 PEP 810: Explicit lazy imports would allow 
# for the import below to be called lazily, however for best 
# compatability this project would not use Python 3.15 for this issue
from aether.layers.activations import SoftMax
import aether.custom_kernels.loss_kernel as gpu_loss

def _to_sparse_labels(xp, y_true):
    """Canonicalize labels to sparse class-index form (S,).
    """
    return y_true if y_true.ndim == 1 else xp.argmax(y_true, axis=1)


def _cce_per_sample_loss(xp, probs_clip, y_true_sparse, n_classes, label_smoothing, training):
    """Per-sample CCE loss via direct gather"""
    samples = len(probs_clip)
    sample_idx = xp.arange(samples)
    target_probs = probs_clip[sample_idx, y_true_sparse]

    if label_smoothing > 0 and training:
        sum_log = xp.sum(xp.log(probs_clip), axis=1)
        return -(1.0 - label_smoothing) * xp.log(target_probs) - (label_smoothing / n_classes) * sum_log

    return -xp.log(target_probs)


class Loss:
    def __init__(self):
        self.new_pass()

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False, training = True):
        xp = config.get_array_module(output)
        sample_losses = self.forward(output, y, training) #calc sample losses
        data_loss = xp.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += xp.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization = False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss() 
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        regularization_loss = 0             # if we don't do this, we risk overfitting.
                                            # We will have to denote partials for this too...
        for layer in self.trainable_layers:        
            xp = config.get_array_module(layer.weights)
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                        xp.sum(xp.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        xp.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        xp.sum(xp.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        xp.sum(layer.biases * layer.biases) 
        return regularization_loss


class CategoricalCrossEntropy(Loss):
    def __init__(self, label_smoothing = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing 

    def forward(self, y_pred, y_true, training = True):
        xp = config.get_array_module(y_pred)
        n_classes = y_pred.shape[1]
        y_pred_clip = xp.clip(y_pred, 1e-7, 1 - 1e-7) #.000001 -> .999999
        y_true_sparse = _to_sparse_labels(xp, y_true)

        return _cce_per_sample_loss(xp, y_pred_clip, y_true_sparse, n_classes, self.label_smoothing, training)

    def backward(self, dvalues, y_true, training=True):
        xp = config.get_array_module(dvalues)
        samples = len(dvalues)
        n_classes = dvalues.shape[1]

        y_true_sparse = _to_sparse_labels(xp, y_true)
        dvalues_clip = xp.clip(dvalues, 1e-7, 1 - 1e-7)
        sample_idx = xp.arange(samples)

        dinputs = xp.zeros_like(dvalues)
        target_probs = dvalues_clip[sample_idx, y_true_sparse]

        if self.label_smoothing > 0 and training:
            dinputs += -(self.label_smoothing / n_classes) / dvalues_clip / samples
            dinputs[sample_idx, y_true_sparse] -= (1.0 - self.label_smoothing) / target_probs / samples
        else:
            dinputs[sample_idx, y_true_sparse] = -1.0 / target_probs / samples

        self.dinputs = dinputs


class SoftmaxCategoricalCrossEntropy(Loss):
    def __init__(self, label_smoothing = 0.0):
        self.activation = SoftMax()
        self.label_smoothing = label_smoothing
        super().__init__()

        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to bind the fused elementwise
        backward kernel or the fallback."""
        if device == 'cupy' and gpu_loss.is_gpu_softmax_cce_backward_available():
            self.backward = self._backward_gpu
        else:
            self.backward = self._backward_fallback

    #y_true is the vector of correct class indices, one per sample.
    #inputs is the raw logits, shape (n_samples, n_classes)
    def forward(self, inputs, y_true, training = True):
        xp = config.get_array_module(inputs)

        self.activation.forward(inputs, training=training)  # call forward function of softmax
        self.output = self.activation.output                

        n_classes = self.output.shape[1]
        y_true_sparse = _to_sparse_labels(xp, y_true)
        probs_clip = xp.clip(self.output, 1e-7, 1 - 1e-7)

        # Per-sample array, NOT a reduced scalar -- Loss.calculate() means
        # to call xp.mean()/xp.sum()/len() on this directly.
        return _cce_per_sample_loss(xp, probs_clip, y_true_sparse, n_classes, self.label_smoothing, training)

    def predictions(self, outputs):
        """Mirrors the functionality of softmax predictions, we require this as we pop softmax in combined pass"""
        xp = config.get_array_module(outputs)
        return xp.argmax(outputs, axis = 1)

    def _backward_fallback(self, dvalues, y_true, training = True):
        xp = config.get_array_module(dvalues)
        samples = len(dvalues)
        n_classes = dvalues.shape[1]

        y_true_sparse = _to_sparse_labels(xp, y_true)
        sample_idx = xp.arange(samples)

        # dinputs = (dvalues - y_true_smooth) / samples, built via copy +
        # scatter instead of materializing y_true_smooth as a full (S, C)
        # one-hot array.
        dinputs = dvalues.copy()

        if self.label_smoothing > 0 and training:
            dinputs -= self.label_smoothing / n_classes
            dinputs[sample_idx, y_true_sparse] -= (1.0 - self.label_smoothing)
        else:
            dinputs[sample_idx, y_true_sparse] -= 1.0

        dinputs /= samples
        self.dinputs = dinputs
        return self.dinputs

    def _backward_gpu(self, dvalues, y_true, training = True):
        xp = config.get_array_module(dvalues)
        samples = len(dvalues)
        n_classes = dvalues.shape[1]

        y_true_sparse = _to_sparse_labels(xp, y_true).astype(xp.int64, copy=False)
        class_idx = xp.arange(n_classes, dtype=xp.int64).reshape(1, n_classes)
        y_true_row = y_true_sparse.reshape(samples, 1)

        apply_smoothing = self.label_smoothing > 0 and training
        smooth_offset = np.float32(self.label_smoothing / n_classes if apply_smoothing else 0.0)
        target_offset = np.float32(1.0 - self.label_smoothing if apply_smoothing else 1.0)
        inv_samples = np.float32(1.0 / samples)

        self.dinputs = gpu_loss.softmax_cce_backward(
            dvalues, y_true_row, class_idx, smooth_offset, target_offset, inv_samples
        )
        return self.dinputs