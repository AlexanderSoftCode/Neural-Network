import aether.config as config
from aether.base import Layer
from aether.losses import Loss, CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy
from aether.metrics import Accuracy
from aether.layers.activations import SoftMax
class Model():
    def __init__(self):
        self.layers = []
        self.is_finalized = False
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self._seed = None

    def add(self, layer):
        if self.is_finalized:   
            raise RuntimeError("Cannot modify model after finalize() has been called.")
        
        if not isinstance(layer, Layer):
            raise TypeError(
                f"Expected an instance of 'Layer', but got `{type(layer).__name__}`."
                "Make sure the layer your passing in inherits from aether.base.Layer"
            )
        self.layers.append(layer)

    def manual_seed(self, seed: int):
        if self.is_finalized:
            raise RuntimeError("Cannot set a new seed after finalize() has been called.")
        self._seed = int(seed)
        return self
    
    def configure(self, loss=None, optimizer=None, accuracy=None):
        """Configures the training components (loss, optimizer, metrics) for the model.
        Utilizes strict type checking for Loss and duck typing for optimizer/metric.
        """
        if loss is None and optimizer is None and accuracy is None:
            raise ValueError(
                "At least one component (loss, optimizer, or metrics) must be"
                " provided to configure()."
            )

        if loss is not None:
            if not isinstance(loss, Loss):
                raise TypeError(
                    f"Expected an instance of 'Loss', but got {type(loss).__name__}."
                    "Make sure the loss your passing in inherits from aether.losses.loss"
                )
            else:
                self.loss = loss

        if optimizer is not None:
            if not hasattr(optimizer, "update_params") and not hasattr(optimizer, "step"):                
                raise TypeError(
                    f"Object '{type(optimizer).__name__}' is not a valid optimizer."
                    "Expected method 'update_params' or 'step'"
                )
            self.optimizer = optimizer

        if accuracy is not None:
            if not isinstance(accuracy, Accuracy):
                raise TypeError(
                    f"Object '{type(accuracy).__name__}' is not a valid accuracy metric."
                    "Make sure the accuracy your passing in inherits from aether.metrics.accuracy"
                )
            self.accuracy = accuracy

    def _sync_device(self, target_device=None):
        """Internal dispatch to compile all registered components into active backend."""
        dev = target_device or getattr(self, "device", None)
        if dev is None:
            return

        # Consolidate all model items (layers + training modules)
        components = [
            *self.layers,
            getattr(self, "loss", None),
            getattr(self, "optimizer", None),
            getattr(self, "accuracy", None),
        ]

        for comp in components:
            if comp is not None and hasattr(comp, "_compile_for_device"):
                comp._compile_for_device(dev)

    def to(self, device):
        """
        Ahead-Of-Time compilation and device migration.

        Configures the global execution backend (numpy or cupy) and
        recursively prepares all components device backend, and dedicated
        kernels if user is using cupy.

        Args:
            device (str): The target hardware execution device 
            ('cupy' or 'numpy)
        Raises:
            RuntimeError: If user calls model.to() before model.finalize()
            ValueError: If `device` specifies an unsupported or unconfigured backend
        
        Note:
            Method must be called prior to training or evaluation if switching backends
            (e.g., NumPy to CuPy), as it triggers internal array migrations and kernel
            allocations across all underlying components.
        """
        if self.is_finalized:
            raise RuntimeError(
                "[aether] model.to() must be called BEFORE model.finalize(). " \
                "Re-migrating an already finalized graph causes needless host-device copies."
            )
        target_device = device.lower()
        config.set_backend(target_device)
        self.device = target_device

        self._sync_device(target_device=target_device)

    def set_precision(self, compute_dtype):
        """
        Sets the target floating-point precision policy on the model and
        dispatches it to all registered layers, skipping any layer marked with
        the `_precision_exempt` attribute.

        Args:
            compute_dtype (str): The target floating point precision, 
            ('float16','float32', 'float64') 
        Raises:
            TypeError: If `compute_dtype` is not an str or None
            ValueError: If `compute_dtype` is not one of the currently supported precision
            RuntimeError: If `compute_dtype` is not supported in NumPy
        Note:
            Method can be called before or after both Model.to and Model.finalize, but must 
            be called before training and inference. Layers makred with '_precision_exempt=True'
            (e.g.) normalization or softmax layers retain single-precision compute to 
            prevent numerical instabilility. Optimizer and Loss classes will also always 
            perform single precision calculations. 
        """
        self.precision_policy = config.DTypePolicy(compute_dtype)
        for layer in self.layers:
            if (hasattr(layer, "_apply_precision")
                 and not getattr(layer, "_precision_exempt", False)):
                layer._apply_precision(self.precision_policy)

    def finalize(self):
        if self.is_finalized:   
            raise RuntimeError("Cannot modify model after finalize() has been called.")

        if not self.layers:
            raise RuntimeError("[aether] Cannot finalize an empty model. Please add layers via Model.add() first.")

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "build") and callable(layer.build):
                layer_seed = (self._seed + idx) if self._seed is not None else None
                layer.build(seed=layer_seed)
                
            elif hasattr(layer, "_set_seed") and callable(layer._set_seed):
                if getattr(layer, "seed", None) is None and self._seed is not None:
                    layer._set_seed(self._seed)
                
        self.trainable_layers = [
            layer for layer in self.layers
            if hasattr(layer, "weights") or hasattr(layer, "biases")
        ]

        if self.loss is not None:
            last_layer = self.layers[-1]
            if isinstance(self.loss, SoftmaxCategoricalCrossEntropy) and isinstance(last_layer, SoftMax):
                raise ValueError(
                    "[aether] 'SoftmaxCategoricalCrossEntrpy' operates directly on unnormalized logits (S, C)." \
                    "Do not add an explicit 'SoftMax' activation layer to the model."
                )

            self.loss.remember_trainable_layers(self.trainable_layers)
            

        if self.optimizer is not None:
            if not hasattr(self.optimizer, "step") or not hasattr(self.optimizer, "init_params"):
                raise TypeError(
                    f"Optimizer '{type(self.optimizer).__name__}' must implement "
                    "'init_params(trainable_layers)' and 'step()'."
                )
            self.optimizer.init_params(self.trainable_layers)
            self._step_optimizer = self.optimizer.step
                
        self._sync_device()
        self.is_finalized = True

    def forward(self, X, training=True):
        """
        Sequentially propagate input batch through all layers.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, loss_dinputs):
        """
        Sequentially propagate loss gradients backward through layers in reverse order 
        """
        dinputs = loss_dinputs
        for layer in reversed(self.layers):
            dinputs = layer.backward(dinputs)
        return dinputs

    def train(
        self,
        X,
        y,
        *,
        epochs=1,
        batch_size=None,
        print_every=1,
        verbose = True,
        validation_data=None
    ):
        """
        Train the compiled neural network on dataset (X, y).

        AOT scheduling to avoid python conditionals inside the main batch loops, minimizing 
        CPU overhead

        Args:
            X (ndarray): Input features (NumPy or CuPy ndarray).
            y (ndarray): Target labels or one-hot ground truths.
            epochs (int): Number of full passes over the dataset.
            batch_size (int, optional): Mini-batch sample size. Defaults to None (full-batch).
            print_every (int): Step interval frequency for logging telemetry.
            verbose (bool, optional): If True, prints training loss and accuracy metrics to stdout. Defaults to True. 
            validation_data (tuple, optional): (X_val, y_val) tuple for out-of-sample testing.
        """
        if not self.is_finalized:
            self.finalize()

        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        train_steps = (num_samples + effective_batch_size - 1) // effective_batch_size

        batch_slices = [
            (step * effective_batch_size, min((step + 1) * effective_batch_size, num_samples))
            for step in range(train_steps)
        ]

        # -----------------------------------------------------------------
        # 2. Local Namespace Binding (Attribute caching to eliminate lookup latency)
        # -----------------------------------------------------------------
        forward_fn = self.forward
        backward_fn = self.backward
        loss = self.loss
        accuracy = self.accuracy
        step_optimizer = getattr(self, "_step_optimizer", None)

        has_loss = loss is not None
        has_acc = accuracy is not None
        has_opt = callable(step_optimizer)

        has_reg = False
        if has_loss and hasattr(loss, "regularization_loss"):
            has_reg = any(
                getattr(layer, "weight_regularizer_l1", 0.0) > 0.0
                or getattr(layer, "weight_regularizer_l2", 0.0) > 0.0
                or getattr(layer, "bias_regularizer_l1", 0.0) > 0.0
                or getattr(layer, "bias_regularizer_l2", 0.0) > 0.0
                for layer in getattr(self, "trainable_layers", [])
            )
        reg_loss_fn = loss.regularization_loss if has_reg else None

        if has_loss and has_acc and has_opt:
            if has_reg:
                def run_step(batch_X, batch_y):
                    out = forward_fn(batch_X, training=True)
                    data_l = loss.calculate(out, batch_y)
                    reg_l = reg_loss_fn()
                    acc_l = accuracy.calculate(out, batch_y)
                    loss.backward(out, batch_y)
                    backward_fn(loss.dinputs)
                    step_optimizer()
                    return data_l, reg_l, acc_l
            else:
                def run_step(batch_X, batch_y):
                    out = forward_fn(batch_X, training=True)
                    data_l = loss.calculate(out, batch_y)
                    acc_l = accuracy.calculate(out, batch_y)
                    loss.backward(out, batch_y)
                    backward_fn(loss.dinputs)
                    step_optimizer()
                    return data_l, 0.0, acc_l

        elif has_loss and has_opt:
            if has_reg:
                def run_step(batch_X, batch_y):
                    out = forward_fn(batch_X, training=True)
                    data_l = loss.calculate(out, batch_y)
                    reg_l = reg_loss_fn()
                    loss.backward(out, batch_y)
                    backward_fn(loss.dinputs)
                    step_optimizer()
                    return data_l, reg_l, 0.0
            else:
                def run_step(batch_X, batch_y):
                    out = forward_fn(batch_X, training=True)
                    data_l = loss.calculate(out, batch_y)
                    loss.backward(out, batch_y)
                    backward_fn(loss.dinputs)
                    step_optimizer()
                    return data_l, 0.0, 0.0
        else:
            # Generic fallback
            def run_step(batch_X, batch_y):
                out = forward_fn(batch_X, training=True)
                data_l = loss.calculate(out, batch_y) if has_loss else 0.0
                reg_l = reg_loss_fn() if has_reg else 0.0
                acc_l = accuracy.calculate(out, batch_y) if has_acc else 0.0
                if has_loss:
                    loss.backward(out, batch_y)
                    backward_fn(loss.dinputs)
                if has_opt:
                    step_optimizer()
                return data_l, reg_l, acc_l

        get_lr = None
        if self.optimizer is not None:
            opt = self.optimizer
            get_lr = lambda: getattr(opt, "current_learning_rate", getattr(opt, "lr", None))

        for epoch in range(1, epochs + 1):
            if has_loss:
                loss.new_pass()
            if has_acc:
                accuracy.new_pass()

            for step, (start_idx, end_idx) in enumerate(batch_slices):
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]

                data_loss, reg_loss, acc_val = run_step(batch_X, batch_y)

                # Telemetry: GPU-to-Host sync barrier strictly confined to log steps
                if verbose and print_every and (step % print_every == 0 or step == train_steps - 1):
                    s_data_loss = float(data_loss)
                    s_reg_loss = float(reg_loss)
                    s_loss = s_data_loss + s_reg_loss
                    s_acc = float(acc_val)

                    lr_str = ""
                    if get_lr is not None:
                        lr = get_lr()
                        if lr is not None:
                            lr_str = f" - lr: {float(lr):.6f}"

                    print(
                        f"Epoch {epoch}/{epochs} | Step {step + 1}/{train_steps} "
                        f"- loss: {s_loss:.4f} (data: {s_data_loss:.4f}, reg: {s_reg_loss:.4f}) "
                        f"- acc: {s_acc:.4f}{lr_str}"
                    )

            if has_loss:
                accum = loss.calculate_accumulated(include_regularization=has_reg)
                epoch_loss = float(accum[0] + accum[1]) if isinstance(accum, tuple) else float(accum)
            else:
                epoch_loss = 0.0

            epoch_acc = float(accuracy.calculate_accumulated()) if has_acc else 0.0

            lr_summary = ""
            if get_lr is not None:
                lr = get_lr()
                if lr is not None:
                    lr_summary = f" - lr: {float(lr):.6f}"
            if verbose and print_every:
                print(
                    f"[Epoch {epoch}/{epochs} Total] "
                    f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}{lr_summary}"
                )

            if validation_data is not None:
                X_val, y_val = validation_data
                self.evaluate(X_val, y_val, batch_size=batch_size, verbose=verbose)

    def evaluate(self, X_val, y_val, *, batch_size=None, verbose=True):
        """
        Evaluate the model's loss and metrics on validation/test data in inference mode.

        Args:
            X_val (ndarray): Evaluation input features (NumPy or CuPy ndarray).
            y_val (ndarray): Ground truth labels or targets.
            batch_size (int, optional): Mini-batch size. Defaults to None (full-batch).
            verbose (bool, optional): If True, prints evaluation loss and accuracy metrics to stdout. Defaults to True.   
        Returns:
            tuple: (val_loss, val_acc) representing the computed validation metrics.
        """
        if not self.is_finalized:
            self.finalize()

        num_samples = len(X_val)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        eval_steps = (num_samples + effective_batch_size - 1) // effective_batch_size


        has_loss = self.loss is not None
        has_acc = self.accuracy is not None

        if has_loss:
            self.loss.new_pass()
        if has_acc:
            self.accuracy.new_pass()

        # Cache local forward function
        forward_fn = self.forward
        loss = self.loss
        accuracy = self.accuracy


        for step in range(eval_steps):
            start_idx = step * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, num_samples)

            batch_X = X_val[start_idx:end_idx]
            batch_y = y_val[start_idx:end_idx]

            # Forward propagation in evaluation mode
            output = forward_fn(batch_X, training=False)

            # Accumulate batch metrics without tracking gradients or regularization
            if has_loss:
                loss.calculate(output, batch_y, training=False)

            if has_acc:
                accuracy.calculate(output, batch_y)


        val_loss = float(loss.calculate_accumulated(include_regularization=False)) if has_loss else 0.0
        val_acc = float(accuracy.calculate_accumulated()) if has_acc else 0.0

        if verbose: 
            print(f"[Validation] loss: {val_loss:.4f} - acc: {val_acc:.4f}")

        return val_loss, val_acc

    def predict(self, X, *, batch_size=None, return_logits=False):
        """
        Run batch inference on inputs and return model predictions.

        Automatically routes raw outputs through any fused loss activation
        (e.g., SoftMax in SoftmaxCategoricalCrossEntropy) unless raw logits are requested.

        Args:
            X (ndarray): Input feature batch (NumPy or CuPy array).
            batch_size (int, optional): Mini-batch sample size. Defaults to None (full-batch).
            return_logits (bool, optional): If True, returns raw network logits
                even if an output activation was fused into the loss. Defaults to False.

        Returns:
            ndarray: Model predictions (probabilities or stacked batch outputs).
        """
        if not self.is_finalized:
            self.finalize()

        xp = config.get_array_module(X)
        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        prediction_steps = (num_samples + effective_batch_size - 1) // effective_batch_size

        forward_fn = self.forward
        outputs = []

        for step in range(prediction_steps):
            start_idx = step * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, num_samples)

            batch_X = X[start_idx:end_idx]
            batch_output = forward_fn(batch_X, training=False)
            outputs.append(batch_output)

        result = xp.vstack(outputs)

        if not return_logits:
            loss_activation = getattr(self.loss, "activation", None)
            if loss_activation is not None and hasattr(loss_activation, "forward"):
                act_out = loss_activation.forward(result, training=False)
                result = act_out if act_out is not None else getattr(loss_activation, "output", result)

        return result