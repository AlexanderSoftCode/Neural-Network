import json
import zipfile
import numpy as np

import aether.config as config
import aether.layers as layer_module

from aether.base import Layer
from aether.losses import Loss
from aether.metrics import Accuracy
from aether.optimizers import Optimizer
from aether._utils import NullAccuracy, NullOptimizer
from aether._utils.progress import make_progress
try:
    import safetensors.numpy
except ImportError:
    safetensors = None

class Model():
    def __init__(self):
        self.layers = []
        self.is_finalized = False
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self._seed = None
        self._predict_activation = None

    def add(self, layer):
        if self.is_finalized:   
            raise RuntimeError("Cannot modify model after finalize() has been called.")
        
        if not isinstance(layer, Layer):
            raise TypeError(
                f"Expected an instance of 'Layer', but got `{type(layer).__name__}`."
                "Make sure the layer you are passing in inherits from aether.base.Layer"
            )
        self.layers.append(layer)

    def manual_seed(self, seed: int):
        if self.is_finalized:
            raise RuntimeError("Cannot set a new seed after finalize() has been called.")
        self._seed = int(seed)
        return self
    
    def configure(self, loss=None, optimizer=None, accuracy=None):
        """Configures the training components (loss, optimizer, metrics) for the model.
        Utilizes strict type checking for loss, optimizer, and accuracy. 
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
                    "Make sure the loss your are passing in inherits from aether.losses.loss"
                )
            else:
                self.loss = loss

        if optimizer is not None:
            if not isinstance(optimizer, Optimizer):
                raise TypeError(
                    f"Expected an instance of 'Optimizer', but got '{type(optimizer).__name__}'. "
                    "Make sure the optimizer you are passing in inherits from aether.optimizers.Optimizer."
                )
            else:
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
        kernels if user is using cupy. As only trainable layers hold tensors and are
        accessed during the loop, the layers are updated in place, meaning its possible
        to use this after training for say inference.

        Args:
            device (str): The target hardware execution device 
            ('cupy' or 'numpy)
        Raises:
            ValueError: If `device` specifies an unsupported or unconfigured backend
        Example:
            >>> model = ae.Model()
            ... model.add(...)
            ... model.add(...)
            ... model.to("cupy")
            ... # Now we can train and evaluate in cupy backend
            ... model.configure
            ... model.finalize(input_shape=(...))
            ... model.train(X, y, epochs=5, shuffle=True, evaluat)
            ... model.evaluate(X_val, y_val)
            ... model.to("numpy")
            ... # We can perform inference on the CPU now
            ... preds = model.predict(X_cpu)
        """
        target_device = device.lower()
        config.set_backend(target_device)
        self.device = target_device
        self._sync_device(target_device=target_device)

        if self.is_finalized:
            for layer in self.trainable_layers:
                param_dict = layer.get_parameters()
                migrated_dict = {}

                for name, tensor in param_dict.items():
                    if tensor is not None:
                        migrated_dict[name] = config.to_device(tensor, target=target_device)

                layer.set_parameters(**migrated_dict)

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

    def finalize(self, input_shape: tuple[int, ...]):
        """
        Finalizes the model architecture, propagating tensor dimensions
        and allocating weights/buffers across all registered layers.
        """
        if self.is_finalized:   
            raise RuntimeError("Cannot modify model after finalize() has been called.")

        if not self.layers:
            raise RuntimeError("[aether] Cannot finalize an empty model. Please add layers via Model.add() first.")

        current_shape = input_shape
        
        for idx, layer in enumerate(self.layers):
            layer_seed = (self._seed + idx) if self._seed is not None else None
            
            # Unconditionally build and propagate shape
            try:
                current_shape = layer.build(current_shape, seed=layer_seed)
            except TypeError:
                current_shape = layer.build(current_shape)

            # Stochastic layers (e.g. Dropout RNG stream) rebind if needed
            if hasattr(layer, "_set_seed"):
                if layer.seed is None and layer_seed is not None:
                    layer._set_seed(layer_seed)

        self.trainable_layers = [
            layer for layer in self.layers
            if getattr(layer, "weights", None) is not None or getattr(layer, "biases", None) is not None
        ]

        if self.loss is not None:
            self.loss.validate_graph(layers=self.layers)
            self.loss.remember_trainable_layers(self.trainable_layers)

            if self.loss.has_fused_activation:
                self._predict_activation = self.loss.activation

        if self.optimizer is not None:
            if not hasattr(self.optimizer, "step") or not hasattr(self.optimizer, "init_params"):
                raise TypeError(
                    f"Optimizer '{type(self.optimizer).__name__}' must implement "
                    "'init_params(trainable_layers)' and 'step()'."
                )
            self.optimizer.init_params(self.trainable_layers)
        else:
            self.optimizer = NullOptimizer()

        self._step_optimizer = self.optimizer.step

        if self.accuracy is None:
            self.accuracy = NullAccuracy()
        self._sync_device()
        self.is_finalized = True

    def _assert_device_alignment(self, X, y=None):
        """Ensure incoming input tensors strictly match the configured hardware backend."""
        expected_backend = getattr(self, "device", "numpy")
        
        is_x_cupy = type(X).__module__.startswith("cupy")
        is_y_cupy = type(y).__module__.startswith("cupy") if y is not None else is_x_cupy

        if expected_backend == "cupy":
            if not is_x_cupy or not is_y_cupy:
                raise TypeError(
                    f"[aether] Device mismatch: Model is configured for CuPy backend ('cupy'), "
                    f"but received input tensors on host (X: {type(X).__name__}, y: {type(y).__name__ if y is not None else 'None'}). "
                    f"Transfer your arrays using cp.asarray() before calling train/evaluate/predict."
                )
        else:  # numpy
            if is_x_cupy or is_y_cupy:
                raise TypeError(
                    f"[aether] Device mismatch: Model is configured for NumPy backend ('numpy'), "
                    f"but received CuPy GPU tensors. Transfer your data to host memory via cp.asnumpy() "
                    f"or migrate the model first via model.to('cupy')."
                )

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
        shuffle=True,
        print_every=1,
        verbose = 1,
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
            shuffle (bool, optional): If True (default), draws a fresh sample-index
                permutation every epoch on the active device backend and gathers
                mini-batches from it via fancy indexing, so sample order differs
                epoch to epoch without ever copying or reordering the full X/y
                arrays. If False, mini-batches are taken as contiguous, zero-copy
                slices in dataset order every epoch. Never applied to
                `validation_data` -- evaluation order doesn't affect the metric.
            print_every (int): Step interval frequency for logging telemetry.
            verbose (int, optional): Verbosity mode for training telemetry.
                - `0`: Silent mode (no output printed).
                - `1`: Dynamic graphical progress bar with metrics (default).
                - `2`: Plain text summary per epoch without carriage returns/ANSI escapes.
            Defaults to 1. 
            validation_data (tuple, optional): (X_val, y_val) tuple for out-of-sample testing.
        """
        if not self.is_finalized:
            raise RuntimeError(
                "[aether] Model must be explicitly finalized before training. "
                "Call model.finalize(input_shape) first."
            )
        if self.loss is None:
            raise RuntimeError(
                "[aether] Cannot train a model without a loss function."
                "Pass in a valid loss function to model.configure(loss=...) before finalize & train"
            )
        self._assert_device_alignment(X, y)
        if validation_data is not None:
            self._assert_device_alignment(validation_data[0], validation_data[1])

        xp = config.get_array_module(X)

        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        train_steps = (num_samples + effective_batch_size - 1) // effective_batch_size

        batch_slices = [
            (step * effective_batch_size, min((step + 1) * effective_batch_size, num_samples))
            for step in range(train_steps)
        ]

        if shuffle:
            def get_batch(epoch_indices, start_idx, end_idx):
                batch_idx = epoch_indices[start_idx:end_idx]
                return X[batch_idx], y[batch_idx]
        else:
            def get_batch(epoch_indices, start_idx, end_idx):
                return X[start_idx:end_idx], y[start_idx:end_idx]

        forward_fn = self.forward
        backward_fn = self.backward
        loss = self.loss
        accuracy = self.accuracy
        optimizer_obj = self.optimizer
        step_optimizer = self._step_optimizer

        has_reg = hasattr(loss, "regularization_loss") and any(
            getattr(layer, "weight_regularizer_l1", 0.0) > 0.0
            or getattr(layer, "weight_regularizer_l2", 0.0) > 0.0
            or getattr(layer, "bias_regularizer_l1", 0.0) > 0.0
            or getattr(layer, "bias_regularizer_l2", 0.0) > 0.0
            for layer in getattr(self, "trainable_layers", [])
        )
        reg_loss_fn = loss.regularization_loss if has_reg else None

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

        get_lr = lambda: getattr(optimizer_obj, "current_learning_rate", getattr(optimizer_obj, "lr", None))

        progress = make_progress(verbose, train_steps, epochs, has_reg)

        for epoch in range(1, epochs + 1):
            loss.new_pass()
            accuracy.new_pass()
            progress.start_epoch(epoch)

            epoch_indices = xp.random.permutation(num_samples) if shuffle else None

            for step, (start_idx, end_idx) in enumerate(batch_slices):
                batch_X, batch_y = get_batch(epoch_indices, start_idx, end_idx)

                data_loss, reg_loss, acc_val = run_step(batch_X, batch_y)

                # Bar-only tick: pure host-side math, zero GPU sync.
                progress.tick(step + 1)

                # Telemetry: GPU-to-Host sync barrier strictly confined to log steps.
                if print_every and (step % print_every == 0 or step == train_steps - 1):
                    s_data_loss = float(data_loss)
                    s_reg_loss = float(reg_loss)
                    s_acc = float(acc_val)
                    lr = get_lr()

                    progress.update_metrics(
                        step + 1,
                        s_data_loss + s_reg_loss,
                        s_acc,
                        float(lr) if lr is not None else 0.0,
                        reg_loss=s_reg_loss if has_reg else None,
                    )

            accum = loss.calculate_accumulated(include_regularization=has_reg)
            epoch_loss = float(accum[0] + accum[1]) if isinstance(accum, tuple) else float(accum)
            epoch_acc = float(accuracy.calculate_accumulated())

            lr = get_lr()
            progress.commit_epoch(epoch, epoch_loss, epoch_acc, float(lr) if lr is not None else 0.0)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
                progress.commit_validation(val_loss, val_acc)

        progress.close()

    def evaluate(self, X_val, y_val, *, batch_size=None, verbose=1):
        """
        Evaluate the model's loss and metrics on validation/test data in inference mode.

        Args:
            X_val (ndarray): Evaluation input features (NumPy or CuPy ndarray).
            y_val (ndarray): Ground truth labels or targets.
            batch_size (int, optional): Mini-batch size. Defaults to None (full-batch).
            verbose (int | bool, optional): Verbosity mode for evaluation.
                - `0` / `False`: Silent mode (no output printed).
                - `1` / `True`: Prints the final evaluation loss and accuracy metrics.
                Defaults to 1.
        Returns:
            tuple: (val_loss, val_acc) representing the computed validation metrics.
        Raises:
            RuntimeError: If model has not been explicitly finalized before calling evaluate().
        """
        if not self.is_finalized:
            raise RuntimeError(
                "[aether] Model must be explicitly finalized before evaluation. "
                "Call model.finalize(input_shape) first."
            )
        self._assert_device_alignment(X_val, y_val)
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

    def predict(self, X, *, batch_size=None, return_logits=False, stream_to_host=True):
        """
        Run batch inference on inputs and return model predictions.

        Automatically routes raw outputs through any fused loss activation
        (e.g., SoftMax in SoftmaxCategoricalCrossEntropy) unless raw logits are requested.

        Args:
            X (ndarray): Input feature batch (NumPy or CuPy array).
            batch_size (int, optional): Mini-batch sample size. Defaults to None (full-batch).
            return_logits (bool, optional): If True, returns raw network logits
                even if an output activation was fused into the loss. Defaults to False.
            stream_to_host (bool, optional): If True, incrementally transfers batch
                outputs to host memory (NumPy) to minimize GPU VRAM consumption.
                Defaults to True.

        Returns:
            ndarray: Model predictions (probabilities or stacked batch outputs).

        Raises:
            RuntimeError: If model has not been explicitly finalized before calling predict().
        """
        if not self.is_finalized:
            raise RuntimeError(
                "[aether] Model must be explicitly finalized before prediction. "
                "Call model.finalize(input_shape) first."
            )

        xp = config.get_array_module(X)
        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        prediction_steps = (num_samples + effective_batch_size - 1) // effective_batch_size

        forward_fn = self.forward
        activation = self._predict_activation
        apply_activation = (not return_logits) and (activation is not None)

        output_module = np if stream_to_host else xp
        output_buffer = None

        for step in range(prediction_steps):
            start_idx = step * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, num_samples)

            batch_output = forward_fn(X[start_idx:end_idx], training=False)
            if apply_activation:
                act_out = activation.forward(batch_output, training=False)
                batch_output = act_out if act_out is not None else activation.output

            if output_buffer is None:
                output_buffer = output_module.empty(
                    (num_samples,) + batch_output.shape[1:], dtype=batch_output.dtype
                )

            output_buffer[start_idx:end_idx] = (
                config.to_device(batch_output, target="numpy") if stream_to_host else batch_output
            )

        return output_buffer

    def save(self, filepath: str):
        """
        Saves the model architecture and parameters to a single .aether archive.

        Args:
            filepath (str): Path to output file (e.g. 'cifar10.aether').
        Raises:
            RuntimeError: If model is not finalized.
            ImportError: If safetensors is not installed.
        """
        if not self.is_finalized:
            raise RuntimeError(
                "[aether] Cannot save an unfinalized model. "
                "Call model.finalize(input_shape) first."
            )

        if safetensors is None:
            raise ImportError(
                "[aether] 'safetensors' is required for model serialization. "
                "Install it with `pip install safetensors`."
            )

        # Capture Top-Level Config & Layer Manifest
        input_shape = self.layers[0].input_shape
        precision_str = (
            self.precision_policy.compute_dtype_name
            if getattr(self, "precision_policy", None)
            else None
        )

        arch_manifest = {
            "schema_version": "1.0",
            "input_shape": list(input_shape),
            "seed": self._seed,
            "precision_policy": precision_str,
            "layers": [
                {
                    "index": idx,
                    "class_name": type(layer).__name__,
                    "config": layer.get_config(),
                }
                for idx, layer in enumerate(self.layers)
            ],
        }

        weight_dict = {}
        for idx, layer in enumerate(self.layers):
            for name, tensor in layer.get_parameters().items():
                if tensor is None:
                    continue
                cpu_array = config.to_device(tensor, target="numpy")
                weight_dict[f"{idx}.{name}"] = np.ascontiguousarray(cpu_array)

        weights_bytes = safetensors.numpy.save(weight_dict)
        arch_json_bytes = json.dumps(arch_manifest, indent=2).encode("utf-8")

        with zipfile.ZipFile(filepath, "w") as zipf:
            zipf.writestr(
                "architecture.json", arch_json_bytes, compress_type=zipfile.ZIP_DEFLATED
            )
            zipf.writestr(
                "weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED
            )

    @classmethod
    def load(cls, filepath: str, device: str | None = None) -> "Model":
        """Loads a model architecture and parameters from an .aether zip archive.

        Args:
            filepath (str): Path to the saved .aether model file.
            device (str, optional): Target hardware device ('numpy' or 'cupy').
                Defaults to None (uses current active backend).

        Returns:
            Model: A finalized and initialized Model instance.
        """
        if safetensors is None:
            raise ImportError(
                "[aether] 'safetensors' is required to load models. "
                "Install it with `pip install safetensors`."
            )

        active_backend = "cupy" if getattr(config.xp, "__name__", "") == "cupy" else "numpy"
        target_device = device if device is not None else active_backend

        with zipfile.ZipFile(filepath, "r") as zipf:
            arch_bytes = zipf.read("architecture.json")
            weights_bytes = zipf.read("weights.safetensors")

        manifest = json.loads(arch_bytes.decode("utf-8"))
        raw_weights = safetensors.numpy.load(weights_bytes)

        model = cls()

        if manifest.get("seed") is not None:
            model.manual_seed(manifest["seed"])

        for layer_entry in manifest.get("layers", []):
            class_name = layer_entry["class_name"]
            cfg = layer_entry.get("config", {})

            sanitized_cfg = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in cfg.items()
            }

            # Resolved on demand via layer_module.__getattr__
            layer_cls = getattr(layer_module, class_name, None)
            if layer_cls is None:
                raise ValueError(
                    f"[aether] Unknown layer class '{class_name}' found in saved manifest."
                )

            model.add(layer_cls(**sanitized_cfg))

        model.to(target_device)

        input_shape = tuple(manifest["input_shape"])
        model.finalize(input_shape=input_shape)

        layer_param_map: dict[int, dict] = {}
        for flat_key, array in raw_weights.items():
            if "." not in flat_key:
                continue
            idx_str, param_name = flat_key.split(".", 1)
            idx = int(idx_str)
            if idx not in layer_param_map:
                layer_param_map[idx] = {}

            # Route arrays directly to the resolved target device
            layer_param_map[idx][param_name] = config.to_device(
                array, target=target_device
            )

        for idx, params in layer_param_map.items():
            if idx < len(model.layers):
                model.layers[idx].set_parameters(**params)

        if manifest.get("precision_policy") is not None:
            model.set_precision(manifest["precision_policy"])

        return model