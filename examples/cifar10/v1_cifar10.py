import aether.config as config 
import numpy as np
import cupy as cp
import aether as ae

from pathlib import Path

TRAIN_DIR = Path("data") / "cifar-10" / "cifar-10_train.npz"
TEST_DIR  = Path("data") / "cifar-10" / "cifar-10_test.npz"

# Using cp.load saves it as a cupy array
with cp.load(TRAIN_DIR, allow_pickle=False) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]

with cp.load(TEST_DIR, allow_pickle=False) as data:
    X_test = data["X_test"]
    y_test = data["y_test"]

TARGET_DEVICE = "cupy"
feature_pipeline = ae.Compose([
    ae.ToTensor(dtype='float32', target_device=TARGET_DEVICE),
    ae.Rescale(factor=1.0 / 255.0),
    ae.StandardScaler()
]).fit(X_train)

# 2. Transform train and test features seamlessly
X_train_tensor = feature_pipeline(X_train)
X_test_tensor = feature_pipeline(X_test)

# 3. Convert target labels
y_train_tensor, y_test_tensor = ae.to_tensor(
    y_train, y_test, target_device=TARGET_DEVICE, preserve_integers=True
)

print(f"{X_train.shape=}, {X_train.dtype=}, {type(X_train)=}")
print(f"{y_train.shape=}, {y_train.dtype=}, {type(y_train)}")
print(f"{X_train_tensor.shape=}, {X_train_tensor.dtype=}, {type(X_train_tensor)=}")
print(f"{y_train_tensor.shape=}, {y_train_tensor.dtype=}, {type(y_train_tensor)}")

SEED = 42
model = ae.Model()

"""
model.add(ae.Flatten())
model.add(ae.Dense(32*32*3, 1024))
model.add(ae.BatchNorm(n_features=1024))
model.add(ae.ReLU())
model.add(ae.Dense(1024, 256))
model.add(ae.LeakyReLU(alpha=0.01))
model.add(ae.Dropout(rate=0.05, seed=SEED))
model.add(ae.Dense(256, 10))
"""

model.manual_seed(seed=42)
# --- Block 1: Feature Extraction ---
model.add(ae.Conv2d(3, 32, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm())                                   # 1. Normalize before activation
model.add(ae.ReLU())                                        # 2. Non-linearity
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))    # 3. Spatial downsampling

# --- Block 2: Higher-Level Representations ---
model.add(ae.Conv2d(32, 64, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm())                                   # 1. Normalize before activation
model.add(ae.LeakyReLU(alpha=0.01))                         # 2. Non-linearity
model.add(ae.SpatialDropout(rate=0.05, seed=42))            # 3. Regularization on active feature maps

# --- Block 3: Classifier Head ---
model.add(ae.GlobalAvgPool())                               # Pool spatially: (Batch, H, W, 64) -> (Batch, 64)
model.add(ae.Dense(64, 10))

# --- Model Configuration ---
model.configure(
    loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.01),
    optimizer=ae.Adam(learning_rate=0.001, decay=5e-5),
    accuracy=ae.CategoricalAccuracy()
)

#model.set_precision("float16")
model.to('cupy')
model.set_precision(compute_dtype="float16")
#the could also write model.set_precision("blfoat16")
model.finalize(input_shape=(32,32,3))
model.train(
    X=X_train_tensor, 
    y=y_train_tensor, 
    epochs=5, 
    batch_size=128,
    print_every=100, 
    validation_data=(X_test_tensor, y_test_tensor)
)