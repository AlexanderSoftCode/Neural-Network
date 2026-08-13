from aether.optimizers.adam import Adam
from tests.optimizers.adam_base_suite import make_suite, backends_to_test

TARGET_LAYER = Adam

for backend in backends_to_test:
    suite_cls = make_suite(backend_name=backend, Optimizer_Class=TARGET_LAYER)
    class_name = f"Test_{TARGET_LAYER.__name__}_{backend.upper()}"

    suite_cls.__name__ = class_name
    suite_cls.__qualname__ = class_name
    suite_cls.__module__ = __name__  # Bind module ownership strictly to this file

    globals()[class_name] = suite_cls

# Clean up the loop variable to prevent unittest from discovering it as a duplicate test case
del suite_cls