import numpy as np 
import aether.config as config

class Layer:
    def __init__(self, seed=None): 
        self.seed = seed

    def _compile_for_device(self, device):
        """
        Hook for layers utilizing fused modules to 
        re-bind runtime pointers
        """
        pass

    def to(self, device):
        """
        Polymorphic state migration engine.
        Each layer isolates its own hardware allocations.
        """
        # Trigger compile/pointer swaps if specific layer overrides it
        self._compile_for_device(device)

        # Locate and migrate any tracking arrays/tensors
        # This catches, weights, biases, or future states dynamically
        for attr_name, attr_value in self.__dict__.items():
            # If attribute is a tensor, shift its VRAM boundary safely
            if hasattr(attr_value, 'shape'):
                setattr(self, attr_name, config.to_device(attr_value, target=device))
    @staticmethod
    def _derive_stream_seed(base_seed, stream_id):
        """
        Derives a deterministic 64-bit seed for a sepcific random stream. 
        Passing in the same `base_seed` and `stream_id` guarantees identical output
        across runs. 
        """
        if base_seed is None: 
            entropy = None
            spawn_key = (int(stream_id),)
        else:
            entropy = [int(base_seed), int(stream_id)]
            spawn_key = ()
        
        seed_seq = np.random.SeedSequence(entropy, spawn_key=spawn_key)
        return int(seed_seq.generate_state(1, dtype=np.uint64)[0])