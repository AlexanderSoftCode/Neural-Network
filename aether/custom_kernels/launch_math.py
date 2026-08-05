import numpy as np


def _compute_magic_numbers(d: int, max_numerator: int):
    """Derives (scale, shift) analytically in O(1) time.
    
    Guarantees zero off-by-one errors across [0, max_numerator] by 
    evaluating the worst-case remainder candidate (n_bad) directly.
    """
    if d == 1:
        # (n * 1) >> (32 + (-32))  ==  (n * 1) >> 0  ==  n
        return np.uint32(1), np.int32(-32)

    # Find n_bad: the largest integer <= max_numerator where n % d == d - 1
    # This is the exact value in [0, max_numerator] most vulnerable to rounding overflow.
    rem = (max_numerator + 1) % d
    n_bad = max_numerator - rem

    p = 0
    while True:
        two_pow_32_p = 1 << (32 + p)
        scale = (two_pow_32_p + d - 1) // d  # exact ceil(2^(32+p) / d)

        if scale < (1 << 32):
            # If n_bad < 0, all numbers in range are < d, so division is always 0
            if n_bad < 0:
                break
            
            # O(1) Worst-Case Verification
            approx = (n_bad * scale) >> (32 + p)
            expected = n_bad // d
            
            if approx == expected:
                break

        p += 1
        if p > 32:
            raise RuntimeError(f"Cannot compute 32-bit magic number for divisor {d}")

    return np.uint32(scale), np.uint32(p)