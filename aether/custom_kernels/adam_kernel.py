import aether.config as config

_adam_update_kernel = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T param, T grad, T m, T v, T lr, T beta1, T beta2, T eps, '
    'T bias_correction1, T bias_correction2, T weight_decay',
    'T out_param, T out_m, T out_v',
    '''
    out_param = param;
    if (weight_decay > (T)0.0) {
        out_param -= lr * weight_decay * out_param;
    }
    out_m = beta1 * m + ((T)1.0 - beta1) * grad;
    out_v = beta2 * v + ((T)1.0 - beta2) * (grad * grad);
    T m_hat = out_m / bias_correction1;
    T v_hat = out_v / bias_correction2;
    out_param -= lr * m_hat / (sqrt(v_hat) + eps);
    ''',
    'adam_update_kernel',
    ),
    name='_adam_update_kernel'
)