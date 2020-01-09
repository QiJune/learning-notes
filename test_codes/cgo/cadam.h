#ifdef __cplusplus
extern "C" {
#endif

void c_adam(float beta1, float beta2, float epsilon, float beta1_pow,
            float beta2_pow, float *mom1, float *mom1_out, float *mom2,
            float *mom2_out, float lr, float *grad, float *param,
            float *param_out, int numel);

#ifdef __cplusplus
}
#endif