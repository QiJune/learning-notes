#include "adam.h"

extern "C" {
void c_adam(float beta1, float beta2, float epsilon, float beta1_pow,
            float beta2_pow, float *mom1, float *mom1_out, float *mom2,
            float *mom2_out, float lr, float *grad, float *param,
            float *param_out, int numel) {
  Adam(beta1, beta2, epsilon, beta1_pow, beta2_pow, mom1, mom1_out, mom2,
       mom2_out, lr, grad, param, param_out, numel);
}
}