#pragma once
void Adam(float beta1, float beta2, float epsilon, float beta1_pow,
          float beta2_pow, float *mom1, float *mom1_out, float *mom2,
          float *mom2_out, float lr, float *grad, float *param,
          float *param_out, int numel);
