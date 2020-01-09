#include "adam.h"
#include <chrono>
#include <iostream>
#include <random>

int main() {
  std::default_random_engine e;
  std::uniform_real_distribution<> dist(0, 1);

  size_t numel = 1000;
  float beta1 = 0.1;
  float beta2 = 0.2;
  float epsilon = 0.01;
  float beta1_pow = 0.3;
  float beta2_pow = 0.4;
  float *mom1 = new float[numel];
  float *mom1_out = mom1;
  float *mom2 = new float[numel];
  float *mom2_out = mom2;
  float lr = 0.1;
  float *grad = new float[numel];
  float *param = new float[numel];
  float *param_out = param;

  for (size_t i = 0; i < numel; i++) {
    mom1[i] = dist(e);
    mom2[i] = dist(e);
    grad[i] = dist(e);
    param[i] = dist(e);
  }

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 1000000; i++) {
    Adam(beta1, beta2, epsilon, beta1_pow, beta2_pow, mom1, mom1_out, mom2,
         mom2_out, lr, grad, param, param_out, numel);
  }
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::cout << "Time comsumd: "
            << double(duration) * std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << " s" << std::endl;

  delete[] mom1;
  delete[] mom2;
  delete[] grad;
  delete[] param;
}
