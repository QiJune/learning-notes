#include "adam.h"
#include <Eigen/Dense>
using namespace Eigen;

void Adam(float beta1, float beta2, float epsilon, float beta1_pow,
          float beta2_pow, float *mom1, float *mom1_out, float *mom2,
          float *mom2_out, float lr, float *grad, float *param,
          float *param_out, int numel) {
  Eigen::Map<const Eigen::Array<float, 1, Eigen::Dynamic>> eg{
      grad, static_cast<Eigen::Index>(numel)};
  Eigen::Map<const Eigen::Array<float, 1, Eigen::Dynamic>> emom1{
      mom1, static_cast<Eigen::Index>(numel)};
  Eigen::Map<const Eigen::Array<float, 1, Eigen::Dynamic>> emom2{
      mom2, static_cast<Eigen::Index>(numel)};
  Eigen::Map<const Eigen::Array<float, 1, Eigen::Dynamic>> eparam{
      param, static_cast<Eigen::Index>(numel)};

  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> eparam_out{
      param_out, static_cast<Eigen::Index>(numel)};
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> emoment1_out{
      mom1_out, static_cast<Eigen::Index>(numel)};
  Eigen::Map<Eigen::Array<float, 1, Eigen::Dynamic>> emoment2_out{
      mom2_out, static_cast<Eigen::Index>(numel)};

  // Calculation
  lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

  emoment1_out = beta1 * emom1 + (1 - beta1) * eg;
  emoment2_out = beta2 * emom2 + (1 - beta2) * eg * eg;
  eparam_out = eparam - lr * (emoment1_out / (emoment2_out.sqrt() + epsilon));
}