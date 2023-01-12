#include "vecadd.h"
#include <stdio.h>
#include <vector>

int main() {
  int count = 100000;
  std::vector<float> h_A(count, 1);
  std::vector<float> h_B(count, 2);
  std::vector<float> h_C(count);

  vecAdd(h_A.data(), h_B.data(), h_C.data(), count);
  for (size_t i = 0; i < 100; ++i) {
    printf("%.0f ", h_C[i]);
  }

  return 0;
}
