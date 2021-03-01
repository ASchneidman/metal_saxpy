//
//  saxpy.metal
//  saxpy
//
//  Created by Alex Schneidman on 2/28/21.
//

#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device const float *x,
                  device const float *y,
                  device float *result,
                  device uint *N,
                  device float *alpha,
                  uint index [[thread_position_in_grid]])
{
    if (index < *N)
       result[index] = *alpha * x[index] + y[index];
}
