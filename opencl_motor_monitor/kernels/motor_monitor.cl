// OpenCL C 1.2. float4 lanes: temperature, current, vibration, RPM.
// Features are window means, except vibration uses root mean square.
__constant float NOMINAL_RPM = 1800.0f;

float4 finish_features(float4 sum, float vibration_energy, int window) {
    float4 feature = sum / (float)window;
    feature.z = sqrt(vibration_energy / (float)window);
    return feature;
}

__kernel void features_naive(__global const float4* input,
                             __global float4* features,
                             int count, int radius) {
    const int i = (int)get_global_id(0);
    if (i >= count) return;
    float4 sum = (float4)(0.0f);
    float energy = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float4 value = input[clamp(i + offset, 0, count - 1)];
        sum += value;
        energy += value.z * value.z;
    }
    features[i] = finish_features(sum, energy, radius * 2 + 1);
}

__kernel void features_tiled(__global const float4* input,
                             __global float4* features,
                             int count, int radius, __local float4* tile) {
    const int lane = (int)get_local_id(0);
    const int group_size = (int)get_local_size(0);
    const int base = (int)get_group_id(0) * group_size - radius;
    for (int j = lane; j < group_size + 2 * radius; j += group_size) {
        tile[j] = input[clamp(base + j, 0, count - 1)];
    }
    // Padded work-items must reach this barrier before the bounds check.
    barrier(CLK_LOCAL_MEM_FENCE);
    const int i = (int)get_global_id(0);
    if (i >= count) return;
    float4 sum = (float4)(0.0f);
    float energy = 0.0f;
    for (int j = 0; j <= 2 * radius; ++j) {
        const float4 value = tile[lane + j];
        sum += value;
        energy += value.z * value.z;
    }
    features[i] = finish_features(sum, energy, radius * 2 + 1);
}

__kernel void classify_faults(__global const float4* features,
                              __global float4* scores,
                              __global uint* flags,
                              __constant float4* limits, int count) {
    const size_t i = get_global_id(0);
    if (i >= (size_t)count) return;
    float4 observed = features[i];
    observed.w = fabs(observed.w - NOMINAL_RPM);
    const float4 score = observed / limits[0];
    const int4 abnormal = score > (float4)(1.0f);
    scores[i] = score;
    flags[i] = (abnormal.x ? 1u : 0u) | (abnormal.y ? 2u : 0u) |
               (abnormal.z ? 4u : 0u) | (abnormal.w ? 8u : 0u);
}

__kernel void reduce_faults(__global const uint* flags,
                            __global const float4* scores,
                            __global uint4* partial_counts,
                            __global float* partial_peaks, int count,
                            __local uint4* counts, __local float* peaks) {
    const size_t i = get_global_id(0);
    const size_t lane = get_local_id(0);
    const uint mask = i < (size_t)count ? flags[i] : 0u;
    const float4 value = i < (size_t)count ? scores[i] : (float4)(0.0f);
    counts[lane] = (uint4)(mask & 1u, (mask >> 1) & 1u,
                           (mask >> 2) & 1u, (mask >> 3) & 1u);
    peaks[lane] = fmax(fmax(value.x, value.y), fmax(value.z, value.w));
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t step = get_local_size(0) / 2; step > 0; step /= 2) {
        if (lane < step) {
            counts[lane] += counts[lane + step];
            peaks[lane] = fmax(peaks[lane], peaks[lane + step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        partial_counts[get_group_id(0)] = counts[0];
        partial_peaks[get_group_id(0)] = peaks[0];
    }
}
