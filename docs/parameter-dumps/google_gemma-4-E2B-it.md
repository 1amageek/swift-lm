# Parameter Dump: b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf

- **Path**: `/Users/1amageek/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`
- **Files**: model.safetensors
- **Total parameters**: 5,123,178,979 (5.123B)
- **Tensor count**: 2011

| # | Name | Shape | Dtype | Norm | Mean | Std | Min | Max | First 8 |
|---|------|-------|-------|------|------|-----|-----|-----|---------|
| 0 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.81 | 12.81 | 0 | 12.81 | 12.81 | 12.81 |
| 1 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.88 | -12.88 | 0 | -12.88 | -12.88 | -12.88 |
| 2 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.32 | -0.007862 | 0.02838 | -0.0918 | 0.0459 | -0.06689, 0, -0.03345, -0.06689, 0, -0.06689, -0.03345, 0.03345 |
| 3 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 39.75 | 39.75 | 0 | 39.75 | 39.75 | 39.75 |
| 4 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 40 | -40 | 0 | -40 | -40 | -40 |
| 5 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 11.06 | 11.06 | 0 | 11.06 | 11.06 | 11.06 |
| 6 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 11.19 | -11.19 | 0 | -11.19 | -11.19 | -11.19 |
| 7 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.21 | 0.0001733 | 0.01474 | -0.07471 | 0.03735 | 0.01648, 0.01648, -0.01648, 0.01648, 0, -0.03296, -0.01648, 0.01648 |
| 8 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 34 | 34 | 0 | 34 | 34 | 34 |
| 9 | `model.audio_tower.layers.0.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 34.25 | -34.25 | 0 | -34.25 | -34.25 | -34.25 |
| 10 | `model.audio_tower.layers.0.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 181.7 | 3.326 | 4.606 | -10 | 30.75 | 19.62, 1.445, 1.047, 10.88, 2.062, 1.055, 2.797, 1.516 |
| 11 | `model.audio_tower.layers.0.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 104.3 | 2.734 | 1.775 | -2.359 | 9.188 | 4.531, 2.031, 1.898, 3.391, 5.75, 2, 5.281, 3.016 |
| 12 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 11.06 | 11.06 | 0 | 11.06 | 11.06 | 11.06 |
| 13 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 11.19 | -11.19 | 0 | -11.19 | -11.19 | -11.19 |
| 14 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.56 | -0.008331 | 0.02836 | -0.09619 | 0.0481 | -0.03149, 0.03149, 0, 0.03149, 0.03149, -0.03149, -0.03149, 0 |
| 15 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 26.12 | 26.12 | 0 | 26.12 | 26.12 | 26.12 |
| 16 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 26.38 | -26.38 | 0 | -26.38 | -26.38 | -26.38 |
| 17 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 9.375 | 9.375 | 0 | 9.375 | 9.375 | 9.375 |
| 18 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 9.438 | -9.438 | 0 | -9.438 | -9.438 | -9.438 |
| 19 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.78 | 2.178e-05 | 0.01454 | -0.1167 | 0.05835 | 0, 0, 0.0376, 0.0376, 0.0376, 0.0376, 0, 0.0376 |
| 20 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 33 | 33 | 0 | 33 | 33 | 33 |
| 21 | `model.audio_tower.layers.0.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 33.25 | -33.25 | 0 | -33.25 | -33.25 | -33.25 |
| 22 | `model.audio_tower.layers.0.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 311 | 7.323 | 6.391 | -29.75 | 31.5 | -0.4668, 3.797, 2.172, 17.25, 13.38, 1.992, 14.5, 3.125 |
| 23 | `model.audio_tower.layers.0.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 142 | 4.075 | 1.761 | -3.141 | 7.781 | 1.328, 5.219, 2.938, -3.141, 5.938, 3.25, 5.531, 3.766 |
| 24 | `model.audio_tower.layers.0.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 119.1 | 3.138 | 2.004 | -3.641 | 19.88 | 2.578, 3.062, 3.547, 3.531, 3.641, 2.766, 2.234, 2.281 |
| 25 | `model.audio_tower.layers.0.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.001402 | 0.4473 | -2.844 | 3.391 | -0.5, -0.3457, -0.4707, -0.2676, 0.459, -0.5781, -0.3809, -0.4395 |
| 26 | `model.audio_tower.layers.0.lconv1d.linear_end.input_max` |  | bfloat16 | 5.781 | 5.781 | 0 | 5.781 | 5.781 | 5.781 |
| 27 | `model.audio_tower.layers.0.lconv1d.linear_end.input_min` |  | bfloat16 | 5.812 | -5.812 | 0 | -5.812 | -5.812 | -5.812 |
| 28 | `model.audio_tower.layers.0.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 30.07 | -0.0003578 | 0.02936 | -0.1157 | 0.05786 | -0.03955, -0.03955, 0, 0.03955, 0.03955, -0.03955, 0.03955, 0 |
| 29 | `model.audio_tower.layers.0.lconv1d.linear_end.output_max` |  | bfloat16 | 6.188 | 6.188 | 0 | 6.188 | 6.188 | 6.188 |
| 30 | `model.audio_tower.layers.0.lconv1d.linear_end.output_min` |  | bfloat16 | 6.25 | -6.25 | 0 | -6.25 | -6.25 | -6.25 |
| 31 | `model.audio_tower.layers.0.lconv1d.linear_start.input_max` |  | bfloat16 | 32.25 | 32.25 | 0 | 32.25 | 32.25 | 32.25 |
| 32 | `model.audio_tower.layers.0.lconv1d.linear_start.input_min` |  | bfloat16 | 32.5 | -32.5 | 0 | -32.5 | -32.5 | -32.5 |
| 33 | `model.audio_tower.layers.0.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.94 | 0.001034 | 0.03102 | -0.1709 | 0.1494 | 0.01251, 0.0376, 0.01251, 0, -0.01251, -0.02502, 0.02502, 0.01251 |
| 34 | `model.audio_tower.layers.0.lconv1d.linear_start.output_max` |  | bfloat16 | 29.38 | 29.38 | 0 | 29.38 | 29.38 | 29.38 |
| 35 | `model.audio_tower.layers.0.lconv1d.linear_start.output_min` |  | bfloat16 | 29.62 | -29.62 | 0 | -29.62 | -29.62 | -29.62 |
| 36 | `model.audio_tower.layers.0.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 204.8 | 5.554 | 3.184 | -6.625 | 73 | 2.547, 6.656, 3.375, 3.328, 7.25, 2.578, 7.875, 7.281 |
| 37 | `model.audio_tower.layers.0.norm_out.weight` | 1024 | bfloat16 | 231.2 | 5.441 | 4.757 | 0.1289 | 23.12 | 0.5312, 7.531, 8.688, 1.125, 0.9375, 14.62, 1.258, 3.172 |
| 38 | `model.audio_tower.layers.0.norm_post_attn.weight` | 1024 | bfloat16 | 117.4 | 1.534 | 3.334 | -13.69 | 22.38 | 0.4609, 0.6602, 1.422, 0.2197, -0.3789, 0.02173, 2.141, 1.398 |
| 39 | `model.audio_tower.layers.0.norm_pre_attn.weight` | 1024 | bfloat16 | 111 | 2.967 | 1.798 | -0.9922 | 19.88 | 3.547, 4.031, 2.891, 2.344, 1.531, 0.8398, 3.047, 4.188 |
| 40 | `model.audio_tower.layers.0.self_attn.k_proj.input_max` |  | bfloat16 | 20.25 | 20.25 | 0 | 20.25 | 20.25 | 20.25 |
| 41 | `model.audio_tower.layers.0.self_attn.k_proj.input_min` |  | bfloat16 | 20.38 | -20.38 | 0 | -20.38 | -20.38 | -20.38 |
| 42 | `model.audio_tower.layers.0.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.73 | -0.002509 | 0.02893 | -0.1172 | 0.05859 | 0, 0, 0, -0.08154, 0.04077, 0, 0, 0 |
| 43 | `model.audio_tower.layers.0.self_attn.k_proj.output_max` |  | bfloat16 | 34.25 | 34.25 | 0 | 34.25 | 34.25 | 34.25 |
| 44 | `model.audio_tower.layers.0.self_attn.k_proj.output_min` |  | bfloat16 | 34.5 | -34.5 | 0 | -34.5 | -34.5 | -34.5 |
| 45 | `model.audio_tower.layers.0.self_attn.per_dim_scale` | 128 | bfloat16 | 38.3 | -3.338 | 0.5699 | -5.344 | -1.992 | -2.062, -3.406, -2.812, -3.984, -3.188, -3.938, -3.016, -3.547 |
| 46 | `model.audio_tower.layers.0.self_attn.post.input_max` |  | bfloat16 | 26.5 | 26.5 | 0 | 26.5 | 26.5 | 26.5 |
| 47 | `model.audio_tower.layers.0.self_attn.post.input_min` |  | bfloat16 | 26.62 | -26.62 | 0 | -26.62 | -26.62 | -26.62 |
| 48 | `model.audio_tower.layers.0.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 27.77 | 0.000267 | 0.02711 | -0.123 | 0.09229 | -0.02136, -0.02136, 0, -0.02136, -0.02136, 0.02136, 0, -0.02136 |
| 49 | `model.audio_tower.layers.0.self_attn.post.output_max` |  | bfloat16 | 100.5 | 100.5 | 0 | 100.5 | 100.5 | 100.5 |
| 50 | `model.audio_tower.layers.0.self_attn.post.output_min` |  | bfloat16 | 101.5 | -101.5 | 0 | -101.5 | -101.5 | -101.5 |
| 51 | `model.audio_tower.layers.0.self_attn.q_proj.input_max` |  | bfloat16 | 20.25 | 20.25 | 0 | 20.25 | 20.25 | 20.25 |
| 52 | `model.audio_tower.layers.0.self_attn.q_proj.input_min` |  | bfloat16 | 20.38 | -20.38 | 0 | -20.38 | -20.38 | -20.38 |
| 53 | `model.audio_tower.layers.0.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.82 | -0.003564 | 0.0289 | -0.1128 | 0.0564 | -0.0752, 0, 0, -0.0752, 0, -0.0376, -0.0376, 0.0376 |
| 54 | `model.audio_tower.layers.0.self_attn.q_proj.output_max` |  | bfloat16 | 34.25 | 34.25 | 0 | 34.25 | 34.25 | 34.25 |
| 55 | `model.audio_tower.layers.0.self_attn.q_proj.output_min` |  | bfloat16 | 34.5 | -34.5 | 0 | -34.5 | -34.5 | -34.5 |
| 56 | `model.audio_tower.layers.0.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.005568 | 0.03075 | -0.2461 | 0.2793 | -0.07178, -0.07031, -0.06738, -0.06494, -0.06104, -0.05835, -0.05469, -0.052 |
| 57 | `model.audio_tower.layers.0.self_attn.v_proj.input_max` |  | bfloat16 | 20.25 | 20.25 | 0 | 20.25 | 20.25 | 20.25 |
| 58 | `model.audio_tower.layers.0.self_attn.v_proj.input_min` |  | bfloat16 | 20.38 | -20.38 | 0 | -20.38 | -20.38 | -20.38 |
| 59 | `model.audio_tower.layers.0.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.4 | 0.0001138 | 0.02871 | -0.1572 | 0.07861 | -0.03296, -0.03296, 0, -0.03296, 0, -0.03296, -0.03296, 0 |
| 60 | `model.audio_tower.layers.0.self_attn.v_proj.output_max` |  | bfloat16 | 34.25 | 34.25 | 0 | 34.25 | 34.25 | 34.25 |
| 61 | `model.audio_tower.layers.0.self_attn.v_proj.output_min` |  | bfloat16 | 34.5 | -34.5 | 0 | -34.5 | -34.5 | -34.5 |
| 62 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.12 | 12.12 | 0 | 12.12 | 12.12 | 12.12 |
| 63 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.19 | -12.19 | 0 | -12.19 | -12.19 | -12.19 |
| 64 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.53 | -0.00773 | 0.02852 | -0.1021 | 0.05103 | -0.02734, -0.02734, -0.02734, -0.02734, -0.02734, 0, -0.02734, -0.02734 |
| 65 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 27.38 | 27.38 | 0 | 27.38 | 27.38 | 27.38 |
| 66 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 27.62 | -27.62 | 0 | -27.62 | -27.62 | -27.62 |
| 67 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 9.625 | 9.625 | 0 | 9.625 | 9.625 | 9.625 |
| 68 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 9.688 | -9.688 | 0 | -9.688 | -9.688 | -9.688 |
| 69 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.82 | 6.939e-05 | 0.01455 | -0.0957 | 0.04785 | 0, 0.01636, 0.01636, 0, 0.01636, -0.03271, 0.01636, 0 |
| 70 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 39.5 | 39.5 | 0 | 39.5 | 39.5 | 39.5 |
| 71 | `model.audio_tower.layers.1.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 39.75 | -39.75 | 0 | -39.75 | -39.75 | -39.75 |
| 72 | `model.audio_tower.layers.1.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 81.85 | 2.034 | 1.552 | -1.516 | 24 | 1.797, 1.828, 1.336, 2.25, 4.5, 1.078, 1.82, 0.5977 |
| 73 | `model.audio_tower.layers.1.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 476.1 | 11.98 | 8.819 | -29.75 | 52.75 | 21.88, 8.125, 4.469, 11.94, 26.38, 2.062, 18.5, 14.31 |
| 74 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 11.56 | 11.56 | 0 | 11.56 | 11.56 | 11.56 |
| 75 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 11.62 | -11.62 | 0 | -11.62 | -11.62 | -11.62 |
| 76 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.56 | -0.007613 | 0.02857 | -0.08643 | 0.04321 | -0.02905, 0, -0.02905, 0, -0.02905, 0, 0, -0.02905 |
| 77 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 27.5 | 27.5 | 0 | 27.5 | 27.5 | 27.5 |
| 78 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 27.62 | -27.62 | 0 | -27.62 | -27.62 | -27.62 |
| 79 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 9.562 | 9.562 | 0 | 9.562 | 9.562 | 9.562 |
| 80 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 9.688 | -9.688 | 0 | -9.688 | -9.688 | -9.688 |
| 81 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.69 | -0.0001414 | 0.01449 | -0.09961 | 0.0498 | 0.01636, 0, 0, 0.01636, 0, 0.01636, 0, 0.01636 |
| 82 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 37.75 | 37.75 | 0 | 37.75 | 37.75 | 37.75 |
| 83 | `model.audio_tower.layers.1.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 38.25 | -38.25 | 0 | -38.25 | -38.25 | -38.25 |
| 84 | `model.audio_tower.layers.1.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 194.1 | 5.115 | 3.261 | -8.25 | 48 | 5.219, 5.094, 3.281, 7.531, 6.844, 3.266, 3.953, 2.422 |
| 85 | `model.audio_tower.layers.1.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 164.7 | 4.894 | 1.59 | -6.375 | 9.375 | 3.5, 5.906, 3.141, -6.375, 4.219, 2.219, 6.562, 6.906 |
| 86 | `model.audio_tower.layers.1.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 149 | 4.183 | 2.049 | -2.75 | 49.25 | 5.969, 5.156, 5.625, 2.328, 4.094, 2.656, 5.125, 6.062 |
| 87 | `model.audio_tower.layers.1.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.0008113 | 0.4472 | -4.344 | 3.297 | -0.4395, -0.1621, -0.06592, 0.1108, 0.4316, -0.006287, -0.02185, 0.004669 |
| 88 | `model.audio_tower.layers.1.lconv1d.linear_end.input_max` |  | bfloat16 | 7.688 | 7.688 | 0 | 7.688 | 7.688 | 7.688 |
| 89 | `model.audio_tower.layers.1.lconv1d.linear_end.input_min` |  | bfloat16 | 7.75 | -7.75 | 0 | -7.75 | -7.75 | -7.75 |
| 90 | `model.audio_tower.layers.1.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.98 | 0.0005584 | 0.02928 | -0.1211 | 0.06055 | 0.03882, 0, 0, -0.03882, 0.03882, 0, 0, -0.03882 |
| 91 | `model.audio_tower.layers.1.lconv1d.linear_end.output_max` |  | bfloat16 | 8.062 | 8.062 | 0 | 8.062 | 8.062 | 8.062 |
| 92 | `model.audio_tower.layers.1.lconv1d.linear_end.output_min` |  | bfloat16 | 8.125 | -8.125 | 0 | -8.125 | -8.125 | -8.125 |
| 93 | `model.audio_tower.layers.1.lconv1d.linear_start.input_max` |  | bfloat16 | 20.12 | 20.12 | 0 | 20.12 | 20.12 | 20.12 |
| 94 | `model.audio_tower.layers.1.lconv1d.linear_start.input_min` |  | bfloat16 | 20.38 | -20.38 | 0 | -20.38 | -20.38 | -20.38 |
| 95 | `model.audio_tower.layers.1.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.92 | 0.000656 | 0.03101 | -0.1592 | 0.1396 | -0.05127, -0.01709, -0.03418, 0, -0.05127, 0.01709, 0, -0.05127 |
| 96 | `model.audio_tower.layers.1.lconv1d.linear_start.output_max` |  | bfloat16 | 26.62 | 26.62 | 0 | 26.62 | 26.62 | 26.62 |
| 97 | `model.audio_tower.layers.1.lconv1d.linear_start.output_min` |  | bfloat16 | 26.88 | -26.88 | 0 | -26.88 | -26.88 | -26.88 |
| 98 | `model.audio_tower.layers.1.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 266.5 | 7.325 | 3.966 | -7.969 | 26.38 | 8.188, 5.969, 3.891, 11.56, 1.938, 2.359, 10.25, 12.94 |
| 99 | `model.audio_tower.layers.1.norm_out.weight` | 1024 | bfloat16 | 157.8 | 4.289 | 2.432 | -3.469 | 13.19 | 1.82, 6.156, 6.281, 1.023, 1.531, 6.188, 5.031, 4.812 |
| 100 | `model.audio_tower.layers.1.norm_post_attn.weight` | 1024 | bfloat16 | 56.61 | 1.163 | 1.334 | -2 | 10.69 | 4.562, 0.5781, 1.75, 0.4121, 1.617, 0.002075, 0.7969, 1.328 |
| 101 | `model.audio_tower.layers.1.norm_pre_attn.weight` | 1024 | bfloat16 | 297.3 | 7.13 | 5.958 | -10.25 | 35.25 | 14.31, 3.984, 3.375, 5.375, 8.938, 1.555, 7.406, 15 |
| 102 | `model.audio_tower.layers.1.self_attn.k_proj.input_max` |  | bfloat16 | 11.31 | 11.31 | 0 | 11.31 | 11.31 | 11.31 |
| 103 | `model.audio_tower.layers.1.self_attn.k_proj.input_min` |  | bfloat16 | 11.38 | -11.38 | 0 | -11.38 | -11.38 | -11.38 |
| 104 | `model.audio_tower.layers.1.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.55 | -0.00173 | 0.0288 | -0.09668 | 0.04834 | 0.03198, 0, 0, -0.03198, 0, 0.03198, 0, 0 |
| 105 | `model.audio_tower.layers.1.self_attn.k_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 106 | `model.audio_tower.layers.1.self_attn.k_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 107 | `model.audio_tower.layers.1.self_attn.per_dim_scale` | 128 | bfloat16 | 28.61 | -2.505 | 0.3515 | -3.625 | -1.828 | -2.062, -2.797, -2.812, -2.688, -2.688, -2.828, -1.828, -2.422 |
| 108 | `model.audio_tower.layers.1.self_attn.post.input_max` |  | bfloat16 | 17.75 | 17.75 | 0 | 17.75 | 17.75 | 17.75 |
| 109 | `model.audio_tower.layers.1.self_attn.post.input_min` |  | bfloat16 | 17.88 | -17.88 | 0 | -17.88 | -17.88 | -17.88 |
| 110 | `model.audio_tower.layers.1.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.71 | -0.001225 | 0.02899 | -0.1514 | 0.07568 | -0.02783, 0.02783, 0, -0.02783, 0.02783, 0, 0.02783, -0.02783 |
| 111 | `model.audio_tower.layers.1.self_attn.post.output_max` |  | bfloat16 | 57.5 | 57.5 | 0 | 57.5 | 57.5 | 57.5 |
| 112 | `model.audio_tower.layers.1.self_attn.post.output_min` |  | bfloat16 | 58 | -58 | 0 | -58 | -58 | -58 |
| 113 | `model.audio_tower.layers.1.self_attn.q_proj.input_max` |  | bfloat16 | 11.31 | 11.31 | 0 | 11.31 | 11.31 | 11.31 |
| 114 | `model.audio_tower.layers.1.self_attn.q_proj.input_min` |  | bfloat16 | 11.38 | -11.38 | 0 | -11.38 | -11.38 | -11.38 |
| 115 | `model.audio_tower.layers.1.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.88 | -0.002752 | 0.02905 | -0.08545 | 0.04272 | 0, 0.0332, -0.0332, -0.0332, 0, 0, -0.0332, 0 |
| 116 | `model.audio_tower.layers.1.self_attn.q_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 117 | `model.audio_tower.layers.1.self_attn.q_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 118 | `model.audio_tower.layers.1.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.005018 | 0.03084 | -0.1836 | 0.1602 | -0.01929, -0.01239, -0.01031, -0.009827, -0.009949, -0.01428, -0.01721, -0.02124 |
| 119 | `model.audio_tower.layers.1.self_attn.v_proj.input_max` |  | bfloat16 | 11.31 | 11.31 | 0 | 11.31 | 11.31 | 11.31 |
| 120 | `model.audio_tower.layers.1.self_attn.v_proj.input_min` |  | bfloat16 | 11.38 | -11.38 | 0 | -11.38 | -11.38 | -11.38 |
| 121 | `model.audio_tower.layers.1.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.94 | -0.001489 | 0.0292 | -0.1377 | 0.06885 | -0.03271, 0.03271, 0, 0.03271, -0.03271, 0, 0, 0 |
| 122 | `model.audio_tower.layers.1.self_attn.v_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 123 | `model.audio_tower.layers.1.self_attn.v_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 124 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 6.594 | 6.594 | 0 | 6.594 | 6.594 | 6.594 |
| 125 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 6.656 | -6.656 | 0 | -6.656 | -6.656 | -6.656 |
| 126 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.85 | -0.008094 | 0.02857 | -0.07129 | 0.03564 | 0, 0, 0.0271, -0.0542, -0.0271, -0.0542, -0.0271, 0 |
| 127 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 13.56 | 13.56 | 0 | 13.56 | 13.56 | 13.56 |
| 128 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 13.62 | -13.62 | 0 | -13.62 | -13.62 | -13.62 |
| 129 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 8.75 | 8.75 | 0 | 8.75 | 8.75 | 8.75 |
| 130 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 8.812 | -8.812 | 0 | -8.812 | -8.812 | -8.812 |
| 131 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.21 | -0.0009033 | 0.01471 | -0.09521 | 0.04761 | 0, 0, -0.01648, -0.01648, 0, 0, 0.01648, 0 |
| 132 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 49.5 | 49.5 | 0 | 49.5 | 49.5 | 49.5 |
| 133 | `model.audio_tower.layers.10.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 49.75 | -49.75 | 0 | -49.75 | -49.75 | -49.75 |
| 134 | `model.audio_tower.layers.10.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 127.2 | 2.095 | 3.379 | -7.25 | 30.75 | 6.531, 3.969, 0.9492, 4.094, 6.344, -1.078, -2.969, -3.297 |
| 135 | `model.audio_tower.layers.10.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 261 | 5.279 | 6.222 | -34 | 55 | 3.609, 5.625, 2.078, 5.906, 3.688, 2.031, 8.188, -10.38 |
| 136 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 24.5 | 24.5 | 0 | 24.5 | 24.5 | 24.5 |
| 137 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 24.62 | -24.62 | 0 | -24.62 | -24.62 | -24.62 |
| 138 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.89 | -0.008264 | 0.02855 | -0.07031 | 0.03516 | 0.02966, -0.02966, -0.02966, 0.02966, 0, 0.02966, 0.02966, -0.02966 |
| 139 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 68 | 68 | 0 | 68 | 68 | 68 |
| 140 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 68.5 | -68.5 | 0 | -68.5 | -68.5 | -68.5 |
| 141 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 23.12 | 23.12 | 0 | 23.12 | 23.12 | 23.12 |
| 142 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 23.38 | -23.38 | 0 | -23.38 | -23.38 | -23.38 |
| 143 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.22 | -0.00052 | 0.01474 | -0.05811 | 0.02905 | 0.01697, -0.01697, 0, 0.01697, 0.01697, -0.03394, -0.03394, -0.01697 |
| 144 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 211 | 211 | 0 | 211 | 211 | 211 |
| 145 | `model.audio_tower.layers.10.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 212 | -212 | 0 | -212 | -212 | -212 |
| 146 | `model.audio_tower.layers.10.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 246.8 | 4.62 | 6.18 | -18 | 76 | 3.234, 0.332, -1.633, 6.406, 10.62, 1.609, -6.906, -4.062 |
| 147 | `model.audio_tower.layers.10.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 589.4 | 16.87 | 7.385 | -29.25 | 36.75 | 13.19, 19.62, 9.188, 20.75, 14.31, 6.562, 22.88, 20.25 |
| 148 | `model.audio_tower.layers.10.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 138.9 | 3.618 | 2.398 | -9 | 11.12 | 8, 4.844, 5.344, 4.281, 6.656, 4.219, 3.328, 7.531 |
| 149 | `model.audio_tower.layers.10.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | 0.006362 | 0.4472 | -4.906 | 2.125 | 0.01575, -0.05859, -0.04419, 0.3438, -0.208, 0.4141, 0.3008, 0.5586 |
| 150 | `model.audio_tower.layers.10.lconv1d.linear_end.input_max` |  | bfloat16 | 15.5 | 15.5 | 0 | 15.5 | 15.5 | 15.5 |
| 151 | `model.audio_tower.layers.10.lconv1d.linear_end.input_min` |  | bfloat16 | 15.62 | -15.62 | 0 | -15.62 | -15.62 | -15.62 |
| 152 | `model.audio_tower.layers.10.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 30 | 0.0007399 | 0.02929 | -0.1504 | 0.0752 | 0, -0.03467, 0.03467, 0.03467, 0, 0.03467, 0.03467, 0.03467 |
| 153 | `model.audio_tower.layers.10.lconv1d.linear_end.output_max` |  | bfloat16 | 10.94 | 10.94 | 0 | 10.94 | 10.94 | 10.94 |
| 154 | `model.audio_tower.layers.10.lconv1d.linear_end.output_min` |  | bfloat16 | 11 | -11 | 0 | -11 | -11 | -11 |
| 155 | `model.audio_tower.layers.10.lconv1d.linear_start.input_max` |  | bfloat16 | 10.25 | 10.25 | 0 | 10.25 | 10.25 | 10.25 |
| 156 | `model.audio_tower.layers.10.lconv1d.linear_start.input_min` |  | bfloat16 | 10.31 | -10.31 | 0 | -10.31 | -10.31 | -10.31 |
| 157 | `model.audio_tower.layers.10.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.93 | -0.001676 | 0.03098 | -0.2109 | 0.1846 | 0.02002, -0.01001, 0.03003, 0, -0.02002, 0.04004, -0.01001, 0.04004 |
| 158 | `model.audio_tower.layers.10.lconv1d.linear_start.output_max` |  | bfloat16 | 27.38 | 27.38 | 0 | 27.38 | 27.38 | 27.38 |
| 159 | `model.audio_tower.layers.10.lconv1d.linear_start.output_min` |  | bfloat16 | 27.62 | -27.62 | 0 | -27.62 | -27.62 | -27.62 |
| 160 | `model.audio_tower.layers.10.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 160.2 | 3.552 | 3.531 | -6.312 | 48.75 | 1.422, 3.219, 3.312, 2.844, 1.773, 3.391, 4.469, 5.688 |
| 161 | `model.audio_tower.layers.10.norm_out.weight` | 1024 | bfloat16 | 263.9 | 6.205 | 5.433 | -18.88 | 27.75 | 3.734, 5.062, 14, 4.125, 5.438, 14.25, 2.688, 14.31 |
| 162 | `model.audio_tower.layers.10.norm_post_attn.weight` | 1024 | bfloat16 | 59.97 | 0.6301 | 1.766 | -10.81 | 14.19 | -1.156, -0.9492, -0.009216, -1.039, -1.211, 0.03491, 1.453, 0.7695 |
| 163 | `model.audio_tower.layers.10.norm_pre_attn.weight` | 1024 | bfloat16 | 182 | 3.712 | 4.311 | -23.38 | 28.75 | 1.133, 3.141, 1.789, 2.672, 1.766, -3.156, 6.25, 5.969 |
| 164 | `model.audio_tower.layers.10.self_attn.k_proj.input_max` |  | bfloat16 | 10.31 | 10.31 | 0 | 10.31 | 10.31 | 10.31 |
| 165 | `model.audio_tower.layers.10.self_attn.k_proj.input_min` |  | bfloat16 | 10.44 | -10.44 | 0 | -10.44 | -10.44 | -10.44 |
| 166 | `model.audio_tower.layers.10.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.73 | -0.001734 | 0.02898 | -0.08594 | 0.05005 | 0.03906, 0.03906, 0, 0, 0.03906, 0.03906, 0, 0.03906 |
| 167 | `model.audio_tower.layers.10.self_attn.k_proj.output_max` |  | bfloat16 | 19.12 | 19.12 | 0 | 19.12 | 19.12 | 19.12 |
| 168 | `model.audio_tower.layers.10.self_attn.k_proj.output_min` |  | bfloat16 | 19.38 | -19.38 | 0 | -19.38 | -19.38 | -19.38 |
| 169 | `model.audio_tower.layers.10.self_attn.per_dim_scale` | 128 | bfloat16 | 25.2 | -2.201 | 0.3454 | -3.234 | -1.594 | -2.141, -2.094, -1.594, -1.734, -2.359, -2.5, -2.453, -1.758 |
| 170 | `model.audio_tower.layers.10.self_attn.post.input_max` |  | bfloat16 | 18.75 | 18.75 | 0 | 18.75 | 18.75 | 18.75 |
| 171 | `model.audio_tower.layers.10.self_attn.post.input_min` |  | bfloat16 | 19 | -19 | 0 | -19 | -19 | -19 |
| 172 | `model.audio_tower.layers.10.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.96 | -0.001733 | 0.02921 | -0.1338 | 0.06689 | -0.03345, 0, 0, 0.03345, 0.03345, 0, -0.03345, 0 |
| 173 | `model.audio_tower.layers.10.self_attn.post.output_max` |  | bfloat16 | 99 | 99 | 0 | 99 | 99 | 99 |
| 174 | `model.audio_tower.layers.10.self_attn.post.output_min` |  | bfloat16 | 100 | -100 | 0 | -100 | -100 | -100 |
| 175 | `model.audio_tower.layers.10.self_attn.q_proj.input_max` |  | bfloat16 | 10.31 | 10.31 | 0 | 10.31 | 10.31 | 10.31 |
| 176 | `model.audio_tower.layers.10.self_attn.q_proj.input_min` |  | bfloat16 | 10.44 | -10.44 | 0 | -10.44 | -10.44 | -10.44 |
| 177 | `model.audio_tower.layers.10.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.57 | -0.001973 | 0.02881 | -0.1484 | 0.07422 | -0.03857, -0.03857, -0.03857, -0.03857, 0, 0, 0, 0.03857 |
| 178 | `model.audio_tower.layers.10.self_attn.q_proj.output_max` |  | bfloat16 | 19.12 | 19.12 | 0 | 19.12 | 19.12 | 19.12 |
| 179 | `model.audio_tower.layers.10.self_attn.q_proj.output_min` |  | bfloat16 | 19.38 | -19.38 | 0 | -19.38 | -19.38 | -19.38 |
| 180 | `model.audio_tower.layers.10.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.004738 | 0.03089 | -0.2471 | 0.2363 | 0.02527, 0.02405, 0.02283, 0.02112, 0.01941, 0.01758, 0.01599, 0.0141 |
| 181 | `model.audio_tower.layers.10.self_attn.v_proj.input_max` |  | bfloat16 | 10.31 | 10.31 | 0 | 10.31 | 10.31 | 10.31 |
| 182 | `model.audio_tower.layers.10.self_attn.v_proj.input_min` |  | bfloat16 | 10.44 | -10.44 | 0 | -10.44 | -10.44 | -10.44 |
| 183 | `model.audio_tower.layers.10.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 30.03 | -0.001367 | 0.0293 | -0.07617 | 0.03809 | 0, -0.03467, 0.03467, 0.03467, 0.03467, 0, 0, 0 |
| 184 | `model.audio_tower.layers.10.self_attn.v_proj.output_max` |  | bfloat16 | 19.12 | 19.12 | 0 | 19.12 | 19.12 | 19.12 |
| 185 | `model.audio_tower.layers.10.self_attn.v_proj.output_min` |  | bfloat16 | 19.38 | -19.38 | 0 | -19.38 | -19.38 | -19.38 |
| 186 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 9.188 | 9.188 | 0 | 9.188 | 9.188 | 9.188 |
| 187 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 9.25 | -9.25 | 0 | -9.25 | -9.25 | -9.25 |
| 188 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.99 | -0.008193 | 0.02862 | -0.09521 | 0.04761 | -0.02893, -0.05786, -0.02893, 0, 0.02893, 0, -0.02893, 0 |
| 189 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 21.62 | 21.62 | 0 | 21.62 | 21.62 | 21.62 |
| 190 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 21.88 | -21.88 | 0 | -21.88 | -21.88 | -21.88 |
| 191 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 9.812 | 9.812 | 0 | 9.812 | 9.812 | 9.812 |
| 192 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 9.875 | -9.875 | 0 | -9.875 | -9.875 | -9.875 |
| 193 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.22 | -0.0005692 | 0.01474 | -0.06689 | 0.03345 | -0.0166, -0.0332, 0.0166, 0, 0, -0.0332, 0.0166, -0.0166 |
| 194 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 49.5 | 49.5 | 0 | 49.5 | 49.5 | 49.5 |
| 195 | `model.audio_tower.layers.11.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 50 | -50 | 0 | -50 | -50 | -50 |
| 196 | `model.audio_tower.layers.11.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 177.2 | 4.069 | 3.76 | -17.88 | 44.25 | 6.438, 4.312, 2.062, 4.531, 9.562, 1.867, 3.25, -6.625 |
| 197 | `model.audio_tower.layers.11.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 316 | 7.517 | 6.406 | -28.38 | 50 | 9.75, 9.125, 1.469, 11.62, 5.531, 1.125, 18.62, 3.719 |
| 198 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 12.75 | 12.75 | 0 | 12.75 | 12.75 | 12.75 |
| 199 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 12.88 | -12.88 | 0 | -12.88 | -12.88 | -12.88 |
| 200 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.95 | -0.007956 | 0.02867 | -0.0791 | 0.03955 | -0.0304, -0.06079, -0.0304, -0.0304, 0, -0.0304, 0, 0.0304 |
| 201 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 40 | 40 | 0 | 40 | 40 | 40 |
| 202 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 40.5 | -40.5 | 0 | -40.5 | -40.5 | -40.5 |
| 203 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 11.19 | 11.19 | 0 | 11.19 | 11.19 | 11.19 |
| 204 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 11.31 | -11.31 | 0 | -11.31 | -11.31 | -11.31 |
| 205 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.81 | -0.0004569 | 0.01454 | -0.05933 | 0.02966 | 0, -0.01562, 0.01562, 0.01562, 0.01562, 0, -0.01562, 0.01562 |
| 206 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 56 | 56 | 0 | 56 | 56 | 56 |
| 207 | `model.audio_tower.layers.11.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 56.5 | -56.5 | 0 | -56.5 | -56.5 | -56.5 |
| 208 | `model.audio_tower.layers.11.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 443.5 | 7.611 | 11.59 | -40.25 | 87.5 | -17.75, -1.227, -3.172, -15, 10.31, 2.141, 7.5, 10.44 |
| 209 | `model.audio_tower.layers.11.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 193.4 | 4.746 | 3.743 | -18.5 | 14 | -6.5, 6.938, 1.648, 8.812, 4.219, 1.172, 5.562, 2.938 |
| 210 | `model.audio_tower.layers.11.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 230.2 | 5.255 | 4.917 | -13.44 | 106 | 9.125, 9.375, 6.219, 5.625, -3.062, 8.062, 7.812, 3.906 |
| 211 | `model.audio_tower.layers.11.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.001839 | 0.4473 | -4.344 | 2.484 | -0.08643, -0.05444, -0.04956, 0.001587, -0.5859, -0.01843, -0.06982, -0.1309 |
| 212 | `model.audio_tower.layers.11.lconv1d.linear_end.input_max` |  | bfloat16 | 30.12 | 30.12 | 0 | 30.12 | 30.12 | 30.12 |
| 213 | `model.audio_tower.layers.11.lconv1d.linear_end.input_min` |  | bfloat16 | 30.38 | -30.38 | 0 | -30.38 | -30.38 | -30.38 |
| 214 | `model.audio_tower.layers.11.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 30.14 | -0.0001203 | 0.02944 | -0.1504 | 0.0752 | -0.03271, -0.03271, 0, 0.03271, 0, 0, 0, 0.03271 |
| 215 | `model.audio_tower.layers.11.lconv1d.linear_end.output_max` |  | bfloat16 | 16.25 | 16.25 | 0 | 16.25 | 16.25 | 16.25 |
| 216 | `model.audio_tower.layers.11.lconv1d.linear_end.output_min` |  | bfloat16 | 16.38 | -16.38 | 0 | -16.38 | -16.38 | -16.38 |
| 217 | `model.audio_tower.layers.11.lconv1d.linear_start.input_max` |  | bfloat16 | 11 | 11 | 0 | 11 | 11 | 11 |
| 218 | `model.audio_tower.layers.11.lconv1d.linear_start.input_min` |  | bfloat16 | 11.12 | -11.12 | 0 | -11.12 | -11.12 | -11.12 |
| 219 | `model.audio_tower.layers.11.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.93 | -0.001245 | 0.031 | -0.2832 | 0.25 | 0, -0.03223, -0.02148, -0.03223, 0.01074, -0.01074, -0.04297, 0.03223 |
| 220 | `model.audio_tower.layers.11.lconv1d.linear_start.output_max` |  | bfloat16 | 27.12 | 27.12 | 0 | 27.12 | 27.12 | 27.12 |
| 221 | `model.audio_tower.layers.11.lconv1d.linear_start.output_min` |  | bfloat16 | 27.38 | -27.38 | 0 | -27.38 | -27.38 | -27.38 |
| 222 | `model.audio_tower.layers.11.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 223.2 | 3.463 | 6.058 | -25.88 | 101 | 2.766, 4.531, 1.883, 4.438, -2.062, 1.703, 3.359, 2.484 |
| 223 | `model.audio_tower.layers.11.norm_out.weight` | 1024 | bfloat16 | 211.9 | 3.595 | 5.564 | -18.12 | 24.62 | 5.812, 0.1035, 8.812, -4.25, 1.922, 8.812, -2.391, 6.688 |
| 224 | `model.audio_tower.layers.11.norm_post_attn.weight` | 1024 | bfloat16 | 85.95 | 0.9287 | 2.522 | -7.812 | 22.5 | -1.414, 2.812, 1.234, 1.047, 1.969, 1.156, -6.344, -0.01208 |
| 225 | `model.audio_tower.layers.11.norm_pre_attn.weight` | 1024 | bfloat16 | 174.7 | 3.864 | 3.858 | -20.75 | 21.38 | -3.109, 5.875, 1.047, -4.531, 2.391, 1.844, 11, 2.078 |
| 226 | `model.audio_tower.layers.11.self_attn.k_proj.input_max` |  | bfloat16 | 9.25 | 9.25 | 0 | 9.25 | 9.25 | 9.25 |
| 227 | `model.audio_tower.layers.11.self_attn.k_proj.input_min` |  | bfloat16 | 9.375 | -9.375 | 0 | -9.375 | -9.375 | -9.375 |
| 228 | `model.audio_tower.layers.11.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.85 | -0.00222 | 0.02906 | -0.08252 | 0.04126 | -0.02942, 0.02942, 0, 0, -0.02942, 0, 0, -0.05884 |
| 229 | `model.audio_tower.layers.11.self_attn.k_proj.output_max` |  | bfloat16 | 17 | 17 | 0 | 17 | 17 | 17 |
| 230 | `model.audio_tower.layers.11.self_attn.k_proj.output_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 231 | `model.audio_tower.layers.11.self_attn.per_dim_scale` | 128 | bfloat16 | 23.05 | -2.028 | 0.1969 | -2.828 | -1.727 | -2.031, -2.25, -2.047, -2.203, -1.914, -1.836, -2, -2.109 |
| 232 | `model.audio_tower.layers.11.self_attn.post.input_max` |  | bfloat16 | 15.31 | 15.31 | 0 | 15.31 | 15.31 | 15.31 |
| 233 | `model.audio_tower.layers.11.self_attn.post.input_min` |  | bfloat16 | 15.44 | -15.44 | 0 | -15.44 | -15.44 | -15.44 |
| 234 | `model.audio_tower.layers.11.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 30.03 | -0.001544 | 0.02928 | -0.1387 | 0.06934 | 0.03198, -0.03198, 0, -0.03198, 0, 0, -0.03198, 0 |
| 235 | `model.audio_tower.layers.11.self_attn.post.output_max` |  | bfloat16 | 71.5 | 71.5 | 0 | 71.5 | 71.5 | 71.5 |
| 236 | `model.audio_tower.layers.11.self_attn.post.output_min` |  | bfloat16 | 72 | -72 | 0 | -72 | -72 | -72 |
| 237 | `model.audio_tower.layers.11.self_attn.q_proj.input_max` |  | bfloat16 | 9.25 | 9.25 | 0 | 9.25 | 9.25 | 9.25 |
| 238 | `model.audio_tower.layers.11.self_attn.q_proj.input_min` |  | bfloat16 | 9.375 | -9.375 | 0 | -9.375 | -9.375 | -9.375 |
| 239 | `model.audio_tower.layers.11.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.77 | -0.002199 | 0.02899 | -0.1357 | 0.06787 | 0, -0.0293, 0, 0, 0, -0.05859, 0, 0.0293 |
| 240 | `model.audio_tower.layers.11.self_attn.q_proj.output_max` |  | bfloat16 | 17 | 17 | 0 | 17 | 17 | 17 |
| 241 | `model.audio_tower.layers.11.self_attn.q_proj.output_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 242 | `model.audio_tower.layers.11.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.00217 | 0.03117 | -0.2334 | 0.1396 | 0.09033, 0.0835, 0.08105, 0.07764, 0.0752, 0.0752, 0.07568, 0.07471 |
| 243 | `model.audio_tower.layers.11.self_attn.v_proj.input_max` |  | bfloat16 | 9.25 | 9.25 | 0 | 9.25 | 9.25 | 9.25 |
| 244 | `model.audio_tower.layers.11.self_attn.v_proj.input_min` |  | bfloat16 | 9.375 | -9.375 | 0 | -9.375 | -9.375 | -9.375 |
| 245 | `model.audio_tower.layers.11.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 30.07 | -0.002046 | 0.0293 | -0.07861 | 0.03931 | 0, -0.06738, -0.03369, -0.03369, 0, -0.03369, 0, 0 |
| 246 | `model.audio_tower.layers.11.self_attn.v_proj.output_max` |  | bfloat16 | 17 | 17 | 0 | 17 | 17 | 17 |
| 247 | `model.audio_tower.layers.11.self_attn.v_proj.output_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 248 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.88 | 12.88 | 0 | 12.88 | 12.88 | 12.88 |
| 249 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 13 | -13 | 0 | -13 | -13 | -13 |
| 250 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.61 | -0.007605 | 0.0286 | -0.09082 | 0.04541 | 0, 0.03247, -0.03247, 0, -0.03247, 0.03247, -0.06494, -0.06494 |
| 251 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 28.12 | 28.12 | 0 | 28.12 | 28.12 | 28.12 |
| 252 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 28.38 | -28.38 | 0 | -28.38 | -28.38 | -28.38 |
| 253 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 9.188 | 9.188 | 0 | 9.188 | 9.188 | 9.188 |
| 254 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 9.25 | -9.25 | 0 | -9.25 | -9.25 | -9.25 |
| 255 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.77 | 4.562e-05 | 0.01453 | -0.09619 | 0.0481 | 0.01672, -0.01672, 0, 0, -0.01672, 0, 0, 0 |
| 256 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 35.75 | 35.75 | 0 | 35.75 | 35.75 | 35.75 |
| 257 | `model.audio_tower.layers.2.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 36 | -36 | 0 | -36 | -36 | -36 |
| 258 | `model.audio_tower.layers.2.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 66.31 | 1.645 | 1.261 | -5.188 | 7.844 | 0.7305, 2.125, 1.367, 1.492, 2.141, 0.9531, 1.758, 0.7812 |
| 259 | `model.audio_tower.layers.2.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 599.8 | 15.07 | 11.16 | -56.25 | 54.25 | 15.5, 8.625, 6.031, 45.25, 28.38, 3.469, 13.56, 15.62 |
| 260 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 261 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 11.88 | -11.88 | 0 | -11.88 | -11.88 | -11.88 |
| 262 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.59 | -0.007793 | 0.02853 | -0.09473 | 0.04736 | 0, -0.05908, 0, -0.02954, 0, -0.02954, -0.02954, 0 |
| 263 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 27.38 | 27.38 | 0 | 27.38 | 27.38 | 27.38 |
| 264 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 27.5 | -27.5 | 0 | -27.5 | -27.5 | -27.5 |
| 265 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 9.188 | 9.188 | 0 | 9.188 | 9.188 | 9.188 |
| 266 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 9.25 | -9.25 | 0 | -9.25 | -9.25 | -9.25 |
| 267 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.7 | -0.0002553 | 0.0145 | -0.08203 | 0.04102 | -0.01587, -0.01587, 0.01587, 0, 0, 0.01587, 0.01587, 0.01587 |
| 268 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 38.75 | 38.75 | 0 | 38.75 | 38.75 | 38.75 |
| 269 | `model.audio_tower.layers.2.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 39 | -39 | 0 | -39 | -39 | -39 |
| 270 | `model.audio_tower.layers.2.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 133.7 | 3.538 | 2.225 | -5.688 | 16.38 | -1.188, 4.375, 2.312, 2.578, -0.07373, 2.438, 3.453, 2.406 |
| 271 | `model.audio_tower.layers.2.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 228.7 | 6.513 | 2.946 | -11.06 | 12.44 | 2.484, 6.125, 4.875, 9.875, 8.812, 4.094, 8.062, 9.938 |
| 272 | `model.audio_tower.layers.2.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 147.5 | 4.033 | 2.23 | -6 | 44.5 | 3.812, 1.391, 2.891, 5.781, 2.141, 5, 4.562, 5.375 |
| 273 | `model.audio_tower.layers.2.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.003985 | 0.4472 | -4 | 2.625 | -0.2373, -0.2578, -0.5742, -0.7539, -0.1611, 0.08447, 0.02161, 0.5078 |
| 274 | `model.audio_tower.layers.2.lconv1d.linear_end.input_max` |  | bfloat16 | 6.219 | 6.219 | 0 | 6.219 | 6.219 | 6.219 |
| 275 | `model.audio_tower.layers.2.lconv1d.linear_end.input_min` |  | bfloat16 | 6.25 | -6.25 | 0 | -6.25 | -6.25 | -6.25 |
| 276 | `model.audio_tower.layers.2.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.95 | 0.0007329 | 0.02924 | -0.1147 | 0.05737 | 0, 0, 0.03418, 0, -0.03418, 0, 0, 0 |
| 277 | `model.audio_tower.layers.2.lconv1d.linear_end.output_max` |  | bfloat16 | 6.312 | 6.312 | 0 | 6.312 | 6.312 | 6.312 |
| 278 | `model.audio_tower.layers.2.lconv1d.linear_end.output_min` |  | bfloat16 | 6.375 | -6.375 | 0 | -6.375 | -6.375 | -6.375 |
| 279 | `model.audio_tower.layers.2.lconv1d.linear_start.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 280 | `model.audio_tower.layers.2.lconv1d.linear_start.input_min` |  | bfloat16 | 11.88 | -11.88 | 0 | -11.88 | -11.88 | -11.88 |
| 281 | `model.audio_tower.layers.2.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.95 | 0.0003592 | 0.03104 | -0.1436 | 0.126 | -0.0238, -0.0238, 0, -0.0119, 0.07129, 0.0119, -0.07129, 0.0238 |
| 282 | `model.audio_tower.layers.2.lconv1d.linear_start.output_max` |  | bfloat16 | 26.25 | 26.25 | 0 | 26.25 | 26.25 | 26.25 |
| 283 | `model.audio_tower.layers.2.lconv1d.linear_start.output_min` |  | bfloat16 | 26.5 | -26.5 | 0 | -26.5 | -26.5 | -26.5 |
| 284 | `model.audio_tower.layers.2.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 240.6 | 6.927 | 2.924 | -10.06 | 19.25 | 4.469, 5.344, 4, 9.5, 7.781, 7.875, 7.688, 10.06 |
| 285 | `model.audio_tower.layers.2.norm_out.weight` | 1024 | bfloat16 | 130.7 | 3.656 | 1.819 | -0.9258 | 13.69 | 6, 2.891, 4.812, 2.438, 1.406, 2.828, 2.891, 5.156 |
| 286 | `model.audio_tower.layers.2.norm_post_attn.weight` | 1024 | bfloat16 | 32.18 | 0.813 | 0.5925 | -1.062 | 8.375 | 3, 0.4082, 0.8516, 0.6836, 0.6836, -1.062, 0.3203, 1 |
| 287 | `model.audio_tower.layers.2.norm_pre_attn.weight` | 1024 | bfloat16 | 332.2 | 8.445 | 6.04 | -8.375 | 32.25 | 18.88, 3.469, 3.391, 19.38, 12.25, 3.922, 5.062, 14 |
| 288 | `model.audio_tower.layers.2.self_attn.k_proj.input_max` |  | bfloat16 | 12.38 | 12.38 | 0 | 12.38 | 12.38 | 12.38 |
| 289 | `model.audio_tower.layers.2.self_attn.k_proj.input_min` |  | bfloat16 | 12.5 | -12.5 | 0 | -12.5 | -12.5 | -12.5 |
| 290 | `model.audio_tower.layers.2.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.36 | -0.001569 | 0.02863 | -0.1064 | 0.05322 | 0.02576, -0.02576, 0, 0, -0.05151, 0, -0.05151, 0 |
| 291 | `model.audio_tower.layers.2.self_attn.k_proj.output_max` |  | bfloat16 | 20.75 | 20.75 | 0 | 20.75 | 20.75 | 20.75 |
| 292 | `model.audio_tower.layers.2.self_attn.k_proj.output_min` |  | bfloat16 | 20.88 | -20.88 | 0 | -20.88 | -20.88 | -20.88 |
| 293 | `model.audio_tower.layers.2.self_attn.per_dim_scale` | 128 | bfloat16 | 23.15 | -2.034 | 0.2198 | -2.875 | -1.516 | -2.031, -1.859, -2.141, -2.188, -1.953, -2.469, -1.906, -2.141 |
| 294 | `model.audio_tower.layers.2.self_attn.post.input_max` |  | bfloat16 | 18 | 18 | 0 | 18 | 18 | 18 |
| 295 | `model.audio_tower.layers.2.self_attn.post.input_min` |  | bfloat16 | 18.12 | -18.12 | 0 | -18.12 | -18.12 | -18.12 |
| 296 | `model.audio_tower.layers.2.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.87 | -0.001183 | 0.02915 | -0.1768 | 0.08838 | -0.02844, -0.02844, 0.02844, 0.02844, -0.02844, -0.02844, -0.02844, 0.02844 |
| 297 | `model.audio_tower.layers.2.self_attn.post.output_max` |  | bfloat16 | 61.25 | 61.25 | 0 | 61.25 | 61.25 | 61.25 |
| 298 | `model.audio_tower.layers.2.self_attn.post.output_min` |  | bfloat16 | 61.75 | -61.75 | 0 | -61.75 | -61.75 | -61.75 |
| 299 | `model.audio_tower.layers.2.self_attn.q_proj.input_max` |  | bfloat16 | 12.38 | 12.38 | 0 | 12.38 | 12.38 | 12.38 |
| 300 | `model.audio_tower.layers.2.self_attn.q_proj.input_min` |  | bfloat16 | 12.5 | -12.5 | 0 | -12.5 | -12.5 | -12.5 |
| 301 | `model.audio_tower.layers.2.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.75 | -0.002531 | 0.02894 | -0.08301 | 0.0415 | -0.0271, 0, -0.0542, 0, 0, -0.0271, -0.0271, 0.0271 |
| 302 | `model.audio_tower.layers.2.self_attn.q_proj.output_max` |  | bfloat16 | 20.75 | 20.75 | 0 | 20.75 | 20.75 | 20.75 |
| 303 | `model.audio_tower.layers.2.self_attn.q_proj.output_min` |  | bfloat16 | 20.88 | -20.88 | 0 | -20.88 | -20.88 | -20.88 |
| 304 | `model.audio_tower.layers.2.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.004458 | 0.03093 | -0.1787 | 0.1494 | -0.03223, -0.02832, -0.02686, -0.02515, -0.02356, -0.02332, -0.02283, -0.02161 |
| 305 | `model.audio_tower.layers.2.self_attn.v_proj.input_max` |  | bfloat16 | 12.38 | 12.38 | 0 | 12.38 | 12.38 | 12.38 |
| 306 | `model.audio_tower.layers.2.self_attn.v_proj.input_min` |  | bfloat16 | 12.5 | -12.5 | 0 | -12.5 | -12.5 | -12.5 |
| 307 | `model.audio_tower.layers.2.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.97 | -0.001541 | 0.02923 | -0.2197 | 0.1099 | 0.03113, 0.03113, 0, 0, 0.03113, 0, 0, -0.03113 |
| 308 | `model.audio_tower.layers.2.self_attn.v_proj.output_max` |  | bfloat16 | 20.75 | 20.75 | 0 | 20.75 | 20.75 | 20.75 |
| 309 | `model.audio_tower.layers.2.self_attn.v_proj.output_min` |  | bfloat16 | 20.88 | -20.88 | 0 | -20.88 | -20.88 | -20.88 |
| 310 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.25 | 12.25 | 0 | 12.25 | 12.25 | 12.25 |
| 311 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.31 | -12.31 | 0 | -12.31 | -12.31 | -12.31 |
| 312 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.63 | -0.007684 | 0.02858 | -0.1123 | 0.05615 | 0.02673, 0.02673, 0, 0.02673, 0, -0.05347, -0.02673, -0.02673 |
| 313 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 26.75 | 26.75 | 0 | 26.75 | 26.75 | 26.75 |
| 314 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 27 | -27 | 0 | -27 | -27 | -27 |
| 315 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 9.125 | 9.125 | 0 | 9.125 | 9.125 | 9.125 |
| 316 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 9.188 | -9.188 | 0 | -9.188 | -9.188 | -9.188 |
| 317 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.75 | 7.91e-05 | 0.01452 | -0.09814 | 0.04907 | 0, 0.04346, 0.04346, 0.04346, 0.04346, 0, 0, 0 |
| 318 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 40.75 | 40.75 | 0 | 40.75 | 40.75 | 40.75 |
| 319 | `model.audio_tower.layers.3.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 41 | -41 | 0 | -41 | -41 | -41 |
| 320 | `model.audio_tower.layers.3.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 61.37 | 1.497 | 1.199 | -3.906 | 9.062 | 0.05542, 1.445, 1.016, 0.9023, 1.484, 0.498, 1.195, 1.172 |
| 321 | `model.audio_tower.layers.3.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 558.9 | 14.22 | 10.15 | -41.5 | 57 | 3.594, 11.62, 7.5, 24, 35.25, 9.438, 17, 11.62 |
| 322 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 13.69 | 13.69 | 0 | 13.69 | 13.69 | 13.69 |
| 323 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 13.81 | -13.81 | 0 | -13.81 | -13.81 | -13.81 |
| 324 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.58 | -0.007749 | 0.02854 | -0.1206 | 0.06104 | -0.0332, 0, -0.0332, 0.0332, -0.0332, -0.0332, 0, 0 |
| 325 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 29.25 | 29.25 | 0 | 29.25 | 29.25 | 29.25 |
| 326 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 29.5 | -29.5 | 0 | -29.5 | -29.5 | -29.5 |
| 327 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 13.25 | 13.25 | 0 | 13.25 | 13.25 | 13.25 |
| 328 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 13.38 | -13.38 | 0 | -13.38 | -13.38 | -13.38 |
| 329 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.57 | 0.0002076 | 0.01443 | -0.09668 | 0.04834 | 0, 0.03418, 0.03418, 0, 0.03418, 0, -0.03418, 0 |
| 330 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 92.5 | 92.5 | 0 | 92.5 | 92.5 | 92.5 |
| 331 | `model.audio_tower.layers.3.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 93 | -93 | 0 | -93 | -93 | -93 |
| 332 | `model.audio_tower.layers.3.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 140.5 | 3.156 | 3.054 | -13.44 | 56.5 | 0.01367, 3.688, 1.203, 2.188, 6.688, 1.016, 4.031, 6.25 |
| 333 | `model.audio_tower.layers.3.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 222.6 | 6.575 | 2.268 | -6.438 | 12 | 4.812, 7.438, 5.062, 8.125, 7.406, 3.281, 10.12, 8.438 |
| 334 | `model.audio_tower.layers.3.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 122.3 | 3.604 | 1.269 | -4.5 | 9.312 | 3.188, 4.969, 2.531, 4.031, 2.531, 3.734, 5, 3.516 |
| 335 | `model.audio_tower.layers.3.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.01093 | 0.4471 | -2.969 | 2.344 | 0.0415, -0.005829, 0.5781, -0.6797, 0.07422, 0.001846, -0.07666, -0.6484 |
| 336 | `model.audio_tower.layers.3.lconv1d.linear_end.input_max` |  | bfloat16 | 7.5 | 7.5 | 0 | 7.5 | 7.5 | 7.5 |
| 337 | `model.audio_tower.layers.3.lconv1d.linear_end.input_min` |  | bfloat16 | 7.562 | -7.562 | 0 | -7.562 | -7.562 | -7.562 |
| 338 | `model.audio_tower.layers.3.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.93 | 0.0006381 | 0.02922 | -0.127 | 0.06348 | 0, 0, -0.03052, -0.03052, 0, 0, 0, -0.03052 |
| 339 | `model.audio_tower.layers.3.lconv1d.linear_end.output_max` |  | bfloat16 | 7.875 | 7.875 | 0 | 7.875 | 7.875 | 7.875 |
| 340 | `model.audio_tower.layers.3.lconv1d.linear_end.output_min` |  | bfloat16 | 7.938 | -7.938 | 0 | -7.938 | -7.938 | -7.938 |
| 341 | `model.audio_tower.layers.3.lconv1d.linear_start.input_max` |  | bfloat16 | 11.19 | 11.19 | 0 | 11.19 | 11.19 | 11.19 |
| 342 | `model.audio_tower.layers.3.lconv1d.linear_start.input_min` |  | bfloat16 | 11.25 | -11.25 | 0 | -11.25 | -11.25 | -11.25 |
| 343 | `model.audio_tower.layers.3.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.96 | 0.000627 | 0.03104 | -0.1387 | 0.1216 | -0.01038, 0.01038, -0.03113, 0.01038, -0.03113, 0.0415, -0.02075, 0.01038 |
| 344 | `model.audio_tower.layers.3.lconv1d.linear_start.output_max` |  | bfloat16 | 24.75 | 24.75 | 0 | 24.75 | 24.75 | 24.75 |
| 345 | `model.audio_tower.layers.3.lconv1d.linear_start.output_min` |  | bfloat16 | 25 | -25 | 0 | -25 | -25 | -25 |
| 346 | `model.audio_tower.layers.3.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 183.5 | 5.287 | 2.22 | -4.594 | 14.31 | 3.641, 6.781, 4.188, 7, 4.094, 8.562, 7.75, 6.156 |
| 347 | `model.audio_tower.layers.3.norm_out.weight` | 1024 | bfloat16 | 152.7 | 4.296 | 2.075 | -4.219 | 22.38 | 2.141, 2.125, 5.188, 3.969, 2.938, 5.969, 3.359, 1.18 |
| 348 | `model.audio_tower.layers.3.norm_post_attn.weight` | 1024 | bfloat16 | 41.97 | 0.9721 | 0.8809 | -1.008 | 8.312 | 2.984, 0.4336, 1.133, 1.359, 1.891, -0.5469, 0.5117, 0.04883 |
| 349 | `model.audio_tower.layers.3.norm_pre_attn.weight` | 1024 | bfloat16 | 260.8 | 6.822 | 4.461 | -4.469 | 38.5 | 6.5, 4.656, 3.031, 14.19, 12.56, 5.938, 5.75, 7.281 |
| 350 | `model.audio_tower.layers.3.self_attn.k_proj.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 351 | `model.audio_tower.layers.3.self_attn.k_proj.input_min` |  | bfloat16 | 11.81 | -11.81 | 0 | -11.81 | -11.81 | -11.81 |
| 352 | `model.audio_tower.layers.3.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.63 | -0.001993 | 0.02886 | -0.09277 | 0.04639 | -0.03491, 0, 0, 0, -0.03491, 0, 0.03491, -0.06982 |
| 353 | `model.audio_tower.layers.3.self_attn.k_proj.output_max` |  | bfloat16 | 18.62 | 18.62 | 0 | 18.62 | 18.62 | 18.62 |
| 354 | `model.audio_tower.layers.3.self_attn.k_proj.output_min` |  | bfloat16 | 18.75 | -18.75 | 0 | -18.75 | -18.75 | -18.75 |
| 355 | `model.audio_tower.layers.3.self_attn.per_dim_scale` | 128 | bfloat16 | 21.62 | -1.889 | 0.2903 | -2.734 | -1.367 | -2.188, -1.883, -1.617, -1.578, -1.906, -1.938, -2.047, -1.609 |
| 356 | `model.audio_tower.layers.3.self_attn.post.input_max` |  | bfloat16 | 16.38 | 16.38 | 0 | 16.38 | 16.38 | 16.38 |
| 357 | `model.audio_tower.layers.3.self_attn.post.input_min` |  | bfloat16 | 16.5 | -16.5 | 0 | -16.5 | -16.5 | -16.5 |
| 358 | `model.audio_tower.layers.3.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.78 | -0.001032 | 0.02907 | -0.1602 | 0.08008 | 0, 0.02856, 0.02856, 0, -0.02856, 0, 0, -0.02856 |
| 359 | `model.audio_tower.layers.3.self_attn.post.output_max` |  | bfloat16 | 46 | 46 | 0 | 46 | 46 | 46 |
| 360 | `model.audio_tower.layers.3.self_attn.post.output_min` |  | bfloat16 | 46.25 | -46.25 | 0 | -46.25 | -46.25 | -46.25 |
| 361 | `model.audio_tower.layers.3.self_attn.q_proj.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 362 | `model.audio_tower.layers.3.self_attn.q_proj.input_min` |  | bfloat16 | 11.81 | -11.81 | 0 | -11.81 | -11.81 | -11.81 |
| 363 | `model.audio_tower.layers.3.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.78 | -0.002309 | 0.02899 | -0.08887 | 0.04443 | 0.03442, 0, -0.03442, 0.03442, 0.03442, 0, -0.06885, -0.03442 |
| 364 | `model.audio_tower.layers.3.self_attn.q_proj.output_max` |  | bfloat16 | 18.62 | 18.62 | 0 | 18.62 | 18.62 | 18.62 |
| 365 | `model.audio_tower.layers.3.self_attn.q_proj.output_min` |  | bfloat16 | 18.75 | -18.75 | 0 | -18.75 | -18.75 | -18.75 |
| 366 | `model.audio_tower.layers.3.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.006865 | 0.03049 | -0.2637 | 0.2969 | -0.1406, -0.1338, -0.126, -0.1187, -0.1108, -0.104, -0.09619, -0.09033 |
| 367 | `model.audio_tower.layers.3.self_attn.v_proj.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 368 | `model.audio_tower.layers.3.self_attn.v_proj.input_min` |  | bfloat16 | 11.81 | -11.81 | 0 | -11.81 | -11.81 | -11.81 |
| 369 | `model.audio_tower.layers.3.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.93 | -0.001678 | 0.02918 | -0.1807 | 0.09033 | 0, 0, 0, -0.03076, 0, 0.03076, 0.03076, 0 |
| 370 | `model.audio_tower.layers.3.self_attn.v_proj.output_max` |  | bfloat16 | 18.62 | 18.62 | 0 | 18.62 | 18.62 | 18.62 |
| 371 | `model.audio_tower.layers.3.self_attn.v_proj.output_min` |  | bfloat16 | 18.75 | -18.75 | 0 | -18.75 | -18.75 | -18.75 |
| 372 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.19 | 12.19 | 0 | 12.19 | 12.19 | 12.19 |
| 373 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.31 | -12.31 | 0 | -12.31 | -12.31 | -12.31 |
| 374 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.64 | -0.007722 | 0.02858 | -0.1011 | 0.05054 | 0.02661, 0, 0.02661, 0, -0.02661, 0, 0.02661, 0.02661 |
| 375 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 25.75 | 25.75 | 0 | 25.75 | 25.75 | 25.75 |
| 376 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 26 | -26 | 0 | -26 | -26 | -26 |
| 377 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 9.438 | 9.438 | 0 | 9.438 | 9.438 | 9.438 |
| 378 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 9.562 | -9.562 | 0 | -9.562 | -9.562 | -9.562 |
| 379 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.79 | -2.024e-05 | 0.01453 | -0.08887 | 0.04443 | -0.0155, 0.0155, 0, 0, 0.0155, 0.0155, 0.0155, 0 |
| 380 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 47 | 47 | 0 | 47 | 47 | 47 |
| 381 | `model.audio_tower.layers.4.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 47.5 | -47.5 | 0 | -47.5 | -47.5 | -47.5 |
| 382 | `model.audio_tower.layers.4.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 107 | 2.493 | 2.227 | -7.219 | 34.75 | 0.2891, 1.5, 1.023, 1.719, 4.75, 0.9844, -2.062, 2.406 |
| 383 | `model.audio_tower.layers.4.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 369.5 | 9.541 | 6.506 | -18.88 | 40.5 | 10.94, 16.62, -5.312, 11.75, 9.625, 3.719, 12.06, 25.25 |
| 384 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 13.19 | 13.19 | 0 | 13.19 | 13.19 | 13.19 |
| 385 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 13.31 | -13.31 | 0 | -13.31 | -13.31 | -13.31 |
| 386 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.66 | -0.008112 | 0.02848 | -0.09668 | 0.04834 | 0.03003, -0.03003, 0.03003, -0.03003, -0.06006, 0, 0, 0.03003 |
| 387 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 29.75 | 29.75 | 0 | 29.75 | 29.75 | 29.75 |
| 388 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 30 | -30 | 0 | -30 | -30 | -30 |
| 389 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 12.25 | 12.25 | 0 | 12.25 | 12.25 | 12.25 |
| 390 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 12.31 | -12.31 | 0 | -12.31 | -12.31 | -12.31 |
| 391 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.73 | -0.0001839 | 0.01452 | -0.06787 | 0.03394 | -0.02319, 0, 0, -0.02319, 0.02319, 0, -0.04639, 0 |
| 392 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 74.5 | 74.5 | 0 | 74.5 | 74.5 | 74.5 |
| 393 | `model.audio_tower.layers.4.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 75 | -75 | 0 | -75 | -75 | -75 |
| 394 | `model.audio_tower.layers.4.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 223.7 | 5.089 | 4.795 | -13.56 | 41.25 | 0.1216, 3.422, 1.477, 5.906, -11.69, 1.5, 13.31, 0.2129 |
| 395 | `model.audio_tower.layers.4.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 134.8 | 4.024 | 1.248 | -5.031 | 6.781 | 1.711, 6.781, 4.438, 4, 3.469, 3.094, 5.344, 4.25 |
| 396 | `model.audio_tower.layers.4.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 119.5 | 3.456 | 1.415 | -6.812 | 12.12 | 2.969, 3.297, -5.375, 2.125, 3.203, -4.125, 5.094, 3.359 |
| 397 | `model.audio_tower.layers.4.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 31.99 | 0.00926 | 0.4471 | -10.81 | 2.109 | 0.168, 0.2158, 0.3594, 0.377, 0.4883, -0.126, -0.4395, -0.7305 |
| 398 | `model.audio_tower.layers.4.lconv1d.linear_end.input_max` |  | bfloat16 | 10.81 | 10.81 | 0 | 10.81 | 10.81 | 10.81 |
| 399 | `model.audio_tower.layers.4.lconv1d.linear_end.input_min` |  | bfloat16 | 10.88 | -10.88 | 0 | -10.88 | -10.88 | -10.88 |
| 400 | `model.audio_tower.layers.4.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.68 | 0.0009373 | 0.02897 | -0.1797 | 0.08984 | -0.04395, 0, 0, 0.04395, 0, 0.04395, 0, 0 |
| 401 | `model.audio_tower.layers.4.lconv1d.linear_end.output_max` |  | bfloat16 | 7.75 | 7.75 | 0 | 7.75 | 7.75 | 7.75 |
| 402 | `model.audio_tower.layers.4.lconv1d.linear_end.output_min` |  | bfloat16 | 7.812 | -7.812 | 0 | -7.812 | -7.812 | -7.812 |
| 403 | `model.audio_tower.layers.4.lconv1d.linear_start.input_max` |  | bfloat16 | 10.31 | 10.31 | 0 | 10.31 | 10.31 | 10.31 |
| 404 | `model.audio_tower.layers.4.lconv1d.linear_start.input_min` |  | bfloat16 | 10.44 | -10.44 | 0 | -10.44 | -10.44 | -10.44 |
| 405 | `model.audio_tower.layers.4.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.96 | 3.563e-06 | 0.03104 | -0.1709 | 0.1494 | -0.04565, -0.04565, -0.009094, 0.01819, -0.009094, 0.009094, 0.009094, 0 |
| 406 | `model.audio_tower.layers.4.lconv1d.linear_start.output_max` |  | bfloat16 | 23.62 | 23.62 | 0 | 23.62 | 23.62 | 23.62 |
| 407 | `model.audio_tower.layers.4.lconv1d.linear_start.output_min` |  | bfloat16 | 23.75 | -23.75 | 0 | -23.75 | -23.75 | -23.75 |
| 408 | `model.audio_tower.layers.4.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 130.2 | 3.546 | 1.998 | -8.188 | 11.31 | 3.016, 6.25, 3.594, 2.078, 3.156, 2.609, 6.438, 2.828 |
| 409 | `model.audio_tower.layers.4.norm_out.weight` | 1024 | bfloat16 | 297.1 | 7.306 | 5.73 | -1.453 | 19.62 | 2.047, 17, 17.12, 1.914, 2.406, 17.5, 1.039, 0.6406 |
| 410 | `model.audio_tower.layers.4.norm_post_attn.weight` | 1024 | bfloat16 | 58.84 | 1.256 | 1.344 | -5.562 | 17 | 2.297, 0.5, 1.438, 3.062, 0.7773, -1.359, 0.418, 2.453 |
| 411 | `model.audio_tower.layers.4.norm_pre_attn.weight` | 1024 | bfloat16 | 186.2 | 4.619 | 3.54 | -3.391 | 29.25 | 13.88, 4.406, 2.656, 7.031, 2.578, 3.672, 4.219, 7.094 |
| 412 | `model.audio_tower.layers.4.self_attn.k_proj.input_max` |  | bfloat16 | 10.56 | 10.56 | 0 | 10.56 | 10.56 | 10.56 |
| 413 | `model.audio_tower.layers.4.self_attn.k_proj.input_min` |  | bfloat16 | 10.62 | -10.62 | 0 | -10.62 | -10.62 | -10.62 |
| 414 | `model.audio_tower.layers.4.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.44 | -0.001693 | 0.0287 | -0.09375 | 0.04688 | 0, 0, -0.03149, 0.03149, 0.03149, 0.03149, 0, 0.03149 |
| 415 | `model.audio_tower.layers.4.self_attn.k_proj.output_max` |  | bfloat16 | 18.88 | 18.88 | 0 | 18.88 | 18.88 | 18.88 |
| 416 | `model.audio_tower.layers.4.self_attn.k_proj.output_min` |  | bfloat16 | 19 | -19 | 0 | -19 | -19 | -19 |
| 417 | `model.audio_tower.layers.4.self_attn.per_dim_scale` | 128 | bfloat16 | 23.69 | -2.077 | 0.2672 | -2.891 | -1.266 | -1.836, -2.094, -1.266, -2.141, -2.094, -2.391, -2.062, -2.328 |
| 418 | `model.audio_tower.layers.4.self_attn.post.input_max` |  | bfloat16 | 18.38 | 18.38 | 0 | 18.38 | 18.38 | 18.38 |
| 419 | `model.audio_tower.layers.4.self_attn.post.input_min` |  | bfloat16 | 18.5 | -18.5 | 0 | -18.5 | -18.5 | -18.5 |
| 420 | `model.audio_tower.layers.4.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.79 | -0.0009986 | 0.02908 | -0.2295 | 0.1147 | -0.03369, 0.03369, -0.06738, -0.06738, -0.06738, 0.03369, -0.03369, -0.03369 |
| 421 | `model.audio_tower.layers.4.self_attn.post.output_max` |  | bfloat16 | 53 | 53 | 0 | 53 | 53 | 53 |
| 422 | `model.audio_tower.layers.4.self_attn.post.output_min` |  | bfloat16 | 53.5 | -53.5 | 0 | -53.5 | -53.5 | -53.5 |
| 423 | `model.audio_tower.layers.4.self_attn.q_proj.input_max` |  | bfloat16 | 10.56 | 10.56 | 0 | 10.56 | 10.56 | 10.56 |
| 424 | `model.audio_tower.layers.4.self_attn.q_proj.input_min` |  | bfloat16 | 10.62 | -10.62 | 0 | -10.62 | -10.62 | -10.62 |
| 425 | `model.audio_tower.layers.4.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.71 | -0.002952 | 0.02887 | -0.08301 | 0.0415 | 0, 0, -0.05664, -0.05664, 0, -0.02832, 0, -0.02832 |
| 426 | `model.audio_tower.layers.4.self_attn.q_proj.output_max` |  | bfloat16 | 18.88 | 18.88 | 0 | 18.88 | 18.88 | 18.88 |
| 427 | `model.audio_tower.layers.4.self_attn.q_proj.output_min` |  | bfloat16 | 19 | -19 | 0 | -19 | -19 | -19 |
| 428 | `model.audio_tower.layers.4.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.006339 | 0.0306 | -0.167 | 0.1855 | -0.1279, -0.127, -0.1104, -0.08057, -0.04956, -0.02588, -0.008179, 0.005341 |
| 429 | `model.audio_tower.layers.4.self_attn.v_proj.input_max` |  | bfloat16 | 10.56 | 10.56 | 0 | 10.56 | 10.56 | 10.56 |
| 430 | `model.audio_tower.layers.4.self_attn.v_proj.input_min` |  | bfloat16 | 10.62 | -10.62 | 0 | -10.62 | -10.62 | -10.62 |
| 431 | `model.audio_tower.layers.4.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.89 | -0.0008969 | 0.02918 | -0.09473 | 0.04736 | 0.03174, 0, 0, -0.03174, 0, 0, -0.03174, 0 |
| 432 | `model.audio_tower.layers.4.self_attn.v_proj.output_max` |  | bfloat16 | 18.88 | 18.88 | 0 | 18.88 | 18.88 | 18.88 |
| 433 | `model.audio_tower.layers.4.self_attn.v_proj.output_min` |  | bfloat16 | 19 | -19 | 0 | -19 | -19 | -19 |
| 434 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.81 | 12.81 | 0 | 12.81 | 12.81 | 12.81 |
| 435 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.88 | -12.88 | 0 | -12.88 | -12.88 | -12.88 |
| 436 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.74 | -0.008283 | 0.02847 | -0.07764 | 0.03882 | 0.02881, -0.05762, 0, 0, 0, 0, -0.05762, 0 |
| 437 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 29.5 | 29.5 | 0 | 29.5 | 29.5 | 29.5 |
| 438 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 29.75 | -29.75 | 0 | -29.75 | -29.75 | -29.75 |
| 439 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 10.38 | 10.38 | 0 | 10.38 | 10.38 | 10.38 |
| 440 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 10.44 | -10.44 | 0 | -10.44 | -10.44 | -10.44 |
| 441 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.72 | -0.0001599 | 0.0145 | -0.1689 | 0.08447 | 0, 0, 0, 0.05444, 0, 0, 0, 0.05444 |
| 442 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 58.25 | 58.25 | 0 | 58.25 | 58.25 | 58.25 |
| 443 | `model.audio_tower.layers.5.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 58.75 | -58.75 | 0 | -58.75 | -58.75 | -58.75 |
| 444 | `model.audio_tower.layers.5.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 115.2 | 2.762 | 2.312 | -6.312 | 28.5 | 0.02234, 3.453, 0.4941, 2.609, 6.75, 0.7812, 5.969, 0.2275 |
| 445 | `model.audio_tower.layers.5.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 626.2 | 13.24 | 14.42 | -50.5 | 79.5 | 11.81, 4.531, 2.453, 19.75, 11.5, 1.25, 30.88, 42 |
| 446 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 12.88 | 12.88 | 0 | 12.88 | 12.88 | 12.88 |
| 447 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 12.94 | -12.94 | 0 | -12.94 | -12.94 | -12.94 |
| 448 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.5 | -0.008179 | 0.02838 | -0.1104 | 0.05518 | -0.03101, 0, 0, 0, 0, -0.06201, -0.03101, 0 |
| 449 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 31 | 31 | 0 | 31 | 31 | 31 |
| 450 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 31.25 | -31.25 | 0 | -31.25 | -31.25 | -31.25 |
| 451 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 8.438 | 8.438 | 0 | 8.438 | 8.438 | 8.438 |
| 452 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 8.5 | -8.5 | 0 | -8.5 | -8.5 | -8.5 |
| 453 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.65 | -0.0004005 | 0.01447 | -0.05371 | 0.02686 | 0.01599, 0, 0.01599, -0.01599, 0, 0, 0, 0.01599 |
| 454 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 35.75 | 35.75 | 0 | 35.75 | 35.75 | 35.75 |
| 455 | `model.audio_tower.layers.5.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 36 | -36 | 0 | -36 | -36 | -36 |
| 456 | `model.audio_tower.layers.5.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 177.3 | 4.262 | 3.543 | -12.94 | 40.25 | 0.4648, 6.219, 2.234, 4.719, 6.156, 1.852, 10.44, 0.03149 |
| 457 | `model.audio_tower.layers.5.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 182.7 | 4.87 | 2.983 | -10.5 | 21.25 | 6.406, 3.938, 2.953, 7.438, 2.109, 0.5977, 3.266, 13 |
| 458 | `model.audio_tower.layers.5.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 149.7 | 4.062 | 2.322 | -10.94 | 17.38 | 6.094, 3.875, 0.8867, 5.5, 3.094, 2.969, 4.031, 6.531 |
| 459 | `model.audio_tower.layers.5.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.007294 | 0.4472 | -2.328 | 2.047 | -0.2598, 0.4941, 0.7031, -0.1377, 0.1235, 0.05957, -0.209, -0.582 |
| 460 | `model.audio_tower.layers.5.lconv1d.linear_end.input_max` |  | bfloat16 | 13.81 | 13.81 | 0 | 13.81 | 13.81 | 13.81 |
| 461 | `model.audio_tower.layers.5.lconv1d.linear_end.input_min` |  | bfloat16 | 13.94 | -13.94 | 0 | -13.94 | -13.94 | -13.94 |
| 462 | `model.audio_tower.layers.5.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.78 | 1.83e-05 | 0.02908 | -0.1641 | 0.08203 | 0, 0.03394, 0, -0.06787, 0.03394, 0, 0, 0 |
| 463 | `model.audio_tower.layers.5.lconv1d.linear_end.output_max` |  | bfloat16 | 8 | 8 | 0 | 8 | 8 | 8 |
| 464 | `model.audio_tower.layers.5.lconv1d.linear_end.output_min` |  | bfloat16 | 8.062 | -8.062 | 0 | -8.062 | -8.062 | -8.062 |
| 465 | `model.audio_tower.layers.5.lconv1d.linear_start.input_max` |  | bfloat16 | 11.44 | 11.44 | 0 | 11.44 | 11.44 | 11.44 |
| 466 | `model.audio_tower.layers.5.lconv1d.linear_start.input_min` |  | bfloat16 | 11.5 | -11.5 | 0 | -11.5 | -11.5 | -11.5 |
| 467 | `model.audio_tower.layers.5.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.95 | -0.000133 | 0.03103 | -0.1562 | 0.1367 | 0.02075, -0.01038, -0.01038, 0.02075, 0.01038, -0.0415, -0.02075, 0.03113 |
| 468 | `model.audio_tower.layers.5.lconv1d.linear_start.output_max` |  | bfloat16 | 24.88 | 24.88 | 0 | 24.88 | 24.88 | 24.88 |
| 469 | `model.audio_tower.layers.5.lconv1d.linear_start.output_min` |  | bfloat16 | 25.12 | -25.12 | 0 | -25.12 | -25.12 | -25.12 |
| 470 | `model.audio_tower.layers.5.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 271.7 | 5.995 | 6.015 | -19.5 | 42.25 | 20.25, 2.5, 1.531, 9.688, 3.688, -0.1582, 5.062, 16.75 |
| 471 | `model.audio_tower.layers.5.norm_out.weight` | 1024 | bfloat16 | 252.9 | 7.156 | 3.353 | -10.94 | 16.5 | 7.375, 9.5, 11.69, 4.969, 6.844, 12.31, 5.25, 9.938 |
| 472 | `model.audio_tower.layers.5.norm_post_attn.weight` | 1024 | bfloat16 | 71.98 | 1.355 | 1.796 | -6.906 | 12.19 | 1.117, 0.1729, 0.02576, 2.016, 4.656, 0.02014, 2.875, -0.06348 |
| 473 | `model.audio_tower.layers.5.norm_pre_attn.weight` | 1024 | bfloat16 | 274.4 | 5.516 | 6.568 | -1.367 | 34.75 | 19, 1.109, 0.373, 8.375, 4.719, 0.6797, 5.688, 21.25 |
| 474 | `model.audio_tower.layers.5.self_attn.k_proj.input_max` |  | bfloat16 | 9.5 | 9.5 | 0 | 9.5 | 9.5 | 9.5 |
| 475 | `model.audio_tower.layers.5.self_attn.k_proj.input_min` |  | bfloat16 | 9.562 | -9.562 | 0 | -9.562 | -9.562 | -9.562 |
| 476 | `model.audio_tower.layers.5.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.45 | -0.001749 | 0.02871 | -0.1001 | 0.05005 | 0.03467, 0, 0.03467, 0.03467, 0, 0, 0, 0 |
| 477 | `model.audio_tower.layers.5.self_attn.k_proj.output_max` |  | bfloat16 | 16.88 | 16.88 | 0 | 16.88 | 16.88 | 16.88 |
| 478 | `model.audio_tower.layers.5.self_attn.k_proj.output_min` |  | bfloat16 | 17 | -17 | 0 | -17 | -17 | -17 |
| 479 | `model.audio_tower.layers.5.self_attn.per_dim_scale` | 128 | bfloat16 | 25.15 | -2.186 | 0.4068 | -3.188 | -1.438 | -1.852, -1.578, -1.906, -2.203, -2.094, -2.094, -2.531, -2.359 |
| 480 | `model.audio_tower.layers.5.self_attn.post.input_max` |  | bfloat16 | 15.56 | 15.56 | 0 | 15.56 | 15.56 | 15.56 |
| 481 | `model.audio_tower.layers.5.self_attn.post.input_min` |  | bfloat16 | 15.69 | -15.69 | 0 | -15.69 | -15.69 | -15.69 |
| 482 | `model.audio_tower.layers.5.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.78 | -0.00153 | 0.02905 | -0.1309 | 0.06543 | 0.02893, -0.02893, 0, -0.02893, 0, -0.02893, 0, -0.02893 |
| 483 | `model.audio_tower.layers.5.self_attn.post.output_max` |  | bfloat16 | 50.5 | 50.5 | 0 | 50.5 | 50.5 | 50.5 |
| 484 | `model.audio_tower.layers.5.self_attn.post.output_min` |  | bfloat16 | 50.75 | -50.75 | 0 | -50.75 | -50.75 | -50.75 |
| 485 | `model.audio_tower.layers.5.self_attn.q_proj.input_max` |  | bfloat16 | 9.5 | 9.5 | 0 | 9.5 | 9.5 | 9.5 |
| 486 | `model.audio_tower.layers.5.self_attn.q_proj.input_min` |  | bfloat16 | 9.562 | -9.562 | 0 | -9.562 | -9.562 | -9.562 |
| 487 | `model.audio_tower.layers.5.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.8 | -0.005033 | 0.02866 | -0.09473 | 0.04736 | 0, -0.06128, -0.03064, -0.06128, 0, -0.03064, 0, 0 |
| 488 | `model.audio_tower.layers.5.self_attn.q_proj.output_max` |  | bfloat16 | 16.88 | 16.88 | 0 | 16.88 | 16.88 | 16.88 |
| 489 | `model.audio_tower.layers.5.self_attn.q_proj.output_min` |  | bfloat16 | 17 | -17 | 0 | -17 | -17 | -17 |
| 490 | `model.audio_tower.layers.5.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.007838 | 0.03025 | -0.1348 | 0.1118 | -0.05859, -0.0542, -0.05078, -0.0481, -0.04541, -0.04395, -0.04248, -0.04126 |
| 491 | `model.audio_tower.layers.5.self_attn.v_proj.input_max` |  | bfloat16 | 9.5 | 9.5 | 0 | 9.5 | 9.5 | 9.5 |
| 492 | `model.audio_tower.layers.5.self_attn.v_proj.input_min` |  | bfloat16 | 9.562 | -9.562 | 0 | -9.562 | -9.562 | -9.562 |
| 493 | `model.audio_tower.layers.5.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.57 | -0.001004 | 0.02886 | -0.08838 | 0.04419 | 0, 0.03516, 0, 0, 0.03516, -0.03516, 0, 0.03516 |
| 494 | `model.audio_tower.layers.5.self_attn.v_proj.output_max` |  | bfloat16 | 16.88 | 16.88 | 0 | 16.88 | 16.88 | 16.88 |
| 495 | `model.audio_tower.layers.5.self_attn.v_proj.output_min` |  | bfloat16 | 17 | -17 | 0 | -17 | -17 | -17 |
| 496 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 497 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 498 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.56 | -0.00872 | 0.02824 | -0.08008 | 0.04004 | -0.02759, 0, 0, 0, -0.05518, 0.02759, 0, -0.05518 |
| 499 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 25.5 | 25.5 | 0 | 25.5 | 25.5 | 25.5 |
| 500 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 25.75 | -25.75 | 0 | -25.75 | -25.75 | -25.75 |
| 501 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 7.938 | 7.938 | 0 | 7.938 | 7.938 | 7.938 |
| 502 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 8 | -8 | 0 | -8 | -8 | -8 |
| 503 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.01 | -0.0004332 | 0.01464 | -0.09766 | 0.04883 | 0, 0.01459, 0, -0.01459, 0, 0, 0, 0.01459 |
| 504 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 29.25 | 29.25 | 0 | 29.25 | 29.25 | 29.25 |
| 505 | `model.audio_tower.layers.6.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 29.5 | -29.5 | 0 | -29.5 | -29.5 | -29.5 |
| 506 | `model.audio_tower.layers.6.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 63.54 | 1.61 | 1.163 | -3.141 | 18.62 | 0.4922, 1.797, 0.5781, 2.312, 2.531, 0.6055, 2.781, 0.2754 |
| 507 | `model.audio_tower.layers.6.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 422.1 | 10.29 | 8.253 | -20.38 | 46 | 15.56, 3.609, 1.977, 23.25, 5.719, 0.6328, 10.19, 13.38 |
| 508 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 14.44 | 14.44 | 0 | 14.44 | 14.44 | 14.44 |
| 509 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 14.56 | -14.56 | 0 | -14.56 | -14.56 | -14.56 |
| 510 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.42 | -0.008828 | 0.02814 | -0.09033 | 0.04517 | 0, -0.05615, -0.05615, -0.02808, -0.05615, 0, 0.02808, -0.05615 |
| 511 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 33 | 33 | 0 | 33 | 33 | 33 |
| 512 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 33.25 | -33.25 | 0 | -33.25 | -33.25 | -33.25 |
| 513 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 11.38 | 11.38 | 0 | 11.38 | 11.38 | 11.38 |
| 514 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 11.44 | -11.44 | 0 | -11.44 | -11.44 | -11.44 |
| 515 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.37 | -0.001275 | 0.01477 | -0.04517 | 0.02258 | -0.01746, -0.01746, 0, 0, 0, -0.01746, 0.01746, 0.01746 |
| 516 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 76.5 | 76.5 | 0 | 76.5 | 76.5 | 76.5 |
| 517 | `model.audio_tower.layers.6.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 77 | -77 | 0 | -77 | -77 | -77 |
| 518 | `model.audio_tower.layers.6.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 210 | 3.621 | 5.477 | -24.12 | 34.5 | 8.562, 0.7227, -0.01941, 10.94, 12.69, 0.4766, 4.5, 1 |
| 519 | `model.audio_tower.layers.6.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 312.8 | 8.429 | 4.953 | -6.281 | 26.88 | 7.156, 2.969, 2.297, 10.94, 13.19, 0.3594, 13.56, 16 |
| 520 | `model.audio_tower.layers.6.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 136.2 | 3.802 | 1.913 | -9.062 | 11.94 | 5.531, 4.688, 9.25, 3.781, 4.375, 3.188, 3.047, 2.359 |
| 521 | `model.audio_tower.layers.6.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | 0.003369 | 0.4472 | -1.359 | 2.719 | 0.06982, -0.1099, 0.1523, 0.4863, -0.5273, 0.04028, 0.006531, 0.08252 |
| 522 | `model.audio_tower.layers.6.lconv1d.linear_end.input_max` |  | bfloat16 | 14.44 | 14.44 | 0 | 14.44 | 14.44 | 14.44 |
| 523 | `model.audio_tower.layers.6.lconv1d.linear_end.input_min` |  | bfloat16 | 14.56 | -14.56 | 0 | -14.56 | -14.56 | -14.56 |
| 524 | `model.audio_tower.layers.6.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.67 | 0.001147 | 0.02895 | -0.1206 | 0.0603 | 0, 0.04199, -0.04199, -0.04199, 0, 0, 0, 0.04199 |
| 525 | `model.audio_tower.layers.6.lconv1d.linear_end.output_max` |  | bfloat16 | 7.812 | 7.812 | 0 | 7.812 | 7.812 | 7.812 |
| 526 | `model.audio_tower.layers.6.lconv1d.linear_end.output_min` |  | bfloat16 | 7.875 | -7.875 | 0 | -7.875 | -7.875 | -7.875 |
| 527 | `model.audio_tower.layers.6.lconv1d.linear_start.input_max` |  | bfloat16 | 10.88 | 10.88 | 0 | 10.88 | 10.88 | 10.88 |
| 528 | `model.audio_tower.layers.6.lconv1d.linear_start.input_min` |  | bfloat16 | 10.94 | -10.94 | 0 | -10.94 | -10.94 | -10.94 |
| 529 | `model.audio_tower.layers.6.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.93 | -0.000354 | 0.03102 | -0.1367 | 0.1235 | 0, -0.0105, 0.0105, -0.021, -0.021, -0.04199, 0.03149, 0.0105 |
| 530 | `model.audio_tower.layers.6.lconv1d.linear_start.output_max` |  | bfloat16 | 22.38 | 22.38 | 0 | 22.38 | 22.38 | 22.38 |
| 531 | `model.audio_tower.layers.6.lconv1d.linear_start.output_min` |  | bfloat16 | 22.62 | -22.62 | 0 | -22.62 | -22.62 | -22.62 |
| 532 | `model.audio_tower.layers.6.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 195.2 | 4.889 | 3.649 | -11.25 | 28 | 8.875, 1.891, 1.57, 6.094, 1.977, 0.9297, 3.297, 8.875 |
| 533 | `model.audio_tower.layers.6.norm_out.weight` | 1024 | bfloat16 | 145.6 | 3.824 | 2.465 | -5.688 | 27.12 | 1.25, 6.062, 7.438, 0.9023, 1.836, 8.312, 2.016, 3.828 |
| 534 | `model.audio_tower.layers.6.norm_post_attn.weight` | 1024 | bfloat16 | 45.34 | 0.7744 | 1.187 | -3.703 | 12.38 | 3.453, -0.08594, 0.04346, 2.281, 0.5312, 0.08154, -0.1445, 0.6875 |
| 535 | `model.audio_tower.layers.6.norm_pre_attn.weight` | 1024 | bfloat16 | 302.4 | 5.804 | 7.462 | -24.88 | 34.25 | 24.38, 1.133, 0.4473, 9.562, 1.148, 0.668, 2.234, 34 |
| 536 | `model.audio_tower.layers.6.self_attn.k_proj.input_max` |  | bfloat16 | 10.25 | 10.25 | 0 | 10.25 | 10.25 | 10.25 |
| 537 | `model.audio_tower.layers.6.self_attn.k_proj.input_min` |  | bfloat16 | 10.31 | -10.31 | 0 | -10.31 | -10.31 | -10.31 |
| 538 | `model.audio_tower.layers.6.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.67 | -0.00193 | 0.02891 | -0.08984 | 0.04492 | -0.03882, 0, 0, 0.03882, -0.07764, 0, 0, 0.03882 |
| 539 | `model.audio_tower.layers.6.self_attn.k_proj.output_max` |  | bfloat16 | 15.38 | 15.38 | 0 | 15.38 | 15.38 | 15.38 |
| 540 | `model.audio_tower.layers.6.self_attn.k_proj.output_min` |  | bfloat16 | 15.5 | -15.5 | 0 | -15.5 | -15.5 | -15.5 |
| 541 | `model.audio_tower.layers.6.self_attn.per_dim_scale` | 128 | bfloat16 | 23.2 | -2 | 0.4539 | -3.391 | -0.9141 | -2.219, -1.422, -2.078, -2.312, -2.062, -2.469, -2.469, -1.766 |
| 542 | `model.audio_tower.layers.6.self_attn.post.input_max` |  | bfloat16 | 14.94 | 14.94 | 0 | 14.94 | 14.94 | 14.94 |
| 543 | `model.audio_tower.layers.6.self_attn.post.input_min` |  | bfloat16 | 15.06 | -15.06 | 0 | -15.06 | -15.06 | -15.06 |
| 544 | `model.audio_tower.layers.6.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.71 | -0.001545 | 0.02897 | -0.105 | 0.05249 | 0, 0.02734, 0.02734, -0.02734, 0, 0, 0.02734, 0.02734 |
| 545 | `model.audio_tower.layers.6.self_attn.post.output_max` |  | bfloat16 | 48.5 | 48.5 | 0 | 48.5 | 48.5 | 48.5 |
| 546 | `model.audio_tower.layers.6.self_attn.post.output_min` |  | bfloat16 | 49 | -49 | 0 | -49 | -49 | -49 |
| 547 | `model.audio_tower.layers.6.self_attn.q_proj.input_max` |  | bfloat16 | 10.25 | 10.25 | 0 | 10.25 | 10.25 | 10.25 |
| 548 | `model.audio_tower.layers.6.self_attn.q_proj.input_min` |  | bfloat16 | 10.31 | -10.31 | 0 | -10.31 | -10.31 | -10.31 |
| 549 | `model.audio_tower.layers.6.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.57 | -0.003054 | 0.02871 | -0.09131 | 0.04565 | 0, 0.03662, 0.03662, -0.03662, 0.03662, -0.03662, 0, -0.03662 |
| 550 | `model.audio_tower.layers.6.self_attn.q_proj.output_max` |  | bfloat16 | 15.38 | 15.38 | 0 | 15.38 | 15.38 | 15.38 |
| 551 | `model.audio_tower.layers.6.self_attn.q_proj.output_min` |  | bfloat16 | 15.5 | -15.5 | 0 | -15.5 | -15.5 | -15.5 |
| 552 | `model.audio_tower.layers.6.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.00895 | 0.02994 | -0.2207 | 0.2158 | -0.02039, -0.01471, -0.009766, -0.005615, -0.002579, 0.0001593, 0.002411, 0.00415 |
| 553 | `model.audio_tower.layers.6.self_attn.v_proj.input_max` |  | bfloat16 | 10.25 | 10.25 | 0 | 10.25 | 10.25 | 10.25 |
| 554 | `model.audio_tower.layers.6.self_attn.v_proj.input_min` |  | bfloat16 | 10.31 | -10.31 | 0 | -10.31 | -10.31 | -10.31 |
| 555 | `model.audio_tower.layers.6.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.72 | -0.0008705 | 0.02901 | -0.09863 | 0.04932 | 0.03394, -0.03394, 0, 0.03394, 0, 0, -0.03394, 0.03394 |
| 556 | `model.audio_tower.layers.6.self_attn.v_proj.output_max` |  | bfloat16 | 15.38 | 15.38 | 0 | 15.38 | 15.38 | 15.38 |
| 557 | `model.audio_tower.layers.6.self_attn.v_proj.output_min` |  | bfloat16 | 15.5 | -15.5 | 0 | -15.5 | -15.5 | -15.5 |
| 558 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 12.25 | 12.25 | 0 | 12.25 | 12.25 | 12.25 |
| 559 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 12.38 | -12.38 | 0 | -12.38 | -12.38 | -12.38 |
| 560 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.74 | -0.008559 | 0.02839 | -0.0874 | 0.0437 | -0.06128, -0.03064, -0.03064, -0.03064, 0, 0.03064, -0.03064, 0.03064 |
| 561 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 25.25 | 25.25 | 0 | 25.25 | 25.25 | 25.25 |
| 562 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 25.5 | -25.5 | 0 | -25.5 | -25.5 | -25.5 |
| 563 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 8.5 | 8.5 | 0 | 8.5 | 8.5 | 8.5 |
| 564 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 8.562 | -8.562 | 0 | -8.562 | -8.562 | -8.562 |
| 565 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 29.89 | -0.0003688 | 0.01458 | -0.1069 | 0.05347 | -0.03223, 0.01611, 0, 0, 0, 0, 0, 0.01611 |
| 566 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 42 | 42 | 0 | 42 | 42 | 42 |
| 567 | `model.audio_tower.layers.7.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 42.25 | -42.25 | 0 | -42.25 | -42.25 | -42.25 |
| 568 | `model.audio_tower.layers.7.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 91.33 | 2.158 | 1.869 | -5.062 | 17.25 | 2.109, 2.906, 1.359, 1.945, 4.375, 1.672, 1.898, 0.6016 |
| 569 | `model.audio_tower.layers.7.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 400.6 | 10.46 | 6.878 | -26.5 | 35 | 17.38, 4.969, 2.469, 22.88, 10.25, 1.492, 17.38, 11.06 |
| 570 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 13.12 | 13.12 | 0 | 13.12 | 13.12 | 13.12 |
| 571 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 13.25 | -13.25 | 0 | -13.25 | -13.25 | -13.25 |
| 572 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.61 | -0.00796 | 0.02849 | -0.09619 | 0.0481 | -0.05615, 0, 0.02808, 0.02808, -0.05615, -0.02808, 0, -0.02808 |
| 573 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 29.12 | 29.12 | 0 | 29.12 | 29.12 | 29.12 |
| 574 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 29.38 | -29.38 | 0 | -29.38 | -29.38 | -29.38 |
| 575 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 9.938 | 9.938 | 0 | 9.938 | 9.938 | 9.938 |
| 576 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 10 | -10 | 0 | -10 | -10 | -10 |
| 577 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.14 | -0.0003282 | 0.0147 | -0.1016 | 0.05078 | 0.01733, 0, -0.01733, 0, 0.01733, -0.01733, 0.01733, 0.01733 |
| 578 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 53.75 | 53.75 | 0 | 53.75 | 53.75 | 53.75 |
| 579 | `model.audio_tower.layers.7.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 54 | -54 | 0 | -54 | -54 | -54 |
| 580 | `model.audio_tower.layers.7.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 187.2 | 3.931 | 4.335 | -10.25 | 44 | 11.69, 3.344, -0.8984, 4.531, 7.625, 1.242, 2.531, 2.219 |
| 581 | `model.audio_tower.layers.7.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 257.7 | 7.175 | 3.657 | -18.5 | 16.25 | 7.875, 5.219, 3.531, 13, 6.312, 2.109, 10.56, 6.75 |
| 582 | `model.audio_tower.layers.7.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 132.4 | 3.4 | 2.361 | -7.562 | 43.25 | 3.828, 2.047, 4.188, 2.969, 3.844, 3.297, 6.156, 0.2949 |
| 583 | `model.audio_tower.layers.7.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | 0.006675 | 0.4472 | -3.297 | 1.898 | -0.105, -0.3906, -0.5234, -0.3691, -0.6875, 0.05005, -0.01611, 0.1099 |
| 584 | `model.audio_tower.layers.7.lconv1d.linear_end.input_max` |  | bfloat16 | 7.938 | 7.938 | 0 | 7.938 | 7.938 | 7.938 |
| 585 | `model.audio_tower.layers.7.lconv1d.linear_end.input_min` |  | bfloat16 | 8 | -8 | 0 | -8 | -8 | -8 |
| 586 | `model.audio_tower.layers.7.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.97 | 0.0006037 | 0.02926 | -0.1113 | 0.05566 | -0.03906, 0, 0, 0.03906, 0, 0.03906, 0.03906, -0.03906 |
| 587 | `model.audio_tower.layers.7.lconv1d.linear_end.output_max` |  | bfloat16 | 7.125 | 7.125 | 0 | 7.125 | 7.125 | 7.125 |
| 588 | `model.audio_tower.layers.7.lconv1d.linear_end.output_min` |  | bfloat16 | 7.156 | -7.156 | 0 | -7.156 | -7.156 | -7.156 |
| 589 | `model.audio_tower.layers.7.lconv1d.linear_start.input_max` |  | bfloat16 | 10.62 | 10.62 | 0 | 10.62 | 10.62 | 10.62 |
| 590 | `model.audio_tower.layers.7.lconv1d.linear_start.input_min` |  | bfloat16 | 10.75 | -10.75 | 0 | -10.75 | -10.75 | -10.75 |
| 591 | `model.audio_tower.layers.7.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.94 | -0.0007064 | 0.03102 | -0.1426 | 0.125 | 0.03711, 0, 0, 0.04932, -0.02466, 0.04932, -0.01233, -0.02466 |
| 592 | `model.audio_tower.layers.7.lconv1d.linear_start.output_max` |  | bfloat16 | 23.12 | 23.12 | 0 | 23.12 | 23.12 | 23.12 |
| 593 | `model.audio_tower.layers.7.lconv1d.linear_start.output_min` |  | bfloat16 | 23.25 | -23.25 | 0 | -23.25 | -23.25 | -23.25 |
| 594 | `model.audio_tower.layers.7.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 251.5 | 6.511 | 4.403 | -15.56 | 25.12 | 8.312, 2.922, 2.578, 12.19, 4.656, 2.766, 9.5, 9.812 |
| 595 | `model.audio_tower.layers.7.norm_out.weight` | 1024 | bfloat16 | 153.9 | 4.208 | 2.332 | -5.844 | 23.25 | 0.9688, 5.031, 6.719, 2.609, 4.562, 6.688, 4.875, 3.594 |
| 596 | `model.audio_tower.layers.7.norm_post_attn.weight` | 1024 | bfloat16 | 37.16 | 0.4866 | 1.055 | -5.969 | 22.25 | 1.625, 0.1885, -0.1348, 0.4121, 0.5195, -0.0332, -0.1748, 1.531 |
| 597 | `model.audio_tower.layers.7.norm_pre_attn.weight` | 1024 | bfloat16 | 316.2 | 6.443 | 7.494 | -18.62 | 40 | 12.69, 1.273, 1, 7.812, 1.812, 1.328, 7.719, 25 |
| 598 | `model.audio_tower.layers.7.self_attn.k_proj.input_max` |  | bfloat16 | 10.88 | 10.88 | 0 | 10.88 | 10.88 | 10.88 |
| 599 | `model.audio_tower.layers.7.self_attn.k_proj.input_min` |  | bfloat16 | 10.94 | -10.94 | 0 | -10.94 | -10.94 | -10.94 |
| 600 | `model.audio_tower.layers.7.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.45 | -0.00177 | 0.02871 | -0.1138 | 0.05688 | 0.02576, 0.02576, 0, 0.02576, 0, -0.02576, -0.05151, 0 |
| 601 | `model.audio_tower.layers.7.self_attn.k_proj.output_max` |  | bfloat16 | 17.38 | 17.38 | 0 | 17.38 | 17.38 | 17.38 |
| 602 | `model.audio_tower.layers.7.self_attn.k_proj.output_min` |  | bfloat16 | 17.5 | -17.5 | 0 | -17.5 | -17.5 | -17.5 |
| 603 | `model.audio_tower.layers.7.self_attn.per_dim_scale` | 128 | bfloat16 | 17.75 | -1.463 | 0.5689 | -3.031 | -0.08154 | -0.7656, -0.6016, -0.8906, -0.08154, -1.703, -1.672, -1.703, -1.297 |
| 604 | `model.audio_tower.layers.7.self_attn.post.input_max` |  | bfloat16 | 17 | 17 | 0 | 17 | 17 | 17 |
| 605 | `model.audio_tower.layers.7.self_attn.post.input_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 606 | `model.audio_tower.layers.7.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.77 | -0.001511 | 0.02903 | -0.1289 | 0.06445 | 0, -0.0603, 0.03015, -0.03015, 0.03015, -0.03015, 0.03015, -0.0603 |
| 607 | `model.audio_tower.layers.7.self_attn.post.output_max` |  | bfloat16 | 56.5 | 56.5 | 0 | 56.5 | 56.5 | 56.5 |
| 608 | `model.audio_tower.layers.7.self_attn.post.output_min` |  | bfloat16 | 57 | -57 | 0 | -57 | -57 | -57 |
| 609 | `model.audio_tower.layers.7.self_attn.q_proj.input_max` |  | bfloat16 | 10.88 | 10.88 | 0 | 10.88 | 10.88 | 10.88 |
| 610 | `model.audio_tower.layers.7.self_attn.q_proj.input_min` |  | bfloat16 | 10.94 | -10.94 | 0 | -10.94 | -10.94 | -10.94 |
| 611 | `model.audio_tower.layers.7.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.49 | -0.002729 | 0.02867 | -0.1523 | 0.07617 | -0.02832, 0.02832, 0.02832, 0, -0.02832, -0.02832, 0.02832, 0 |
| 612 | `model.audio_tower.layers.7.self_attn.q_proj.output_max` |  | bfloat16 | 17.38 | 17.38 | 0 | 17.38 | 17.38 | 17.38 |
| 613 | `model.audio_tower.layers.7.self_attn.q_proj.output_min` |  | bfloat16 | 17.5 | -17.5 | 0 | -17.5 | -17.5 | -17.5 |
| 614 | `model.audio_tower.layers.7.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.00399 | 0.03099 | -0.3223 | 0.5586 | -0.0007019, -0.007721, -0.01398, -0.01758, -0.01929, -0.01978, -0.01843, -0.01575 |
| 615 | `model.audio_tower.layers.7.self_attn.v_proj.input_max` |  | bfloat16 | 10.88 | 10.88 | 0 | 10.88 | 10.88 | 10.88 |
| 616 | `model.audio_tower.layers.7.self_attn.v_proj.input_min` |  | bfloat16 | 10.94 | -10.94 | 0 | -10.94 | -10.94 | -10.94 |
| 617 | `model.audio_tower.layers.7.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.88 | -0.0008935 | 0.02916 | -0.09619 | 0.0481 | -0.03394, 0, 0, 0.03394, 0, 0, 0, -0.03394 |
| 618 | `model.audio_tower.layers.7.self_attn.v_proj.output_max` |  | bfloat16 | 17.38 | 17.38 | 0 | 17.38 | 17.38 | 17.38 |
| 619 | `model.audio_tower.layers.7.self_attn.v_proj.output_min` |  | bfloat16 | 17.5 | -17.5 | 0 | -17.5 | -17.5 | -17.5 |
| 620 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 11.12 | 11.12 | 0 | 11.12 | 11.12 | 11.12 |
| 621 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 11.19 | -11.19 | 0 | -11.19 | -11.19 | -11.19 |
| 622 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.88 | -0.008558 | 0.02846 | -0.0752 | 0.0376 | -0.02917, -0.05835, 0, 0, 0.02917, 0.02917, 0, -0.02917 |
| 623 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 22.62 | 22.62 | 0 | 22.62 | 22.62 | 22.62 |
| 624 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 22.75 | -22.75 | 0 | -22.75 | -22.75 | -22.75 |
| 625 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 8.75 | 8.75 | 0 | 8.75 | 8.75 | 8.75 |
| 626 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 8.812 | -8.812 | 0 | -8.812 | -8.812 | -8.812 |
| 627 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.05 | -0.000485 | 0.01465 | -0.07666 | 0.03833 | 0, 0, -0.03271, -0.01636, -0.01636, 0, -0.03271, 0 |
| 628 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 38.5 | 38.5 | 0 | 38.5 | 38.5 | 38.5 |
| 629 | `model.audio_tower.layers.8.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 38.75 | -38.75 | 0 | -38.75 | -38.75 | -38.75 |
| 630 | `model.audio_tower.layers.8.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 93.89 | 2.112 | 2.038 | -5.938 | 20.62 | 2.172, 3.516, 1.375, 2.438, -4.938, 1.562, 1.836, 1.492 |
| 631 | `model.audio_tower.layers.8.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 379.9 | 9.243 | 7.453 | -31.62 | 39 | 21.62, 7, 4.219, 18.5, 5.75, 2.844, 11.12, -17.38 |
| 632 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 633 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 11.88 | -11.88 | 0 | -11.88 | -11.88 | -11.88 |
| 634 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.76 | -0.008001 | 0.02856 | -0.0752 | 0.0376 | 0.03052, 0.03052, 0, -0.03052, -0.06104, 0, -0.03052, 0.03052 |
| 635 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 27.75 | 27.75 | 0 | 27.75 | 27.75 | 27.75 |
| 636 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 28 | -28 | 0 | -28 | -28 | -28 |
| 637 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 9.75 | 9.75 | 0 | 9.75 | 9.75 | 9.75 |
| 638 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 9.812 | -9.812 | 0 | -9.812 | -9.812 | -9.812 |
| 639 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.03 | -0.0002641 | 0.01465 | -0.0752 | 0.0376 | 0, 0.02625, -0.02625, 0.02625, -0.02625, 0.02625, 0, 0 |
| 640 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 48.25 | 48.25 | 0 | 48.25 | 48.25 | 48.25 |
| 641 | `model.audio_tower.layers.8.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 48.75 | -48.75 | 0 | -48.75 | -48.75 | -48.75 |
| 642 | `model.audio_tower.layers.8.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 149.5 | 3.062 | 3.53 | -9.562 | 42.5 | 1.07, 3.734, 0.9688, -6.375, 5.812, 1.164, 1.977, 3.562 |
| 643 | `model.audio_tower.layers.8.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 250.9 | 7.192 | 3.126 | -11.06 | 16.38 | 12, 5.812, 4.375, 10.88, 5.719, 2.547, 10.38, 11.81 |
| 644 | `model.audio_tower.layers.8.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 121.6 | 3.38 | 1.74 | -5.812 | 10.12 | 2.641, 4.906, 3.125, -3.688, 6.375, 4.219, 3.828, 2.875 |
| 645 | `model.audio_tower.layers.8.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | -0.002101 | 0.4473 | -4.125 | 6.562 | -0.1001, -0.1338, -0.6055, -0.6719, 0.3594, -0.01636, -0.02136, 0.007996 |
| 646 | `model.audio_tower.layers.8.lconv1d.linear_end.input_max` |  | bfloat16 | 7.688 | 7.688 | 0 | 7.688 | 7.688 | 7.688 |
| 647 | `model.audio_tower.layers.8.lconv1d.linear_end.input_min` |  | bfloat16 | 7.75 | -7.75 | 0 | -7.75 | -7.75 | -7.75 |
| 648 | `model.audio_tower.layers.8.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 30.02 | 0.0006447 | 0.02931 | -0.1245 | 0.06226 | 0, 0.03857, 0.03857, 0.03857, 0.03857, 0.03857, 0.03857, 0 |
| 649 | `model.audio_tower.layers.8.lconv1d.linear_end.output_max` |  | bfloat16 | 7 | 7 | 0 | 7 | 7 | 7 |
| 650 | `model.audio_tower.layers.8.lconv1d.linear_end.output_min` |  | bfloat16 | 7.062 | -7.062 | 0 | -7.062 | -7.062 | -7.062 |
| 651 | `model.audio_tower.layers.8.lconv1d.linear_start.input_max` |  | bfloat16 | 10.88 | 10.88 | 0 | 10.88 | 10.88 | 10.88 |
| 652 | `model.audio_tower.layers.8.lconv1d.linear_start.input_min` |  | bfloat16 | 10.94 | -10.94 | 0 | -10.94 | -10.94 | -10.94 |
| 653 | `model.audio_tower.layers.8.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.95 | -0.001183 | 0.03101 | -0.1592 | 0.1387 | -0.01025, 0.01025, 0.02051, -0.04102, 0.01025, -0.06152, 0.02051, 0 |
| 654 | `model.audio_tower.layers.8.lconv1d.linear_start.output_max` |  | bfloat16 | 23.38 | 23.38 | 0 | 23.38 | 23.38 | 23.38 |
| 655 | `model.audio_tower.layers.8.lconv1d.linear_start.output_min` |  | bfloat16 | 23.62 | -23.62 | 0 | -23.62 | -23.62 | -23.62 |
| 656 | `model.audio_tower.layers.8.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 215.2 | 5.764 | 3.467 | -9.625 | 26.62 | 10, 4.281, 4.938, 8.438, 3.219, 4.031, 8.562, 18.25 |
| 657 | `model.audio_tower.layers.8.norm_out.weight` | 1024 | bfloat16 | 225.5 | 5.726 | 4.111 | -38.5 | 20.75 | 4.969, 7.094, 7.781, 5.406, 7.344, 7.219, 6.188, 4.531 |
| 658 | `model.audio_tower.layers.8.norm_post_attn.weight` | 1024 | bfloat16 | 72.27 | 0.9739 | 2.039 | -3.703 | 31.62 | 0.0003147, 0.373, -0.08252, 0.09326, 0.4121, -0.4277, 0.6719, 0.2891 |
| 659 | `model.audio_tower.layers.8.norm_pre_attn.weight` | 1024 | bfloat16 | 198.5 | 4.381 | 4.393 | -17.75 | 29.25 | 6.281, 2.422, 2.172, 5.188, -1.281, 2.562, 4.75, 6.406 |
| 660 | `model.audio_tower.layers.8.self_attn.k_proj.input_max` |  | bfloat16 | 9 | 9 | 0 | 9 | 9 | 9 |
| 661 | `model.audio_tower.layers.8.self_attn.k_proj.input_min` |  | bfloat16 | 9.062 | -9.062 | 0 | -9.062 | -9.062 | -9.062 |
| 662 | `model.audio_tower.layers.8.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.67 | -0.001401 | 0.02894 | -0.1104 | 0.05518 | -0.04102, 0.04102, 0.04102, 0, 0.04102, 0, 0.04102, 0.04102 |
| 663 | `model.audio_tower.layers.8.self_attn.k_proj.output_max` |  | bfloat16 | 13.94 | 13.94 | 0 | 13.94 | 13.94 | 13.94 |
| 664 | `model.audio_tower.layers.8.self_attn.k_proj.output_min` |  | bfloat16 | 14.06 | -14.06 | 0 | -14.06 | -14.06 | -14.06 |
| 665 | `model.audio_tower.layers.8.self_attn.per_dim_scale` | 128 | bfloat16 | 21.59 | -1.841 | 0.503 | -3.172 | -0.875 | -1.742, -2.422, -2.156, -1.977, -3.047, -1.984, -1.828, -2.281 |
| 666 | `model.audio_tower.layers.8.self_attn.post.input_max` |  | bfloat16 | 13.19 | 13.19 | 0 | 13.19 | 13.19 | 13.19 |
| 667 | `model.audio_tower.layers.8.self_attn.post.input_min` |  | bfloat16 | 13.31 | -13.31 | 0 | -13.31 | -13.31 | -13.31 |
| 668 | `model.audio_tower.layers.8.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.74 | -0.001289 | 0.02902 | -0.1108 | 0.05542 | 0.02722, -0.02722, 0.02722, 0, 0.02722, -0.02722, 0, 0 |
| 669 | `model.audio_tower.layers.8.self_attn.post.output_max` |  | bfloat16 | 38.5 | 38.5 | 0 | 38.5 | 38.5 | 38.5 |
| 670 | `model.audio_tower.layers.8.self_attn.post.output_min` |  | bfloat16 | 38.75 | -38.75 | 0 | -38.75 | -38.75 | -38.75 |
| 671 | `model.audio_tower.layers.8.self_attn.q_proj.input_max` |  | bfloat16 | 9 | 9 | 0 | 9 | 9 | 9 |
| 672 | `model.audio_tower.layers.8.self_attn.q_proj.input_min` |  | bfloat16 | 9.062 | -9.062 | 0 | -9.062 | -9.062 | -9.062 |
| 673 | `model.audio_tower.layers.8.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.54 | -0.001781 | 0.02879 | -0.1191 | 0.05957 | -0.03418, 0.03418, 0.03418, 0, 0, 0, -0.03418, 0.03418 |
| 674 | `model.audio_tower.layers.8.self_attn.q_proj.output_max` |  | bfloat16 | 13.94 | 13.94 | 0 | 13.94 | 13.94 | 13.94 |
| 675 | `model.audio_tower.layers.8.self_attn.q_proj.output_min` |  | bfloat16 | 14.06 | -14.06 | 0 | -14.06 | -14.06 | -14.06 |
| 676 | `model.audio_tower.layers.8.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.003796 | 0.03102 | -0.2129 | 0.2432 | -0.03394, -0.03076, -0.026, -0.021, -0.01501, -0.01001, -0.005219, -0.0002975 |
| 677 | `model.audio_tower.layers.8.self_attn.v_proj.input_max` |  | bfloat16 | 9 | 9 | 0 | 9 | 9 | 9 |
| 678 | `model.audio_tower.layers.8.self_attn.v_proj.input_min` |  | bfloat16 | 9.062 | -9.062 | 0 | -9.062 | -9.062 | -9.062 |
| 679 | `model.audio_tower.layers.8.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 30.03 | -0.00141 | 0.02929 | -0.09131 | 0.04565 | 0, 0, 0, 0, 0, 0, -0.0354, 0.0354 |
| 680 | `model.audio_tower.layers.8.self_attn.v_proj.output_max` |  | bfloat16 | 13.94 | 13.94 | 0 | 13.94 | 13.94 | 13.94 |
| 681 | `model.audio_tower.layers.8.self_attn.v_proj.output_min` |  | bfloat16 | 14.06 | -14.06 | 0 | -14.06 | -14.06 | -14.06 |
| 682 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_1.input_max` |  | bfloat16 | 7.75 | 7.75 | 0 | 7.75 | 7.75 | 7.75 |
| 683 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_1.input_min` |  | bfloat16 | 7.812 | -7.812 | 0 | -7.812 | -7.812 | -7.812 |
| 684 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.9 | -0.008641 | 0.02844 | -0.06836 | 0.03418 | -0.05908, 0, 0, 0, 0, -0.02954, -0.02954, 0.02954 |
| 685 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_1.output_max` |  | bfloat16 | 14.12 | 14.12 | 0 | 14.12 | 14.12 | 14.12 |
| 686 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_1.output_min` |  | bfloat16 | 14.25 | -14.25 | 0 | -14.25 | -14.25 | -14.25 |
| 687 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_2.input_max` |  | bfloat16 | 8.188 | 8.188 | 0 | 8.188 | 8.188 | 8.188 |
| 688 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_2.input_min` |  | bfloat16 | 8.25 | -8.25 | 0 | -8.25 | -8.25 | -8.25 |
| 689 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.1 | -0.0007139 | 0.01467 | -0.06885 | 0.03442 | 0, 0, -0.0166, 0, -0.0166, 0, 0, -0.0166 |
| 690 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_2.output_max` |  | bfloat16 | 39.75 | 39.75 | 0 | 39.75 | 39.75 | 39.75 |
| 691 | `model.audio_tower.layers.9.feed_forward1.ffw_layer_2.output_min` |  | bfloat16 | 40 | -40 | 0 | -40 | -40 | -40 |
| 692 | `model.audio_tower.layers.9.feed_forward1.post_layer_norm.weight` | 1024 | bfloat16 | 119.2 | 2.596 | 2.67 | -6.688 | 19.38 | 7.344, 2.859, 0.8516, -4.188, 4.906, -0.7969, 1.68, 2.172 |
| 693 | `model.audio_tower.layers.9.feed_forward1.pre_layer_norm.weight` | 1024 | bfloat16 | 271.8 | 5.536 | 6.445 | -40.75 | 52 | 16.75, 4.688, 2.891, 7.812, 4.406, 1.75, 9.438, 12.25 |
| 694 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_1.input_max` |  | bfloat16 | 12.5 | 12.5 | 0 | 12.5 | 12.5 | 12.5 |
| 695 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_1.input_min` |  | bfloat16 | 12.56 | -12.56 | 0 | -12.56 | -12.56 | -12.56 |
| 696 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_1.linear.weight` | 4096×1024 | bfloat16 | 60.8 | -0.008677 | 0.02838 | -0.07764 | 0.03882 | -0.03076, 0, -0.03076, 0.03076, 0, -0.03076, -0.03076, 0.03076 |
| 697 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_1.output_max` |  | bfloat16 | 30.12 | 30.12 | 0 | 30.12 | 30.12 | 30.12 |
| 698 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_1.output_min` |  | bfloat16 | 30.38 | -30.38 | 0 | -30.38 | -30.38 | -30.38 |
| 699 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_2.input_max` |  | bfloat16 | 13.5 | 13.5 | 0 | 13.5 | 13.5 | 13.5 |
| 700 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_2.input_min` |  | bfloat16 | 13.62 | -13.62 | 0 | -13.62 | -13.62 | -13.62 |
| 701 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_2.linear.weight` | 1024×4096 | bfloat16 | 30.22 | -0.0005754 | 0.01473 | -0.08057 | 0.04028 | 0, 0, 0.01746, 0, 0, 0.01746, 0.01746, -0.03491 |
| 702 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_2.output_max` |  | bfloat16 | 89.5 | 89.5 | 0 | 89.5 | 89.5 | 89.5 |
| 703 | `model.audio_tower.layers.9.feed_forward2.ffw_layer_2.output_min` |  | bfloat16 | 90 | -90 | 0 | -90 | -90 | -90 |
| 704 | `model.audio_tower.layers.9.feed_forward2.post_layer_norm.weight` | 1024 | bfloat16 | 199.3 | 3.714 | 5.001 | -11.69 | 49.25 | 12.12, 2.328, -1.188, 2.719, 8.375, 1.258, 3.281, 4.562 |
| 705 | `model.audio_tower.layers.9.feed_forward2.pre_layer_norm.weight` | 1024 | bfloat16 | 378.1 | 10.41 | 5.595 | -23.38 | 32.75 | 9.375, 10.81, 6.719, 11.5, 8.25, 4.656, 15.88, 16.5 |
| 706 | `model.audio_tower.layers.9.lconv1d.conv_norm.weight` | 1024 | bfloat16 | 120.9 | 3.28 | 1.873 | -8.312 | 10.19 | 4.812, 3.391, 3.203, 3.812, 3.641, 4.594, 4.281, 5.5 |
| 707 | `model.audio_tower.layers.9.lconv1d.depthwise_conv1d.weight` | 1024×1×5 | bfloat16 | 32 | 0.002553 | 0.4472 | -4.812 | 2.109 | 0.04102, -0.01239, -0.01483, -0.2988, -0.6211, 0.04224, 0.01721, -0.1133 |
| 708 | `model.audio_tower.layers.9.lconv1d.linear_end.input_max` |  | bfloat16 | 8 | 8 | 0 | 8 | 8 | 8 |
| 709 | `model.audio_tower.layers.9.lconv1d.linear_end.input_min` |  | bfloat16 | 8.062 | -8.062 | 0 | -8.062 | -8.062 | -8.062 |
| 710 | `model.audio_tower.layers.9.lconv1d.linear_end.linear.weight` | 1024×1024 | bfloat16 | 29.98 | 0.0005335 | 0.02927 | -0.1543 | 0.07715 | 0, -0.03442, -0.03442, 0, 0.03442, -0.03442, 0.03442, 0 |
| 711 | `model.audio_tower.layers.9.lconv1d.linear_end.output_max` |  | bfloat16 | 8 | 8 | 0 | 8 | 8 | 8 |
| 712 | `model.audio_tower.layers.9.lconv1d.linear_end.output_min` |  | bfloat16 | 8.062 | -8.062 | 0 | -8.062 | -8.062 | -8.062 |
| 713 | `model.audio_tower.layers.9.lconv1d.linear_start.input_max` |  | bfloat16 | 10.44 | 10.44 | 0 | 10.44 | 10.44 | 10.44 |
| 714 | `model.audio_tower.layers.9.lconv1d.linear_start.input_min` |  | bfloat16 | 10.5 | -10.5 | 0 | -10.5 | -10.5 | -10.5 |
| 715 | `model.audio_tower.layers.9.lconv1d.linear_start.linear.weight` | 2048×1024 | bfloat16 | 44.95 | -0.001124 | 0.03101 | -0.1592 | 0.1387 | 0.02002, 0, 0, -0.07031, 0.03003, 0, 0.03003, -0.03003 |
| 716 | `model.audio_tower.layers.9.lconv1d.linear_start.output_max` |  | bfloat16 | 24.12 | 24.12 | 0 | 24.12 | 24.12 | 24.12 |
| 717 | `model.audio_tower.layers.9.lconv1d.linear_start.output_min` |  | bfloat16 | 24.38 | -24.38 | 0 | -24.38 | -24.38 | -24.38 |
| 718 | `model.audio_tower.layers.9.lconv1d.pre_layer_norm.weight` | 1024 | bfloat16 | 183.6 | 4.213 | 3.896 | -15.94 | 37.25 | 1.398, 3.672, 4.938, 2.844, 2.266, 4.719, 7.406, 7.344 |
| 719 | `model.audio_tower.layers.9.norm_out.weight` | 1024 | bfloat16 | 212.6 | 5.557 | 3.645 | -8.875 | 42 | 5.625, 6.094, 9.688, 5.906, 6.531, 8.688, 6.344, 4.938 |
| 720 | `model.audio_tower.layers.9.norm_post_attn.weight` | 1024 | bfloat16 | 48.18 | 0.6652 | 1.351 | -4.25 | 18.25 | -0.1729, -0.5977, 0.01648, 1.453, 1.055, -0.06079, 1.453, 0.5664 |
| 721 | `model.audio_tower.layers.9.norm_pre_attn.weight` | 1024 | bfloat16 | 169.3 | 3.11 | 4.282 | -20.38 | 29.25 | 1.961, 1.906, -0.166, 2.656, 1.297, 2.453, 9.625, -8.438 |
| 722 | `model.audio_tower.layers.9.self_attn.k_proj.input_max` |  | bfloat16 | 9.062 | 9.062 | 0 | 9.062 | 9.062 | 9.062 |
| 723 | `model.audio_tower.layers.9.self_attn.k_proj.input_min` |  | bfloat16 | 9.125 | -9.125 | 0 | -9.125 | -9.125 | -9.125 |
| 724 | `model.audio_tower.layers.9.self_attn.k_proj.linear.weight` | 1024×1024 | bfloat16 | 29.54 | -0.001567 | 0.02881 | -0.1289 | 0.06445 | 0.03052, 0, 0.03052, 0, 0.03052, 0.03052, 0.03052, 0 |
| 725 | `model.audio_tower.layers.9.self_attn.k_proj.output_max` |  | bfloat16 | 16.25 | 16.25 | 0 | 16.25 | 16.25 | 16.25 |
| 726 | `model.audio_tower.layers.9.self_attn.k_proj.output_min` |  | bfloat16 | 16.38 | -16.38 | 0 | -16.38 | -16.38 | -16.38 |
| 727 | `model.audio_tower.layers.9.self_attn.per_dim_scale` | 128 | bfloat16 | 23.7 | -2.048 | 0.4411 | -3.203 | -1.141 | -2.422, -2.031, -1.148, -2.547, -2.438, -1.945, -1.391, -1.758 |
| 728 | `model.audio_tower.layers.9.self_attn.post.input_max` |  | bfloat16 | 15.69 | 15.69 | 0 | 15.69 | 15.69 | 15.69 |
| 729 | `model.audio_tower.layers.9.self_attn.post.input_min` |  | bfloat16 | 15.81 | -15.81 | 0 | -15.81 | -15.81 | -15.81 |
| 730 | `model.audio_tower.layers.9.self_attn.post.linear.weight` | 1024×1024 | bfloat16 | 29.76 | -0.001137 | 0.02904 | -0.1826 | 0.09131 | -0.04688, 0, 0.04688, 0, -0.04688, 0, -0.04688, 0.04688 |
| 731 | `model.audio_tower.layers.9.self_attn.post.output_max` |  | bfloat16 | 54.5 | 54.5 | 0 | 54.5 | 54.5 | 54.5 |
| 732 | `model.audio_tower.layers.9.self_attn.post.output_min` |  | bfloat16 | 55 | -55 | 0 | -55 | -55 | -55 |
| 733 | `model.audio_tower.layers.9.self_attn.q_proj.input_max` |  | bfloat16 | 9.062 | 9.062 | 0 | 9.062 | 9.062 | 9.062 |
| 734 | `model.audio_tower.layers.9.self_attn.q_proj.input_min` |  | bfloat16 | 9.125 | -9.125 | 0 | -9.125 | -9.125 | -9.125 |
| 735 | `model.audio_tower.layers.9.self_attn.q_proj.linear.weight` | 1024×1024 | bfloat16 | 29.48 | -0.002116 | 0.02871 | -0.2178 | 0.1089 | 0.03149, 0, 0, 0, 0, -0.06299, 0, 0 |
| 736 | `model.audio_tower.layers.9.self_attn.q_proj.output_max` |  | bfloat16 | 16.25 | 16.25 | 0 | 16.25 | 16.25 | 16.25 |
| 737 | `model.audio_tower.layers.9.self_attn.q_proj.output_min` |  | bfloat16 | 16.38 | -16.38 | 0 | -16.38 | -16.38 | -16.38 |
| 738 | `model.audio_tower.layers.9.self_attn.relative_k_proj.weight` | 1024×1024 | bfloat16 | 32 | -0.004653 | 0.0309 | -0.1611 | 0.1836 | 0.03271, 0.02966, 0.02844, 0.0282, 0.02869, 0.0304, 0.03247, 0.03442 |
| 739 | `model.audio_tower.layers.9.self_attn.v_proj.input_max` |  | bfloat16 | 9.062 | 9.062 | 0 | 9.062 | 9.062 | 9.062 |
| 740 | `model.audio_tower.layers.9.self_attn.v_proj.input_min` |  | bfloat16 | 9.125 | -9.125 | 0 | -9.125 | -9.125 | -9.125 |
| 741 | `model.audio_tower.layers.9.self_attn.v_proj.linear.weight` | 1024×1024 | bfloat16 | 29.91 | -0.001327 | 0.02918 | -0.08496 | 0.04248 | 0, 0.03296, 0, -0.06592, -0.03296, 0.03296, 0, 0.03296 |
| 742 | `model.audio_tower.layers.9.self_attn.v_proj.output_max` |  | bfloat16 | 16.25 | 16.25 | 0 | 16.25 | 16.25 | 16.25 |
| 743 | `model.audio_tower.layers.9.self_attn.v_proj.output_min` |  | bfloat16 | 16.38 | -16.38 | 0 | -16.38 | -16.38 | -16.38 |
| 744 | `model.audio_tower.output_proj.bias` | 1536 | bfloat16 | 54.98 | 0.03632 | 1.403 | -13.19 | 14.88 | 0.1387, -0.2812, -0.03149, 8.25, -0.0152, -0.4082, 0.06885, -0.03833 |
| 745 | `model.audio_tower.output_proj.weight` | 1536×1024 | bfloat16 | 39.19 | 3.6e-05 | 0.03125 | -0.2539 | 0.2266 | -0.009888, 0.001205, -0.01532, -0.1416, -0.04004, -0.00106, 0.07861, 0.02539 |
| 746 | `model.audio_tower.subsample_conv_projection.input_proj_linear.weight` | 1024×1024 | bfloat16 | 32 | 0.0006422 | 0.03124 | -0.3711 | 0.3066 | -0.01819, -0.01477, 0.03345, 0.02747, 3.099e-05, -0.03198, 0.008728, -0.01349 |
| 747 | `model.audio_tower.subsample_conv_projection.layer0.conv.weight` | 128×1×3×3 | bfloat16 | 11.31 | -0.07282 | 0.3254 | -1.328 | 1.195 | -0.4062, 0.4824, -0.2871, 0.08008, -0.6875, 0.3984, -0.052, -0.09814 |
| 748 | `model.audio_tower.subsample_conv_projection.layer0.norm.weight` | 128 | bfloat16 | 36.85 | 2.721 | 1.797 | -2.656 | 7.875 | 5.062, 4.375, 3.797, 3.328, 1.523, 1.07, 6.781, 1.656 |
| 749 | `model.audio_tower.subsample_conv_projection.layer1.conv.weight` | 32×128×3×3 | bfloat16 | 5.657 | 0.0005379 | 0.02946 | -0.2637 | 0.2852 | -0.001663, -0.008484, -0.00264, -0.005493, 0.01575, 0.0177, 0.005005, -0.007355 |
| 750 | `model.audio_tower.subsample_conv_projection.layer1.norm.weight` | 32 | bfloat16 | 49.03 | 8.219 | 2.796 | -2.328 | 12.44 | 9.938, 9.188, 9.125, 5.844, -2.328, 6.344, 7.062, 11 |
| 751 | `model.embed_audio.embedding_projection.weight` | 1536×1536 | bfloat16 | 53.5 | -1.558e-06 | 0.03483 | -0.2354 | 0.2754 | 0.01263, 0.02148, -0.03271, 0.0144, -0.007141, -0.02222, -0.006104, -0.03833 |
| 752 | `model.embed_vision.embedding_projection.weight` | 1536×768 | bfloat16 | 50.27 | -3.585e-05 | 0.04628 | -0.6641 | 0.5938 | 0.03101, 0.02625, -0.04956, -0.01624, 0.02124, -0.00322, 0.05933, 0.008545 |
| 753 | `model.language_model.embed_tokens.weight` | 262144×1536 | bfloat16 | 526.1 | 5.859e-05 | 0.03052 | -0.4824 | 0.5586 | -0.0002975, -0.0006065, 0.05762, -0.04492, 0.01855, 0.005768, 0.007782, 0.02966 |
| 754 | `model.language_model.embed_tokens_per_layer.weight` | 262144×8960 | bfloat16 | 1659 | 9.375e-05 | 0.06402 | -1.125 | 1.055 | 0.003662, 0.01367, 0.0003128, 0.0152, 0.03247, -0.002274, 0.008667, -0.02441 |
| 755 | `model.language_model.layers.0.input_layernorm.weight` | 1536 | bfloat16 | 463 | 10.67 | 5.082 | 6.625 | 83 | 9.375, 7.969, 10.69, 12, 8.875, 8.5, 10, 9.812 |
| 756 | `model.language_model.layers.0.layer_scalar` | 1 | bfloat16 | 0.01782 | 0.01782 | 0 | 0.01782 | 0.01782 | 0.01782 |
| 757 | `model.language_model.layers.0.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 88.71 | 5.772e-06 | 0.0289 | -0.498 | 0.5508 | 0.003113, -0.02515, 0.01917, -0.0199, 0.03003, -0.01471, 0.00708, -0.005676 |
| 758 | `model.language_model.layers.0.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 106.3 | -4.257e-05 | 0.03465 | -0.4297 | 0.543 | 0.06152, -0.01147, -0.009949, -0.06201, 0.01648, -0.02405, 0.01672, 0.06201 |
| 759 | `model.language_model.layers.0.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 109.4 | -1.755e-05 | 0.03564 | -0.4492 | 0.4746 | -0.05322, -0.02112, 0.003433, -0.02051, -1.991e-05, -0.01355, -0.01233, 0.06494 |
| 760 | `model.language_model.layers.0.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 0.3499 | 6.02e-07 | 0.000558 | -0.00589 | 0.00589 | 0.001045, 0.0004787, -2.73e-05, -1.931e-05, 0.0004578, 0.0005722, -0.0006523, -0.0007172 |
| 761 | `model.language_model.layers.0.per_layer_projection.weight` | 1536×256 | bfloat16 | 44.54 | -0.0001153 | 0.07102 | -0.7852 | 0.6875 | 0.1001, 0.03564, 0.003815, -0.01068, -0.04883, -0.04297, -0.1206, -0.1211 |
| 762 | `model.language_model.layers.0.post_attention_layernorm.weight` | 1536 | bfloat16 | 705.7 | 5.953 | 17 | 0.08008 | 101 | 0.5391, 16.5, 0.8203, 0.8398, 0.8516, 0.9492, 0.5898, 0.6328 |
| 763 | `model.language_model.layers.0.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 605.9 | 10.5 | 11.36 | 0.03345 | 47.75 | 7.125, 47.75, 7.906, 8.625, 4.188, 6.719, 3.25, 7.375 |
| 764 | `model.language_model.layers.0.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 164 | 3.578 | 2.172 | 0.00412 | 5.438 | 5.438, 5.438, 5.438, 5.438, 1.656, 2.516, 0.9766, 2.953 |
| 765 | `model.language_model.layers.0.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 827 | 19.12 | 8.921 | -0.8086 | 116.5 | 21.62, 4.781, 22.88, 22.75, 19, 18.88, 22.12, 24 |
| 766 | `model.language_model.layers.0.self_attn.k_norm.weight` | 256 | bfloat16 | 2.031 | 0.127 | 0 | 0.127 | 0.127 | 0.127, 0.127, 0.127, 0.127, 0.127, 0.127, 0.127, 0.127 |
| 767 | `model.language_model.layers.0.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 23.81 | 1.357e-05 | 0.03797 | -0.2754 | 0.2041 | 0.01276, -0.07666, -0.03223, 0.003159, -0.03857, -0.001137, -0.009338, -0.0116 |
| 768 | `model.language_model.layers.0.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 57.46 | 1.618e-05 | 0.0324 | -0.4199 | 0.4121 | -0.01404, -0.002228, 0.01282, 0.02905, 0.0004101, -0.04346, -0.1123, -0.02844 |
| 769 | `model.language_model.layers.0.self_attn.q_norm.weight` | 256 | bfloat16 | 15.75 | 0.9844 | 0 | 0.9844 | 0.9844 | 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844 |
| 770 | `model.language_model.layers.0.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 65.19 | 2.202e-05 | 0.03676 | -0.4434 | 0.4629 | -0.007812, 0.02051, -0.001259, 0.006775, 0.001114, -0.01526, 0.01349, 0.0006638 |
| 771 | `model.language_model.layers.0.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 22.56 | -5.101e-05 | 0.03597 | -0.3828 | 0.3457 | -0.003403, 0.03418, -0.02856, -0.01514, 0.007385, 0.0188, 0.02783, 0.03271 |
| 772 | `model.language_model.layers.1.input_layernorm.weight` | 1536 | bfloat16 | 774.3 | 16.31 | 11.15 | -3.484 | 76.5 | 10.81, 1, 20.5, 11.31, 13.31, 22.12, 20.88, 9.062 |
| 773 | `model.language_model.layers.1.layer_scalar` | 1 | bfloat16 | 0.2227 | 0.2227 | 0 | 0.2227 | 0.2227 | 0.2227 |
| 774 | `model.language_model.layers.1.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 89.07 | 1.5e-05 | 0.02901 | -0.5977 | 0.5781 | 0.002716, -0.01019, -0.03247, -0.02686, 0.01416, 0.003616, -0.003876, 0.02759 |
| 775 | `model.language_model.layers.1.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 109.7 | -1.874e-05 | 0.03573 | -0.3809 | 0.3652 | 0.04248, -0.03662, 0.04004, 0.01575, -0.02698, 0.01685, -0.07764, 0.02661 |
| 776 | `model.language_model.layers.1.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 111.9 | 1.537e-05 | 0.03647 | -0.4062 | 0.4004 | 0.01733, 0.01941, -0.03809, 0.05591, -0.03931, -0.008057, -0.05054, 0.02551 |
| 777 | `model.language_model.layers.1.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 4.576 | 1.327e-05 | 0.007298 | -0.08105 | 0.07812 | 0.01178, 0.01068, 0.004059, 0.004669, -0.005646, 0.001686, -0.0108, -0.00148 |
| 778 | `model.language_model.layers.1.per_layer_projection.weight` | 1536×256 | bfloat16 | 43.53 | -4.257e-05 | 0.06942 | -0.5078 | 0.4395 | -0.03589, 0.08203, -0.1611, 0.03833, -0.007721, 0.02686, 0.04102, 0.1602 |
| 779 | `model.language_model.layers.1.post_attention_layernorm.weight` | 1536 | bfloat16 | 39.98 | 0.3745 | 0.9492 | 0.005371 | 8.5 | 0.05005, 4.281, 0.5195, 0.05762, 0.006866, 0.3789, 0.007538, 0.0376 |
| 780 | `model.language_model.layers.1.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 37.99 | 0.6674 | 0.7031 | 0.002014 | 2.266 | 0.1963, 2.266, 1.32, 0.1553, 0.06641, 0.6211, 0.04688, 0.1133 |
| 781 | `model.language_model.layers.1.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 4.611 | 0.09005 | 0.07573 | 0.0001068 | 0.1611 | 0.1611, 0.1611, 0.1611, 0.1611, 0.03687, 0.1611, 0.0001488, 0.000144 |
| 782 | `model.language_model.layers.1.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 757.7 | 14.85 | 12.39 | -1.312 | 101 | 13.56, 0.7422, 5.625, 10.94, 20.75, 7.25, 26.25, 12.12 |
| 783 | `model.language_model.layers.1.self_attn.k_norm.weight` | 256 | bfloat16 | 1.953 | 0.1221 | 0 | 0.1221 | 0.1221 | 0.1221, 0.1221, 0.1221, 0.1221, 0.1221, 0.1221, 0.1221, 0.1221 |
| 784 | `model.language_model.layers.1.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 22.71 | 0.0001178 | 0.03622 | -0.2715 | 0.2734 | -0.006683, 0.008484, 0.005463, -0.01495, -0.004517, 0.001755, -0.02673, -0.02295 |
| 785 | `model.language_model.layers.1.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 56.76 | -4.447e-06 | 0.032 | -0.3379 | 0.3066 | 0.01929, 0.004547, 0.0354, -0.06738, -0.01495, 0.02014, 0.06299, 0.01398 |
| 786 | `model.language_model.layers.1.self_attn.q_norm.weight` | 256 | bfloat16 | 16.38 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 787 | `model.language_model.layers.1.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 63.97 | -3.359e-05 | 0.03607 | -0.5469 | 0.4336 | 0.001175, 0.02124, -0.002945, -0.01215, 0.03113, 0.0008545, -0.03101, -0.01843 |
| 788 | `model.language_model.layers.1.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 23.45 | 5.48e-05 | 0.0374 | -0.4004 | 0.3594 | 0.06543, 0.02271, -0.02502, 0.04663, 0.01227, 0.01117, -0.0004921, -0.05591 |
| 789 | `model.language_model.layers.10.input_layernorm.weight` | 1536 | bfloat16 | 1929 | 11.45 | 47.89 | -64 | 588 | 5.656, 3.609, 2.375, 14.69, 3.125, 5.625, 3.094, 3.328 |
| 790 | `model.language_model.layers.10.layer_scalar` | 1 | bfloat16 | 0.4434 | 0.4434 | 0 | 0.4434 | 0.4434 | 0.4434 |
| 791 | `model.language_model.layers.10.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 70.41 | -4.788e-06 | 0.02294 | -0.4824 | 0.416 | 0.08545, 0.002426, 0.01385, -0.0155, -0.009583, -0.03296, -0.04297, -0.02563 |
| 792 | `model.language_model.layers.10.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 105.9 | -5.68e-05 | 0.03451 | -0.6445 | 0.5234 | 0.08789, 0.0625, -0.02783, -0.02991, 0.05127, -0.03088, -0.006927, 0.03711 |
| 793 | `model.language_model.layers.10.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 105.2 | 6.523e-06 | 0.03427 | -0.5078 | 0.5078 | 0.1157, 0.04688, -0.02759, 0.04199, 0.06934, -0.01672, -0.01324, -0.001289 |
| 794 | `model.language_model.layers.10.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 10.46 | -7.576e-05 | 0.01668 | -0.2852 | 0.2539 | -0.0008812, -0.003372, 0.01111, -0.0166, -0.01483, -0.04028, -0.0007057, 0.004761 |
| 795 | `model.language_model.layers.10.per_layer_projection.weight` | 1536×256 | bfloat16 | 46.26 | 9.235e-05 | 0.07377 | -0.6133 | 0.6172 | -0.002289, -0.01459, -0.02649, -0.01434, -0.03076, -0.02332, -0.021, -0.01733 |
| 796 | `model.language_model.layers.10.post_attention_layernorm.weight` | 1536 | bfloat16 | 39.7 | 0.9104 | 0.444 | 0.0009842 | 1.227 | 0.2002, 1.227, 1.023, 0.1104, 1.227, 0.2539, 1.227, 1.227 |
| 797 | `model.language_model.layers.10.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 113.5 | 2.558 | 1.362 | 0.002838 | 3.609 | 0.3789, 3.609, 3.609, 0.3066, 3.609, 0.5938, 3.609, 3.609 |
| 798 | `model.language_model.layers.10.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 37.61 | 0.7506 | 0.5982 | 0.0009842 | 1.336 | 0.1445, 1.336, 1.336, 0.2314, 1.336, 0.3184, 1.336, 0.001213 |
| 799 | `model.language_model.layers.10.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 601.3 | 2.297 | 15.17 | 0.001907 | 180 | 0.5469, 0.3457, 0.3242, 2.25, 0.3281, 0.6523, 0.3125, 0.3105 |
| 800 | `model.language_model.layers.10.self_attn.k_norm.weight` | 256 | bfloat16 | 1.938 | 0.1211 | 0 | 0.1211 | 0.1211 | 0.1211, 0.1211, 0.1211, 0.1211, 0.1211, 0.1211, 0.1211, 0.1211 |
| 801 | `model.language_model.layers.10.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 20.89 | 3.725e-05 | 0.03331 | -0.3633 | 0.332 | -0.0009499, 0.01459, -0.01471, 0.0119, 0.00592, -0.01904, -0.007385, 0.0004692 |
| 802 | `model.language_model.layers.10.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 57.4 | 7.706e-06 | 0.03237 | -0.3965 | 0.3906 | 0.05005, -0.03149, 0.005524, 0.01123, 0.006104, 0.00705, 0.00325, -0.004852 |
| 803 | `model.language_model.layers.10.self_attn.q_norm.weight` | 256 | bfloat16 | 16.5 | 1.031 | 0 | 1.031 | 1.031 | 1.031, 1.031, 1.031, 1.031, 1.031, 1.031, 1.031, 1.031 |
| 804 | `model.language_model.layers.10.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 61.65 | 1.043e-05 | 0.03476 | -0.4434 | 0.5234 | 0.00103, 0.02014, 0.002426, -0.04395, -0.02356, 0.01855, 0.04492, -0.03271 |
| 805 | `model.language_model.layers.10.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 20.57 | 6.769e-06 | 0.0328 | -0.3516 | 0.3574 | -0.01056, 0.0009193, -0.03638, 0.02112, 0.00769, -0.04321, -0.001808, 0.02698 |
| 806 | `model.language_model.layers.11.input_layernorm.weight` | 1536 | bfloat16 | 2839 | 12.99 | 71.28 | -2.078 | 900 | 7.812, 2.078, 1.969, 15.62, 2.344, 7, 2.422, 2.531 |
| 807 | `model.language_model.layers.11.layer_scalar` | 1 | bfloat16 | 0.3691 | 0.3691 | 0 | 0.3691 | 0.3691 | 0.3691 |
| 808 | `model.language_model.layers.11.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 68.17 | 3.592e-06 | 0.02221 | -0.4355 | 0.4629 | -0.01636, 0.008057, 0.04468, 0.007385, -0.01733, 0.002457, 0.002716, 0.005493 |
| 809 | `model.language_model.layers.11.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 95.58 | -1.621e-05 | 0.03114 | -0.8125 | 0.6836 | 0.01917, 0.001526, 0.001755, 0.005951, -0.005981, -0.0001659, -0.01556, 0.01385 |
| 810 | `model.language_model.layers.11.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 96.39 | -5.616e-06 | 0.0314 | -0.668 | 0.6133 | -0.01831, 0.002136, 0.03638, -0.08545, 0.03198, 0.0116, 0.03589, 0.009399 |
| 811 | `model.language_model.layers.11.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 9.581 | 8.485e-05 | 0.01528 | -0.2598 | 0.2041 | -0.008911, 0.009766, 0.002014, -0.005707, -0.01239, -0.02893, 0.006104, 0.003357 |
| 812 | `model.language_model.layers.11.per_layer_projection.weight` | 1536×256 | bfloat16 | 47.89 | -0.0001037 | 0.07637 | -0.6719 | 0.6328 | -0.03467, 0.01331, -0.06348, 0.01447, 0.002106, -0.05444, -0.00235, -0.07715 |
| 813 | `model.language_model.layers.11.post_attention_layernorm.weight` | 1536 | bfloat16 | 47.67 | 1.041 | 0.6299 | 0.001244 | 1.516 | 0.05762, 1.516, 1.023, 0.06445, 1.516, 0.09521, 1.516, 1.516 |
| 814 | `model.language_model.layers.11.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 141.8 | 3.055 | 1.937 | 0.003433 | 4.594 | 0.1533, 4.594, 3.594, 0.1572, 4.594, 0.248, 4.594, 4.594 |
| 815 | `model.language_model.layers.11.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 38.02 | 0.7394 | 0.6281 | 0.001015 | 1.32 | 1.32, 1.32, 0.001137, 1.32, 1.32, 1.32, 1.32, 1.32 |
| 816 | `model.language_model.layers.11.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 1631 | 5.214 | 41.31 | -0.6133 | 474 | 0.875, 0.3066, 0.377, 4.688, 0.291, 1.125, 0.2793, 0.249 |
| 817 | `model.language_model.layers.11.self_attn.k_norm.weight` | 256 | bfloat16 | 2 | 0.125 | 0 | 0.125 | 0.125 | 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125 |
| 818 | `model.language_model.layers.11.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 19.24 | -3.16e-06 | 0.03068 | -0.3281 | 0.3125 | -0.005951, -0.01721, -0.02808, -0.01746, 0.01324, 0.0008087, -0.001099, 0.02478 |
| 819 | `model.language_model.layers.11.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 54.94 | 1.48e-05 | 0.03098 | -0.543 | 0.4961 | -0.009338, -0.003159, 0.01624, -0.05396, -0.005219, -0.02673, 0.007477, 0.003281 |
| 820 | `model.language_model.layers.11.self_attn.q_norm.weight` | 256 | bfloat16 | 16 | 1 | 0 | 1 | 1 | 1, 1, 1, 1, 1, 1, 1, 1 |
| 821 | `model.language_model.layers.11.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 56.68 | 7.169e-06 | 0.03196 | -0.3965 | 0.375 | 0.001793, 0.009888, 0.00415, -0.01398, 0.00592, -0.01965, 0.01892, 0.002823 |
| 822 | `model.language_model.layers.11.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 19.2 | -8.442e-06 | 0.03062 | -0.2812 | 0.2852 | -0.03809, -0.003479, -0.04639, -0.01965, -0.04639, -0.02893, 0.0144, 0.006714 |
| 823 | `model.language_model.layers.12.input_layernorm.weight` | 1536 | bfloat16 | 1569 | 9.498 | 38.89 | -4.375 | 436 | 5.25, 1.188, 1.25, 11.12, 1.055, 8.188, 1.133, 1.109 |
| 824 | `model.language_model.layers.12.layer_scalar` | 1 | bfloat16 | 0.3242 | 0.3242 | 0 | 0.3242 | 0.3242 | 0.3242 |
| 825 | `model.language_model.layers.12.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 65.9 | 3.63e-06 | 0.02147 | -0.3711 | 0.4297 | -0.00037, -0.02039, -0.09082, -0.002625, -0.01245, -0.01312, 0.02148, -0.01416 |
| 826 | `model.language_model.layers.12.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 98.4 | -5.718e-05 | 0.03206 | -0.875 | 0.7695 | 0.02856, 0.05518, -0.1099, -0.0166, 0.06885, 0.004944, 0.04272, -0.04028 |
| 827 | `model.language_model.layers.12.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 100.6 | -1.892e-06 | 0.03278 | -0.5078 | 0.5 | 0.01123, 0.002182, -0.01166, 0.008179, -0.03516, -0.01489, -0.01636, 0.0105 |
| 828 | `model.language_model.layers.12.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 9.015 | -0.0001123 | 0.01438 | -0.1396 | 0.1895 | 0.01257, -0.01215, 0.007629, 0.01447, -0.01086, -0.007812, 0.01166, -0.01025 |
| 829 | `model.language_model.layers.12.per_layer_projection.weight` | 1536×256 | bfloat16 | 43.38 | -6.518e-06 | 0.06917 | -0.7617 | 0.7773 | 0.05396, -0.09424, -0.0752, 0.07422, 0.0957, -0.02502, 0.1592, 0.04492 |
| 830 | `model.language_model.layers.12.post_attention_layernorm.weight` | 1536 | bfloat16 | 56.19 | 1.198 | 0.7878 | 0.001289 | 1.852 | 0.04663, 1.586, 1.055, 0.02112, 1.852, 0.08057, 1.523, 1.852 |
| 831 | `model.language_model.layers.12.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 151.4 | 3.281 | 2.042 | 0.003891 | 4.875 | 0.1328, 4.875, 4.875, 0.06348, 4.875, 0.1992, 3.922, 4.875 |
| 832 | `model.language_model.layers.12.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 48.19 | 0.5903 | 1.079 | 0.001587 | 2.656 | 0.002304, 0.002396, 0.002319, 0.002441, 2.656, 0.1299, 0.002457, 0.002289 |
| 833 | `model.language_model.layers.12.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 488.8 | 2.917 | 12.13 | 0.1387 | 137 | 1.828, 0.2041, 0.2852, 4.531, 0.1895, 2.141, 0.2109, 0.1836 |
| 834 | `model.language_model.layers.12.self_attn.k_norm.weight` | 256 | bfloat16 | 1.961 | 0.1226 | 0 | 0.1226 | 0.1226 | 0.1226, 0.1226, 0.1226, 0.1226, 0.1226, 0.1226, 0.1226, 0.1226 |
| 835 | `model.language_model.layers.12.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 19.45 | 3.329e-05 | 0.03101 | -0.3379 | 0.3516 | -0.00824, -0.01904, -0.006653, 0.003677, 0.01807, 0.02869, 0.007751, 0.002121 |
| 836 | `model.language_model.layers.12.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 51.56 | -1.969e-05 | 0.02907 | -0.3223 | 0.3418 | -0.04907, -0.01782, -0.003326, -0.01349, 0.03247, 0.0006714, -0.01953, 0.009521 |
| 837 | `model.language_model.layers.12.self_attn.q_norm.weight` | 256 | bfloat16 | 16.38 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 838 | `model.language_model.layers.12.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 57.62 | -1.169e-05 | 0.03249 | -0.4453 | 0.4238 | 0.0144, 0.004883, 0.01306, -0.01147, -0.01483, -0.005524, 0.008179, 0.01685 |
| 839 | `model.language_model.layers.12.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 18.65 | -2.736e-05 | 0.02973 | -0.4082 | 0.4102 | 0.000433, 0.0874, -0.03345, 0.01459, 0.01831, 0.02246, -0.02661, 0.02832 |
| 840 | `model.language_model.layers.13.input_layernorm.weight` | 1536 | bfloat16 | 2078 | 10.94 | 51.91 | -49 | 608 | 17.62, 0.4121, 0.4434, 21.62, 0.3828, 17.75, 0.6602, 0.4316 |
| 841 | `model.language_model.layers.13.layer_scalar` | 1 | bfloat16 | 0.08838 | 0.08838 | 0 | 0.08838 | 0.08838 | 0.08838 |
| 842 | `model.language_model.layers.13.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 72.88 | 9.312e-06 | 0.02375 | -0.6133 | 0.6211 | -0.0006256, -0.04785, 0.01965, -0.01648, -0.007812, -0.01422, 0.02844, -0.02026 |
| 843 | `model.language_model.layers.13.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 96.12 | 1.278e-05 | 0.03132 | -0.7109 | 0.6133 | -0.0008354, -0.0177, 0.003876, 0.01733, 0.001076, 0.02393, 0.01758, 0.01056 |
| 844 | `model.language_model.layers.13.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 95.86 | -1.229e-05 | 0.03123 | -0.4512 | 0.5117 | 0.05786, -0.02039, -0.02014, 0.02942, -0.02576, -0.04565, -0.01587, 0.02966 |
| 845 | `model.language_model.layers.13.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 3.1 | -1.423e-05 | 0.004944 | -0.05591 | 0.04932 | -0.01184, 0.01312, -0.0008011, -0.006927, 0.005005, -0.001984, 1.103e-06, -0.007599 |
| 846 | `model.language_model.layers.13.per_layer_projection.weight` | 1536×256 | bfloat16 | 44.42 | 4.897e-05 | 0.07084 | -0.8555 | 0.7656 | 0.00769, 0.01965, -0.002899, -0.001007, -0.1147, -0.01819, 0.005493, -0.007812 |
| 847 | `model.language_model.layers.13.post_attention_layernorm.weight` | 1536 | bfloat16 | 44.68 | 1.005 | 0.538 | 0.001076 | 1.359 | 0.4824, 1.359, 1.102, 0.0177, 1.359, 0.1621, 1.359, 1.359 |
| 848 | `model.language_model.layers.13.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 496.7 | 12.2 | 3.43 | 0.01105 | 13.44 | 13.44, 13.44, 13.44, 13.38, 13.44, 13.44, 13.44, 13.44 |
| 849 | `model.language_model.layers.13.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 302.6 | 3.317 | 6.975 | 0.0144 | 18.12 | 0.01746, 0.01685, 0.01733, 0.01648, 0.01746, 18.12, 0.01587, 0.01697 |
| 850 | `model.language_model.layers.13.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 1243 | 5.051 | 31.32 | -0.4258 | 366 | 0.9727, 0.2041, 0.2539, 12.69, 0.1992, 2.578, 0.21, 0.1875 |
| 851 | `model.language_model.layers.13.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 852 | `model.language_model.layers.13.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 18.78 | -2.223e-05 | 0.02995 | -0.2256 | 0.2227 | -0.007568, -0.05908, -0.02832, -0.00589, 0.00766, -0.01001, 0.003098, -0.01544 |
| 853 | `model.language_model.layers.13.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 53.9 | 3.938e-06 | 0.03039 | -0.5156 | 0.4922 | 0.03247, -0.04492, 0.04053, 0.04858, -0.02197, -0.01025, 0.02612, 0.00383 |
| 854 | `model.language_model.layers.13.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 855 | `model.language_model.layers.13.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 59.07 | 5.64e-06 | 0.03331 | -0.4316 | 0.4414 | -0.001175, -0.01013, 0.04492, 0.05737, -0.01526, -0.01831, -0.05054, -0.07178 |
| 856 | `model.language_model.layers.13.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 17.27 | 4.838e-05 | 0.02755 | -0.3008 | 0.2969 | 0.01978, 0.02979, -0.03564, -0.03003, 0.04102, -2.57e-07, -0.01721, -0.01733 |
| 857 | `model.language_model.layers.14.input_layernorm.weight` | 1536 | bfloat16 | 629.1 | 9.138 | 13.2 | -7 | 122.5 | 6.188, 5.312, 5.094, 13.44, 5.375, 5.156, 5.812, 5.781 |
| 858 | `model.language_model.layers.14.layer_scalar` | 1 | bfloat16 | 0.02856 | 0.02856 | 0 | 0.02856 | 0.02856 | 0.02856 |
| 859 | `model.language_model.layers.14.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 74.68 | -5.391e-06 | 0.02433 | -0.5938 | 0.5117 | -0.03955, 0.001114, -0.0293, 0.009399, -0.01099, 0.0105, 0.01355, -0.01624 |
| 860 | `model.language_model.layers.14.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 96.03 | 0.0001871 | 0.03129 | -0.5312 | 0.4766 | 0.02209, 0.0152, 0.01093, -0.007141, 0.003357, 0.002563, 0.0188, -0.005096 |
| 861 | `model.language_model.layers.14.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 100.5 | -8.096e-06 | 0.03276 | -0.3672 | 0.3398 | 0.03662, 0.02039, -0.05176, -0.02063, 0.01965, -0.04126, 0.002472, -0.02734 |
| 862 | `model.language_model.layers.14.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 0.6203 | -2.236e-06 | 0.0009892 | -0.01056 | 0.01331 | -0.0002918, 0.001526, 0.000824, 0.002014, -0.001335, 0.001099, 0.0007133, 0.0003204 |
| 863 | `model.language_model.layers.14.per_layer_projection.weight` | 1536×256 | bfloat16 | 44.26 | -0.0001232 | 0.07058 | -0.7344 | 0.7852 | 0.01007, 0.02979, -0.05249, 0.01807, 0.02124, 0.00946, 0.05933, -0.05273 |
| 864 | `model.language_model.layers.14.post_attention_layernorm.weight` | 1536 | bfloat16 | 642.7 | 14.88 | 6.904 | 0.0177 | 18.62 | 18.62, 18.62, 18.62, 8.688, 18.62, 18.62, 18.62, 18.62 |
| 865 | `model.language_model.layers.14.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 1790 | 42.74 | 16.15 | 0.3438 | 61 | 44.75, 61, 61, 21.62, 41.25, 38.75, 42.25, 38 |
| 866 | `model.language_model.layers.14.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 566.1 | 10.33 | 10.1 | 0.02393 | 28.5 | 15.31, 24.5, 28.5, 0.02673, 15.38, 9.312, 9.188, 8 |
| 867 | `model.language_model.layers.14.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 133.5 | 1.173 | 3.198 | -1.164 | 77 | 0.2227, 0.2275, 0.2266, 0.5898, 0.2275, 0.1729, 0.2158, 0.2451 |
| 868 | `model.language_model.layers.14.self_attn.k_norm.weight` | 512 | bfloat16 | 1.381 | 0.06104 | 0 | 0.06104 | 0.06104 | 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104 |
| 869 | `model.language_model.layers.14.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 24.29 | -2.999e-05 | 0.02739 | -0.2217 | 0.2344 | -0.001007, -0.006378, -0.0188, -0.009155, 0.01257, -0.003723, -0.02014, -0.001579 |
| 870 | `model.language_model.layers.14.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 67.73 | -1.189e-05 | 0.02701 | -0.4355 | 0.7188 | -0.006409, -0.05713, 0.0144, -0.006317, -0.02429, -0.01495, 0.03687, -0.00589 |
| 871 | `model.language_model.layers.14.self_attn.q_norm.weight` | 512 | bfloat16 | 23.16 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 872 | `model.language_model.layers.14.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 81.6 | -1.04e-05 | 0.03255 | -0.4512 | 0.5 | 0.02466, -0.04785, 0.0007248, -0.008057, 0.01086, 0.009583, -0.01648, -0.013 |
| 873 | `model.language_model.layers.14.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 26.47 | -4.597e-05 | 0.02984 | -0.3418 | 0.377 | 0.02649, 0.009949, 0.02136, -0.0199, 0.001389, -0.0708, 0.03271, 0.008606 |
| 874 | `model.language_model.layers.15.input_layernorm.weight` | 1536 | bfloat16 | 713.4 | 7.921 | 16.4 | -147 | 194 | 3.312, 2.406, 3.125, 14.94, 3.828, 3.891, 3.312, 4.719 |
| 875 | `model.language_model.layers.15.layer_scalar` | 1 | bfloat16 | 0.2539 | 0.2539 | 0 | 0.2539 | 0.2539 | 0.2539 |
| 876 | `model.language_model.layers.15.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 101.1 | 2.759e-06 | 0.02334 | -0.5273 | 0.5078 | -0.02405, 0.0542, -0.0177, 0.009949, 0.02808, -0.03418, 0.02869, 0.01746 |
| 877 | `model.language_model.layers.15.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 146.7 | -3.063e-05 | 0.03388 | -0.6797 | 0.4785 | -0.01392, 0.06641, -0.02539, 0.0354, 0.008606, -0.01166, 0.04272, 0.01196 |
| 878 | `model.language_model.layers.15.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 147.5 | 1.166e-05 | 0.03404 | -0.4707 | 0.4297 | -0.01428, 0.009766, 0.0542, 0.01013, -0.04736, 0.01318, 0.03931, -0.03345 |
| 879 | `model.language_model.layers.15.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 5.088 | -3.372e-05 | 0.008114 | -0.08887 | 0.0957 | 0.00322, -0.0007362, -0.0003338, -0.0008736, -0.00415, 0.0003071, 0.005402, -0.005615 |
| 880 | `model.language_model.layers.15.per_layer_projection.weight` | 1536×256 | bfloat16 | 38.23 | 3.035e-05 | 0.06097 | -0.9805 | 0.8242 | -0.06299, -0.01782, -0.03857, 0.05127, -0.0564, -0.0282, 0.01489, 0.06104 |
| 881 | `model.language_model.layers.15.post_attention_layernorm.weight` | 1536 | bfloat16 | 36.67 | 0.9144 | 0.1982 | 0.000946 | 0.9844 | 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844, 0.9844 |
| 882 | `model.language_model.layers.15.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 211.8 | 5.132 | 1.696 | 0.06299 | 6.656 | 6.656, 6.656, 6.656, 2.906, 4.656, 4.031, 4.469, 4.469 |
| 883 | `model.language_model.layers.15.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 61.17 | 0.847 | 1.311 | 0.002411 | 3.562 | 1.539, 3.562, 3.562, 0.003326, 1.359, 0.00325, 0.002853, 0.003342 |
| 884 | `model.language_model.layers.15.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 55.67 | 0.9478 | 1.058 | -0.7188 | 20.5 | 0.707, 0.5938, 0.6055, 0.8516, 0.7539, 0.707, 0.6797, 0.7852 |
| 885 | `model.language_model.layers.15.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 886 | `model.language_model.layers.15.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 15.24 | 4.326e-05 | 0.0243 | -0.05518 | 0.05518 | 0.04736, 0.008972, 0.03662, -0.01941, -0.02844, 0.00473, -0.02795, -0.005096 |
| 887 | `model.language_model.layers.15.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 52.27 | -7.303e-06 | 0.02947 | -0.5547 | 0.4648 | 0.02087, -0.01868, 0.02136, 0.01672, -0.03613, -0.04077, 0.02649, -0.05298 |
| 888 | `model.language_model.layers.15.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 889 | `model.language_model.layers.15.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 58.78 | 1.369e-05 | 0.03315 | -0.4414 | 0.4434 | -0.01263, 0.004913, 0.01172, -0.003113, 0.01794, -0.001854, 0.001259, 0.001762 |
| 890 | `model.language_model.layers.15.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 15.24 | 1.711e-05 | 0.0243 | -0.05518 | 0.05518 | -0.0047, 0.01819, 0.01904, 0.00264, 0.04028, 0.01208, -0.01794, 0.01239 |
| 891 | `model.language_model.layers.16.input_layernorm.weight` | 1536 | bfloat16 | 772.9 | 7.996 | 18.03 | -6.125 | 237 | 3.453, 2.969, -3.484, 10.06, 4.594, 5.781, 4.594, 4.656 |
| 892 | `model.language_model.layers.16.layer_scalar` | 1 | bfloat16 | 0.5859 | 0.5859 | 0 | 0.5859 | 0.5859 | 0.5859 |
| 893 | `model.language_model.layers.16.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 105.3 | 5.355e-06 | 0.0243 | -0.5547 | 0.4219 | -0.003906, -0.01636, -0.01538, 0.00267, -0.00885, -0.02307, 0.07129, -0.02002 |
| 894 | `model.language_model.layers.16.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 140.8 | 7.52e-05 | 0.03251 | -0.5664 | 0.6016 | -0.01245, 0.01929, 0.02405, 0.0004673, -0.008728, -0.006683, 0.04907, 0.01111 |
| 895 | `model.language_model.layers.16.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 143.3 | 3.595e-06 | 0.03307 | -0.6211 | 0.668 | -0.04272, 0.05981, 0.07666, 0.05029, 0.009949, -0.04248, -0.05518, -0.06543 |
| 896 | `model.language_model.layers.16.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 10.43 | 1.74e-05 | 0.01663 | -0.4004 | 0.4043 | -0.01086, -0.01508, 0.01233, 0.05054, 0.01672, 0.004059, -0.01276, -0.007996 |
| 897 | `model.language_model.layers.16.per_layer_projection.weight` | 1536×256 | bfloat16 | 36.98 | -5.337e-05 | 0.05897 | -0.7812 | 0.7148 | -0.005737, -0.1079, -0.003342, 0.1582, 0.002579, 0.1943, 0.1621, -0.1318 |
| 898 | `model.language_model.layers.16.post_attention_layernorm.weight` | 1536 | bfloat16 | 36.09 | 0.8831 | 0.2607 | 0.0009422 | 1.008 | 1.008, 1.008, 1.008, 1.008, 1.008, 0.8125, 1.008, 1.008 |
| 899 | `model.language_model.layers.16.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 84.88 | 2.093 | 0.5556 | 0.002319 | 2.391 | 2.391, 2.391, 2.391, 2, 2.391, 2.094, 2.391, 2.328 |
| 900 | `model.language_model.layers.16.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 20.02 | 0.307 | 0.4084 | 0.0007668 | 0.9297 | 0.9297, 0.0008545, 0.9297, 0.000824, 0.6367, 0.0008659, 0.0008698, 0.0008354 |
| 901 | `model.language_model.layers.16.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 57.1 | 0.731 | 1.261 | 0.001907 | 26.62 | 0.4609, 0.4746, 0.543, 0.7031, 0.5469, 0.6484, 0.6055, 0.5742 |
| 902 | `model.language_model.layers.16.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 903 | `model.language_model.layers.16.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 15.41 | -4.901e-06 | 0.02458 | -0.05591 | 0.05591 | 0.01044, 0.01398, 0.04077, 0.0005531, -0.03833, 0.01392, 0.02148, 0.01501 |
| 904 | `model.language_model.layers.16.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 54.99 | 3.194e-06 | 0.03101 | -0.2969 | 0.2852 | 0.02124, -0.003281, 0.03394, 0.02942, 0.03662, -0.03516, 0.03516, -0.01685 |
| 905 | `model.language_model.layers.16.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 906 | `model.language_model.layers.16.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 59.72 | -1.139e-05 | 0.03368 | -0.4102 | 0.4258 | 0.04248, 0.01196, 0.01361, 0.01746, -0.02881, -0.004669, 0.01495, -0.01575 |
| 907 | `model.language_model.layers.16.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 15.41 | 0.0001092 | 0.02458 | -0.05566 | 0.05566 | -0.002304, -0.03442, -0.01208, -0.00174, 0.009888, -0.02502, -0.03442, 0.0188 |
| 908 | `model.language_model.layers.17.input_layernorm.weight` | 1536 | bfloat16 | 879.7 | 8.142 | 20.92 | -8.5 | 308 | 3.75, -3.453, 3.828, 4.594, 4.469, 5.719, 5.375, 3.969 |
| 909 | `model.language_model.layers.17.layer_scalar` | 1 | bfloat16 | 0.6562 | 0.6562 | 0 | 0.6562 | 0.6562 | 0.6562 |
| 910 | `model.language_model.layers.17.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 110.2 | -1.199e-06 | 0.02543 | -0.5742 | 0.6875 | -0.03174, 0.04761, -0.00946, 0.03296, -0.009705, -0.01941, -0.01343, -0.01001 |
| 911 | `model.language_model.layers.17.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 145.3 | 2.832e-05 | 0.03355 | -0.8008 | 0.8125 | 0.02185, -0.01477, -0.03442, -0.06836, -0.01636, 0.00322, 0.04492, -0.04028 |
| 912 | `model.language_model.layers.17.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 145.7 | -6.436e-07 | 0.03362 | -0.8555 | 0.8164 | -0.03784, 0.0007629, -0.08594, 0.02417, -0.06787, -0.001923, -0.02686, 0.004913 |
| 913 | `model.language_model.layers.17.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 11.73 | 0.000244 | 0.0187 | -0.2793 | 0.2598 | 0.0177, 0.01416, -5.555e-05, -0.01782, -0.02844, -8.917e-05, -0.005585, -0.02014 |
| 914 | `model.language_model.layers.17.per_layer_projection.weight` | 1536×256 | bfloat16 | 39.04 | -1.864e-05 | 0.06225 | -0.7227 | 0.6719 | -0.05518, 0.06934, 0.05762, -0.03589, 0.03613, 0.1187, 0.005157, -0.1631 |
| 915 | `model.language_model.layers.17.post_attention_layernorm.weight` | 1536 | bfloat16 | 27.48 | 0.6655 | 0.2205 | 0.0006943 | 0.7695 | 0.7695, 0.7695, 0.7695, 0.7695, 0.7695, 0.6914, 0.7695, 0.7695 |
| 916 | `model.language_model.layers.17.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 72.35 | 1.748 | 0.5948 | 0.001892 | 2.047 | 2.047, 2.047, 2.047, 2.047, 2.047, 1.836, 2.047, 2.047 |
| 917 | `model.language_model.layers.17.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 24.51 | 0.5838 | 0.2243 | 0.0006065 | 0.7227 | 0.7227, 0.7227, 0.7227, 0.6211, 0.7148, 0.4883, 0.5117, 0.7227 |
| 918 | `model.language_model.layers.17.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 57.68 | 0.7445 | 1.27 | 0.1348 | 26.5 | 0.5039, 0.5195, 0.5625, 0.6562, 0.5664, 0.5938, 0.5547, 0.5508 |
| 919 | `model.language_model.layers.17.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 920 | `model.language_model.layers.17.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 15.7 | 4.392e-05 | 0.02504 | -0.05688 | 0.05688 | -0.02319, 0.01263, -0.03027, 0.03271, -0.008911, -0.02893, -0.003143, 0.0108 |
| 921 | `model.language_model.layers.17.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 55.43 | 1.813e-06 | 0.03125 | -0.2773 | 0.332 | -0.0199, -0.0166, -0.02832, 0.01489, 0.026, -0.0437, -0.06445, 0.01733 |
| 922 | `model.language_model.layers.17.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 923 | `model.language_model.layers.17.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 59.6 | -2.769e-05 | 0.03361 | -0.4531 | 0.418 | 0.005157, 0.01031, 0.0002117, -0.006287, 0.003296, 0.009766, 0.0005569, 0.006073 |
| 924 | `model.language_model.layers.17.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 15.7 | 3.514e-05 | 0.02504 | -0.05688 | 0.05688 | 0.0105, -0.03589, -0.03247, 0.04492, -0.02283, -0.02368, -0.0002346, 0.006683 |
| 925 | `model.language_model.layers.18.input_layernorm.weight` | 1536 | bfloat16 | 925.1 | 7.713 | 22.31 | -14.38 | 368 | 4.031, 3.188, 3.562, 3.266, 3.781, 3.906, 3.297, 3.531 |
| 926 | `model.language_model.layers.18.layer_scalar` | 1 | bfloat16 | 0.6016 | 0.6016 | 0 | 0.6016 | 0.6016 | 0.6016 |
| 927 | `model.language_model.layers.18.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 102.3 | -1.784e-06 | 0.02361 | -0.4141 | 0.4648 | 0.009766, 0.01428, 0.03247, -0.01685, 0.001114, -0.01978, 0.03613, 0.005585 |
| 928 | `model.language_model.layers.18.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 150 | 6.179e-05 | 0.03462 | -0.5977 | 0.6914 | -0.01587, -0.01685, -0.009766, 0.02026, 0.02234, -0.01978, -0.01941, -0.01233 |
| 929 | `model.language_model.layers.18.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 149.3 | 1.173e-05 | 0.03446 | -0.8203 | 0.7305 | 0.03088, 0.02844, 0.07422, 0.0354, -0.0177, -0.008423, -0.02344, 0.00322 |
| 930 | `model.language_model.layers.18.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 11.64 | 0.0002171 | 0.01856 | -0.3672 | 0.373 | -0.005646, -0.01831, 0.0002937, -0.003448, -0.02258, -0.004974, -0.01794, 0.005127 |
| 931 | `model.language_model.layers.18.per_layer_projection.weight` | 1536×256 | bfloat16 | 40.54 | 1.683e-05 | 0.06464 | -0.7852 | 0.7969 | 0.1035, -0.09229, -0.06299, -0.03687, -0.06128, -0.0007133, -0.1201, 3.648e-05 |
| 932 | `model.language_model.layers.18.post_attention_layernorm.weight` | 1536 | bfloat16 | 27.59 | 0.6739 | 0.2038 | 0.0007057 | 0.7695 | 0.7617, 0.7695, 0.6875, 0.7695, 0.7695, 0.7695, 0.7695, 0.7695 |
| 933 | `model.language_model.layers.18.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 81.19 | 1.953 | 0.6902 | 0.002075 | 2.312 | 2.312, 2.312, 2.312, 2.312, 2.312, 2.016, 2.312, 2.312 |
| 934 | `model.language_model.layers.18.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 37.9 | 0.9179 | 0.3045 | 0.0007706 | 1.047 | 1.047, 1.047, 1.047, 1.047, 1.047, 1.047, 1.047, 1.047 |
| 935 | `model.language_model.layers.18.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 36.76 | 0.5746 | 0.7417 | 0.1465 | 22.5 | 0.4238, 0.4219, 0.459, 0.4727, 0.4941, 0.4688, 0.4355, 0.4473 |
| 936 | `model.language_model.layers.18.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 937 | `model.language_model.layers.18.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.12 | -6.227e-05 | 0.0257 | -0.05835 | 0.05835 | 0.004669, 0.01697, -0.01178, -0.05615, 0.04688, -0.03931, -0.001564, 0.002304 |
| 938 | `model.language_model.layers.18.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 54.96 | 1.563e-06 | 0.03099 | -0.3242 | 0.3613 | -0.02271, -0.03027, 0.02356, -0.01276, 0.001129, -0.03833, -0.03662, -0.03394 |
| 939 | `model.language_model.layers.18.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 940 | `model.language_model.layers.18.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 62.22 | -2.652e-05 | 0.03509 | -0.4609 | 0.4648 | 0.004517, 0.003052, 0.003677, -0.01001, -0.008423, 0.007019, 0.002594, 0.003159 |
| 941 | `model.language_model.layers.18.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.12 | 7.72e-05 | 0.0257 | -0.05835 | 0.05835 | -0.02734, 0.01929, -0.01514, -0.03662, 0.01941, -0.003616, -0.04346, -0.01404 |
| 942 | `model.language_model.layers.19.input_layernorm.weight` | 1536 | bfloat16 | 729.6 | 9.44 | 16.05 | -24.5 | 260 | 4.125, -4, 4.031, 4.031, 6.094, 6.344, 4.5, 4.469 |
| 943 | `model.language_model.layers.19.layer_scalar` | 1 | bfloat16 | 0.5391 | 0.5391 | 0 | 0.5391 | 0.5391 | 0.5391 |
| 944 | `model.language_model.layers.19.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 101 | -1.584e-06 | 0.02332 | -0.5352 | 0.4805 | -0.03345, 0.01019, -0.008423, -0.02209, 0.01324, 0.02795, 0.02039, -0.02393 |
| 945 | `model.language_model.layers.19.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 145.9 | 6.297e-05 | 0.03369 | -0.7266 | 0.7305 | -0.009094, 0.01709, -0.00592, 0.007141, 0.004486, 0.0127, -0.02283, -0.0007095 |
| 946 | `model.language_model.layers.19.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 145.7 | 9.522e-06 | 0.03363 | -1.039 | 0.5039 | 0.02478, 0.01758, -0.002014, -0.01343, -0.005676, 0.01184, 0.02502, 0.006531 |
| 947 | `model.language_model.layers.19.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 11.53 | 7.734e-05 | 0.01838 | -0.4238 | 0.4727 | -0.003967, 0.008301, 0.003967, 0.005463, -0.0166, 0.001312, 0.008789, -0.0006104 |
| 948 | `model.language_model.layers.19.per_layer_projection.weight` | 1536×256 | bfloat16 | 41.7 | 0.000114 | 0.0665 | -0.9375 | 0.8359 | -0.06934, -0.01385, -0.02734, 0.02271, -0.1406, 0.1201, -0.03662, -0.1025 |
| 949 | `model.language_model.layers.19.post_attention_layernorm.weight` | 1536 | bfloat16 | 41.24 | 1.01 | 0.2942 | 0.001038 | 1.164 | 1.047, 1.164, 0.9375, 1.164, 1.164, 1.164, 1.164, 1.164 |
| 950 | `model.language_model.layers.19.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 94.37 | 2.287 | 0.754 | 0.002563 | 2.688 | 2.688, 2.688, 2.688, 2.688, 2.109, 2.562, 2.688, 2.688 |
| 951 | `model.language_model.layers.19.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 38.96 | 0.9307 | 0.3491 | 0.0009308 | 1.109 | 1.109, 1.109, 1.109, 1.109, 0.6367, 1.109, 1.109, 1.109 |
| 952 | `model.language_model.layers.19.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 33.57 | 0.566 | 0.6433 | 0.1885 | 9.875 | 0.3691, 0.3516, 0.3984, 0.3711, 0.4043, 0.3887, 0.3652, 0.3594 |
| 953 | `model.language_model.layers.19.self_attn.k_norm.weight` | 512 | bfloat16 | 1.381 | 0.06104 | 0 | 0.06104 | 0.06104 | 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104 |
| 954 | `model.language_model.layers.19.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 23.34 | -2.966e-05 | 0.02632 | -0.05981 | 0.05981 | 0.0199, -0.03662, -0.01733, 0.03027, 0.01447, 0.005127, -0.02527, -0.01782 |
| 955 | `model.language_model.layers.19.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 68.9 | -9.147e-06 | 0.02748 | -0.291 | 0.5039 | 0.0177, 0.009827, -0.01245, 0.001556, -0.02136, -0.02441, 0.008911, 0.01917 |
| 956 | `model.language_model.layers.19.self_attn.q_norm.weight` | 512 | bfloat16 | 23.16 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 957 | `model.language_model.layers.19.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 84.66 | -6.168e-06 | 0.03377 | -0.6914 | 0.5156 | 0.01398, 0.0152, -0.01532, 0.02283, 0.003433, 0.007233, -0.004822, 0.00386 |
| 958 | `model.language_model.layers.19.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 23.34 | -2.953e-05 | 0.02632 | -0.05981 | 0.05981 | -0.01868, 0.004578, 0.008484, 0.01074, 0.02917, 0.005371, 0.00354, -0.05469 |
| 959 | `model.language_model.layers.2.input_layernorm.weight` | 1536 | bfloat16 | 1380 | 23.94 | 25.85 | -6.562 | 140 | 16.88, 0.4102, 3.266, 20.88, 33.5, 9.812, 53.5, 19.62 |
| 960 | `model.language_model.layers.2.layer_scalar` | 1 | bfloat16 | 0.793 | 0.793 | 0 | 0.793 | 0.793 | 0.793 |
| 961 | `model.language_model.layers.2.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 86.76 | 1.961e-06 | 0.02826 | -0.5039 | 0.5195 | 0.01746, -0.03027, -0.01111, -0.007721, -0.001518, 0.03931, -0.001205, 0.03491 |
| 962 | `model.language_model.layers.2.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 114.1 | -0.0001088 | 0.03719 | -0.4199 | 0.4629 | -0.03369, -0.009521, 0.006134, 0.07178, 0.0188, 0.002914, -0.02917, 0.005096 |
| 963 | `model.language_model.layers.2.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 115.1 | 3.514e-06 | 0.03751 | -0.4102 | 0.4414 | -0.06396, 0.02686, 0.05615, 0.02661, -0.03491, 0.005188, -0.03467, 0.008484 |
| 964 | `model.language_model.layers.2.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.83 | 2.218e-05 | 0.02684 | -0.2695 | 0.2754 | 0.01965, 0.005096, 0.001511, -0.02161, -0.003601, -0.01276, 0.02625, -0.03052 |
| 965 | `model.language_model.layers.2.per_layer_projection.weight` | 1536×256 | bfloat16 | 43.97 | 0.0002447 | 0.07012 | -0.4238 | 0.4922 | -0.1221, -0.01074, -0.1143, -0.01758, -0.1318, 0.04761, -0.01733, -0.05176 |
| 966 | `model.language_model.layers.2.post_attention_layernorm.weight` | 1536 | bfloat16 | 8.786 | 0.1563 | 0.1607 | 0.0003548 | 0.4082 | 0.01868, 0.4082, 0.2559, 0.01562, 0.006653, 0.3301, 0.004364, 0.01062 |
| 967 | `model.language_model.layers.2.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 35.71 | 0.6471 | 0.6416 | 0.001251 | 1.484 | 0.08643, 1.484, 1.484, 0.05396, 0.02332, 1.484, 0.01556, 0.03857 |
| 968 | `model.language_model.layers.2.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 1.06 | 0.02072 | 0.01738 | 2.301e-05 | 0.03735 | 0.03735, 0.03735, 0.03735, 0.03418, 0.00824, 0.03735, 3.266e-05, 0.01324 |
| 969 | `model.language_model.layers.2.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 448.4 | 7.466 | 8.673 | -0.001404 | 80 | 6.75, 0.3164, 1.133, 6.438, 14.75, 1.398, 19.62, 8.75 |
| 970 | `model.language_model.layers.2.self_attn.k_norm.weight` | 256 | bfloat16 | 1.922 | 0.1201 | 0 | 0.1201 | 0.1201 | 0.1201, 0.1201, 0.1201, 0.1201, 0.1201, 0.1201, 0.1201, 0.1201 |
| 971 | `model.language_model.layers.2.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 21.09 | 6.586e-05 | 0.03363 | -0.3418 | 0.3008 | -0.04639, -0.006866, 0.007935, -0.007721, 0.01154, -0.0304, -0.07568, -0.01331 |
| 972 | `model.language_model.layers.2.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 61.11 | -1.526e-05 | 0.03446 | -0.4453 | 0.4258 | 0.02283, 0.02039, 0.06201, 0.07471, 0.0415, 0.02466, 0.02917, 0.02698 |
| 973 | `model.language_model.layers.2.self_attn.q_norm.weight` | 256 | bfloat16 | 16.62 | 1.039 | 0 | 1.039 | 1.039 | 1.039, 1.039, 1.039, 1.039, 1.039, 1.039, 1.039, 1.039 |
| 974 | `model.language_model.layers.2.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 60.15 | -2.053e-05 | 0.03392 | -0.4434 | 0.4102 | -0.06006, 0.002808, -0.01672, -0.02673, 0.02563, 0.02002, -0.04492, -0.009094 |
| 975 | `model.language_model.layers.2.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 21.25 | -2.708e-05 | 0.03389 | -0.3438 | 0.4082 | -0.03613, 0.01251, -0.04468, -0.02905, 0.03491, -0.01007, 0.01721, -0.01483 |
| 976 | `model.language_model.layers.20.input_layernorm.weight` | 1536 | bfloat16 | 674.7 | 7.291 | 15.6 | -42 | 300 | 3.203, 3.438, 3.312, 3.391, 4.312, 3.812, 4.062, 3.188 |
| 977 | `model.language_model.layers.20.layer_scalar` | 1 | bfloat16 | 0.4941 | 0.4941 | 0 | 0.4941 | 0.4941 | 0.4941 |
| 978 | `model.language_model.layers.20.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 103.7 | 9.144e-07 | 0.02395 | -0.5664 | 0.707 | -0.01111, -0.00235, -0.006683, -0.01196, 0.02698, 0.03088, 0.01746, 0.02197 |
| 979 | `model.language_model.layers.20.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 144 | -5.561e-06 | 0.03326 | -0.625 | 0.6875 | -0.003128, -0.03589, -0.01465, -0.03467, -0.01428, 0.04224, 0.03662, 0.02429 |
| 980 | `model.language_model.layers.20.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 145.4 | 5.754e-06 | 0.03356 | -0.5039 | 0.5664 | -0.001953, -0.01929, -0.001801, 0.06885, -0.05469, 0.03613, 0.03857, 0.01733 |
| 981 | `model.language_model.layers.20.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 11.46 | 9.131e-05 | 0.01827 | -0.2598 | 0.3281 | -0.001801, -0.003799, -0.000412, 0.01929, 0.0155, -0.005341, -0.008362, 0.0006981 |
| 982 | `model.language_model.layers.20.per_layer_projection.weight` | 1536×256 | bfloat16 | 42.12 | 6.775e-06 | 0.06717 | -0.5781 | 0.5938 | -0.1455, 0.06226, -0.04785, 0.07666, -0.008301, 0.009277, 0.03491, 0.05273 |
| 983 | `model.language_model.layers.20.post_attention_layernorm.weight` | 1536 | bfloat16 | 36.42 | 0.8847 | 0.2843 | 0.0009003 | 1.023 | 1.023, 1.023, 1.023, 1.023, 0.8047, 1.023, 1.023, 1.023 |
| 984 | `model.language_model.layers.20.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 108.1 | 2.599 | 0.9225 | 0.00296 | 3.172 | 3.172, 3.172, 3.172, 3.172, 1.43, 3.172, 3.172, 2.797 |
| 985 | `model.language_model.layers.20.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 32.89 | 0.6708 | 0.5045 | 0.0009384 | 1.133 | 1.133, 0.001076, 1.133, 0.001099, 0.001053, 1.133, 0.001099, 0.001045 |
| 986 | `model.language_model.layers.20.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 51.16 | 0.6842 | 1.112 | 0.2373 | 14.12 | 0.3359, 0.3555, 0.3711, 0.3418, 0.3945, 0.3613, 0.3574, 0.3262 |
| 987 | `model.language_model.layers.20.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 988 | `model.language_model.layers.20.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.73 | 1.602e-05 | 0.02668 | -0.06055 | 0.06055 | 0.003143, 0.02905, -0.009277, -0.02173, 0.005371, -0.00322, -0.03296, 0.005035 |
| 989 | `model.language_model.layers.20.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 56.81 | 1.042e-05 | 0.03203 | -0.4355 | 0.4355 | -0.00325, -0.01624, 0.009827, -0.02283, 0.04565, -0.04517, -0.02686, -0.01697 |
| 990 | `model.language_model.layers.20.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 991 | `model.language_model.layers.20.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 64.43 | 3.351e-05 | 0.03633 | -0.6211 | 0.5664 | 0.05103, 0.008362, 0.08838, 0.02258, 0.001289, 0.1016, 0.03735, 0.06396 |
| 992 | `model.language_model.layers.20.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.73 | 9.312e-07 | 0.02668 | -0.06055 | 0.06055 | -0.001511, -0.03052, 0.04517, -0.03149, 0.02344, -0.007416, -0.026, -0.04443 |
| 993 | `model.language_model.layers.21.input_layernorm.weight` | 1536 | bfloat16 | 677.4 | 6.69 | 15.94 | -41.5 | 316 | 2.453, -3.219, 3.062, 2.719, 6.281, 3, 3.547, 2.859 |
| 994 | `model.language_model.layers.21.layer_scalar` | 1 | bfloat16 | 0.6445 | 0.6445 | 0 | 0.6445 | 0.6445 | 0.6445 |
| 995 | `model.language_model.layers.21.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 102.1 | -3.398e-06 | 0.02356 | -0.5391 | 0.5508 | 0.005463, 0.005554, 0.01196, -0.0007362, -0.03467, 0.002487, 0.02332, -0.01208 |
| 996 | `model.language_model.layers.21.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 146.6 | 7.04e-05 | 0.03385 | -0.6406 | 0.6133 | -0.004425, 0.03467, 0.01538, -0.03223, -0.03809, -0.03735, -0.0005913, 0.009949 |
| 997 | `model.language_model.layers.21.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 147.7 | 4.4e-06 | 0.03409 | -0.4766 | 0.5859 | -0.06787, -0.01331, -0.03394, 0.05054, -0.01404, 0.005829, -0.0006294, 0.008362 |
| 998 | `model.language_model.layers.21.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 14.61 | 0.0001262 | 0.02329 | -0.3574 | 0.3867 | -0.01611, -0.01038, 0.01495, 0.008545, 0.01758, 0.02039, 0.006561, 0.001541 |
| 999 | `model.language_model.layers.21.per_layer_projection.weight` | 1536×256 | bfloat16 | 45.75 | 5.509e-05 | 0.07295 | -0.8828 | 0.8711 | 0.02881, 0.0459, 0.2217, 0.1196, -0.001785, 0.05396, 0.08447, 0.04907 |
| 1000 | `model.language_model.layers.21.post_attention_layernorm.weight` | 1536 | bfloat16 | 28.29 | 0.6867 | 0.2224 | 0.0007248 | 0.7969 | 0.7969, 0.7969, 0.7969, 0.7969, 0.4863, 0.7969, 0.7969, 0.7734 |
| 1001 | `model.language_model.layers.21.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 67.72 | 1.605 | 0.6397 | 0.001923 | 2.047 | 2.047, 2.047, 2.047, 2.047, 0.5195, 1.945, 2.047, 1.438 |
| 1002 | `model.language_model.layers.21.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 28.44 | 0.6225 | 0.373 | 0.0007591 | 0.8906 | 0.8906, 0.8906, 0.8906, 0.8906, 0.0008583, 0.8906, 0.8906, 0.582 |
| 1003 | `model.language_model.layers.21.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 61.68 | 0.7599 | 1.379 | 0.03638 | 15.88 | 0.334, 0.3535, 0.3594, 0.3262, 0.457, 0.3574, 0.3516, 0.3418 |
| 1004 | `model.language_model.layers.21.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1005 | `model.language_model.layers.21.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 15.22 | -9.142e-07 | 0.02427 | -0.05518 | 0.05518 | 0.02515, -0.0166, 0.007538, -0.03882, 0.002594, -0.01611, -0.04126, 0.0002413 |
| 1006 | `model.language_model.layers.21.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 52.45 | -1.314e-05 | 0.02957 | -0.5312 | 0.459 | -0.008606, 0.006958, -0.007141, -0.05225, -0.01306, -0.007599, -0.01404, 0.05273 |
| 1007 | `model.language_model.layers.21.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1008 | `model.language_model.layers.21.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 58.35 | 6.877e-06 | 0.0329 | -0.5781 | 0.5391 | 0.003082, 0.01855, 0.03003, 0.01031, -0.002136, 0.007233, -0.02332, 0.03223 |
| 1009 | `model.language_model.layers.21.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 15.22 | 2.715e-05 | 0.02427 | -0.05518 | 0.05518 | 0.003113, -0.02991, -0.01019, 0.02368, 0.003525, -0.01758, 0.02612, -0.03589 |
| 1010 | `model.language_model.layers.22.input_layernorm.weight` | 1536 | bfloat16 | 913.5 | 6.936 | 22.26 | -41.25 | 488 | 2.656, 2.516, 1.938, 2.062, 4.812, 2.234, 2.25, 2.469 |
| 1011 | `model.language_model.layers.22.layer_scalar` | 1 | bfloat16 | 0.6328 | 0.6328 | 0 | 0.6328 | 0.6328 | 0.6328 |
| 1012 | `model.language_model.layers.22.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 94.26 | 8.079e-07 | 0.02175 | -0.4297 | 0.4961 | -0.01074, -0.0047, -0.04346, -0.02319, -0.0481, 0.01178, 0.01807, 0.01941 |
| 1013 | `model.language_model.layers.22.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 139.2 | 0.0001075 | 0.03212 | -0.5664 | 0.5039 | -0.07031, 0.04907, 0.02112, 0.103, 0.06494, -0.02161, 0.02063, 0.003754 |
| 1014 | `model.language_model.layers.22.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 137.7 | 5.735e-06 | 0.03178 | -0.6133 | 0.4512 | 0.02051, 0.01196, -0.0415, -0.006378, -0.1426, 0.05249, -0.02307, 0.01636 |
| 1015 | `model.language_model.layers.22.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 14.76 | 0.0001755 | 0.02353 | -0.3828 | 0.3262 | -0.02075, -0.01257, 0.0001478, -0.00946, 0.01782, -0.00325, -0.01471, 0.0002346 |
| 1016 | `model.language_model.layers.22.per_layer_projection.weight` | 1536×256 | bfloat16 | 41.81 | -9.601e-05 | 0.06667 | -0.6172 | 0.5781 | 0.003876, -0.07959, -0.03906, 0.01392, -0.007263, -0.03516, 0.06348, -0.04736 |
| 1017 | `model.language_model.layers.22.post_attention_layernorm.weight` | 1536 | bfloat16 | 26.25 | 0.6328 | 0.2194 | 0.0006638 | 0.7617 | 0.7617, 0.7617, 0.7617, 0.7617, 0.3262, 0.7617, 0.7617, 0.6328 |
| 1018 | `model.language_model.layers.22.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 66.22 | 1.557 | 0.6566 | 0.001968 | 2.047 | 2.047, 2.047, 2.047, 2.047, 0.3574, 1.781, 2.047, 1.195 |
| 1019 | `model.language_model.layers.22.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 33.61 | 0.7347 | 0.4427 | 0.0007858 | 1.07 | 1.07, 1.07, 1.07, 1.07, 0.000946, 0.8398, 1.07, 0.001007 |
| 1020 | `model.language_model.layers.22.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 78.23 | 0.8813 | 1.792 | 0.03394 | 17.75 | 0.3438, 0.3516, 0.3457, 0.3379, 0.6055, 0.3613, 0.3633, 0.3945 |
| 1021 | `model.language_model.layers.22.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1022 | `model.language_model.layers.22.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.26 | -7.928e-05 | 0.02593 | -0.05908 | 0.05908 | -0.009216, 0.04419, -0.0008659, 0.02686, 0.01239, 0.02539, 0.007782, 0.03857 |
| 1023 | `model.language_model.layers.22.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 57.08 | 1.921e-05 | 0.03218 | -0.6719 | 0.668 | -0.0007172, -0.01843, 0.01385, -0.02673, -0.01636, -0.02197, 0.0047, -0.05957 |
| 1024 | `model.language_model.layers.22.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1025 | `model.language_model.layers.22.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 62.09 | -2.906e-05 | 0.03501 | -0.5 | 0.5938 | 0.008423, 0.001419, -0.01794, -0.009827, 0.001915, 0.01746, 0.0155, -0.004578 |
| 1026 | `model.language_model.layers.22.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.26 | 3.936e-05 | 0.02593 | -0.05884 | 0.05884 | -0.04785, -0.00824, 0.03955, 0.04907, 0.002487, 0.01721, 0.01025, -0.05737 |
| 1027 | `model.language_model.layers.23.input_layernorm.weight` | 1536 | bfloat16 | 863.7 | 5.718 | 21.29 | -48.75 | 412 | 1.461, 1.406, 1.367, 1.297, 4.969, 1.969, 1.555, 2.016 |
| 1028 | `model.language_model.layers.23.layer_scalar` | 1 | bfloat16 | 0.4316 | 0.4316 | 0 | 0.4316 | 0.4316 | 0.4316 |
| 1029 | `model.language_model.layers.23.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 92.57 | 5.946e-06 | 0.02136 | -0.418 | 0.3379 | -0.008667, -0.008301, -0.04395, -0.02014, 0.007355, 0.0116, -0.01166, -0.0108 |
| 1030 | `model.language_model.layers.23.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 150.1 | 0.0001318 | 0.03466 | -0.5508 | 0.7656 | 0.003281, -0.02075, 0.05542, -0.02002, 0.02136, -0.007446, 0.02856, 0.02808 |
| 1031 | `model.language_model.layers.23.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 150.7 | 4.586e-06 | 0.03479 | -0.5898 | 0.5273 | 0.01562, 0.02661, 0.004639, -0.0007401, 0.002151, -0.006775, -0.01611, 0.01624 |
| 1032 | `model.language_model.layers.23.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 9.04 | 5.265e-07 | 0.01442 | -0.2129 | 0.2812 | 0.00824, -0.006622, -0.01562, -0.006592, -0.03882, 0.02673, -0.003036, 0.01855 |
| 1033 | `model.language_model.layers.23.per_layer_projection.weight` | 1536×256 | bfloat16 | 42.24 | 3.663e-05 | 0.06735 | -0.8555 | 0.4941 | -0.0007095, 0.03564, 0.06592, -0.02612, -0.02576, -0.02429, -0.00206, -0.01648 |
| 1034 | `model.language_model.layers.23.post_attention_layernorm.weight` | 1536 | bfloat16 | 22.24 | 0.5473 | 0.1502 | 0.0005608 | 0.6172 | 0.6172, 0.6172, 0.6172, 0.6172, 0.2852, 0.6172, 0.6172, 0.6172 |
| 1035 | `model.language_model.layers.23.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 108.1 | 2.495 | 1.175 | 0.003372 | 3.641 | 3.641, 3.641, 3.641, 3.641, 0.4121, 2.438, 3.328, 1.531 |
| 1036 | `model.language_model.layers.23.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 62.85 | 1.404 | 0.7754 | 0.001511 | 1.969 | 1.969, 0.001793, 1.969, 1.969, 0.001869, 1.273, 1.969, 0.8281 |
| 1037 | `model.language_model.layers.23.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 104.2 | 1.084 | 2.428 | 0.0874 | 27.62 | 0.418, 0.4316, 0.4062, 0.4199, 1.141, 0.4512, 0.4082, 0.5859 |
| 1038 | `model.language_model.layers.23.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1039 | `model.language_model.layers.23.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.68 | 5.631e-06 | 0.02659 | -0.06055 | 0.06055 | 0.02563, 0.02881, 0.01355, 0.004791, 0.01904, -0.01965, 0.03088, 0.0238 |
| 1040 | `model.language_model.layers.23.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 57.84 | 1.415e-06 | 0.03261 | -0.3906 | 0.4219 | 0.03564, -0.01434, 0.01477, 0.06543, -0.05859, 0.03979, 0.007996, -0.0177 |
| 1041 | `model.language_model.layers.23.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1042 | `model.language_model.layers.23.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 64.41 | -1.675e-05 | 0.03632 | -0.5039 | 0.5664 | 0.005829, 0.003204, -0.01721, 0.01917, -0.0166, 0.01062, -0.007874, -0.0001583 |
| 1043 | `model.language_model.layers.23.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.68 | -3.855e-05 | 0.02659 | -0.0603 | 0.0603 | 0.04272, 0.0415, 0.0008965, 0.01111, 0.021, 0.01807, 0.05176, -0.02466 |
| 1044 | `model.language_model.layers.24.input_layernorm.weight` | 1536 | bfloat16 | 437.8 | 5.494 | 9.731 | -23.62 | 82 | 1.539, -1.609, 1.414, 1.617, 9.25, 2.547, 1.836, 3.359 |
| 1045 | `model.language_model.layers.24.layer_scalar` | 1 | bfloat16 | 0.4375 | 0.4375 | 0 | 0.4375 | 0.4375 | 0.4375 |
| 1046 | `model.language_model.layers.24.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 104 | -4.403e-06 | 0.024 | -0.75 | 0.5977 | 0.01257, -0.008118, 0.001923, -0.02185, -0.01965, 0.005005, -0.01355, 0.01978 |
| 1047 | `model.language_model.layers.24.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 149.8 | 7.977e-05 | 0.03458 | -0.6484 | 0.5391 | 0.05298, -0.008911, 0.01306, -0.03198, 0.0007095, 0.01978, -0.0282, -0.02075 |
| 1048 | `model.language_model.layers.24.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 150.7 | -1.314e-06 | 0.03479 | -0.5781 | 0.582 | -0.007172, 0.01062, -2.73e-05, 0.02832, 0.004089, -0.02112, 0.02185, 0.04321 |
| 1049 | `model.language_model.layers.24.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 9.524 | 5.325e-06 | 0.01519 | -0.3184 | 0.1367 | -0.01086, 0.003387, 0.02051, -0.01178, 0.01202, -0.002014, 0.01782, -0.02698 |
| 1050 | `model.language_model.layers.24.per_layer_projection.weight` | 1536×256 | bfloat16 | 40.44 | -9.922e-05 | 0.0645 | -0.7266 | 0.707 | 0.1167, -0.1523, 0.207, -0.01672, 0.01294, -0.07031, 0.07959, -0.07471 |
| 1051 | `model.language_model.layers.24.post_attention_layernorm.weight` | 1536 | bfloat16 | 37.12 | 0.8745 | 0.364 | 0.001137 | 1.25 | 1.086, 1.25, 1.023, 1.25, 0.1582, 0.7305, 1.25, 0.498 |
| 1052 | `model.language_model.layers.24.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 111.5 | 2.692 | 0.9242 | 0.003342 | 3.656 | 3.656, 3.656, 3.656, 3.656, 0.7109, 2.453, 3.438, 1.969 |
| 1053 | `model.language_model.layers.24.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 41.31 | 0.88 | 0.5802 | 0.0009995 | 1.383 | 1.383, 0.001289, 1.383, 1.383, 0.208, 0.6875, 1.383, 0.001312 |
| 1054 | `model.language_model.layers.24.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 60.44 | 0.7367 | 1.355 | 0.1445 | 19.38 | 0.3066, 0.3359, 0.3262, 0.2871, 2.047, 0.4668, 0.3086, 0.6406 |
| 1055 | `model.language_model.layers.24.self_attn.k_norm.weight` | 512 | bfloat16 | 1.381 | 0.06104 | 0 | 0.06104 | 0.06104 | 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104 |
| 1056 | `model.language_model.layers.24.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 23.25 | 7.334e-06 | 0.02622 | -0.05957 | 0.05957 | 0.0152, 0.0271, 0.04004, 0.01978, -0.01532, -0.01831, -0.002869, -0.0332 |
| 1057 | `model.language_model.layers.24.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 65.38 | 4.611e-06 | 0.02608 | -0.4531 | 0.5273 | -0.02563, -0.03955, -0.01318, -0.04126, -0.003311, -0.02026, -0.01166, -0.0188 |
| 1058 | `model.language_model.layers.24.self_attn.q_norm.weight` | 512 | bfloat16 | 23.16 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 1059 | `model.language_model.layers.24.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 83.69 | -1.04e-05 | 0.03338 | -0.4395 | 0.5156 | 0.001709, -0.006317, -0.001144, -0.0002861, -0.001221, 0.002426, 0.004608, -0.005829 |
| 1060 | `model.language_model.layers.24.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 23.25 | -3.165e-05 | 0.02622 | -0.05957 | 0.05957 | -0.02612, 0.00235, -0.0271, -0.002426, -0.02502, -0.0332, 0.02148, 0.006317 |
| 1061 | `model.language_model.layers.25.input_layernorm.weight` | 1536 | bfloat16 | 405.7 | 4.07 | 9.52 | -9.375 | 144 | 1.445, 1.797, 1.203, 1.273, 14.62, 2.594, 1.453, 3.312 |
| 1062 | `model.language_model.layers.25.layer_scalar` | 1 | bfloat16 | 0.7852 | 0.7852 | 0 | 0.7852 | 0.7852 | 0.7852 |
| 1063 | `model.language_model.layers.25.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 105 | -2.125e-06 | 0.02423 | -0.5352 | 0.5156 | 0.04956, 0.00946, -0.04272, 0.01855, -0.009277, -0.03467, 0.02905, 0.0238 |
| 1064 | `model.language_model.layers.25.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 150.4 | -2.21e-05 | 0.03473 | -0.4961 | 0.6172 | -0.01233, 0.02563, 0.01141, -0.03125, 0.01575, 0.04004, -0.02344, 0.005737 |
| 1065 | `model.language_model.layers.25.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 152.3 | -8.763e-06 | 0.03515 | -0.5781 | 0.7188 | 0.08203, -0.03955, -0.04688, -0.01672, 0.002151, -0.0415, 0.004242, 0.01794 |
| 1066 | `model.language_model.layers.25.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.56 | -5.083e-05 | 0.0264 | -0.2676 | 0.334 | 0.003098, -0.01624, -0.0282, 0.002502, 0.0144, 0.00132, -0.0376, -0.04248 |
| 1067 | `model.language_model.layers.25.per_layer_projection.weight` | 1536×256 | bfloat16 | 42.04 | -3.746e-05 | 0.06704 | -0.7344 | 0.7344 | -0.0177, 0.04102, -0.02832, -0.05347, -0.001671, 0.04688, -0.1396, -0.009705 |
| 1068 | `model.language_model.layers.25.post_attention_layernorm.weight` | 1536 | bfloat16 | 25.34 | 0.6367 | 0.1117 | 0.0006523 | 0.6953 | 0.6953, 0.6953, 0.6953, 0.6953, 0.3223, 0.6953, 0.6953, 0.6953 |
| 1069 | `model.language_model.layers.25.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 48.3 | 1.219 | 0.1794 | 0.001244 | 1.312 | 1.312, 1.312, 1.312, 1.312, 0.6953, 1.312, 1.312, 1.312 |
| 1070 | `model.language_model.layers.25.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 16.56 | 0.3752 | 0.1945 | 0.0004349 | 0.5234 | 0.5234, 0.0005035, 0.0005074, 0.0005035, 0.2539, 0.3105, 0.5234, 0.3594 |
| 1071 | `model.language_model.layers.25.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 37.39 | 0.77 | 0.5633 | 0.1396 | 7.844 | 0.4453, 0.582, 0.5078, 0.4023, 2.391, 0.707, 0.4531, 0.875 |
| 1072 | `model.language_model.layers.25.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1073 | `model.language_model.layers.25.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.18 | -1.163e-05 | 0.0258 | -0.05859 | 0.05859 | 0.01483, 0.04565, 0.02234, -0.03955, 0.04736, -0.03223, -0.05249, 0.05029 |
| 1074 | `model.language_model.layers.25.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 59.87 | 4.462e-05 | 0.03376 | -0.6328 | 0.582 | -0.05615, -0.02454, -0.09521, -0.08057, 0.08496, -0.06885, -0.01056, 0.01794 |
| 1075 | `model.language_model.layers.25.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1076 | `model.language_model.layers.25.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 59.63 | -2.516e-05 | 0.03362 | -0.4609 | 0.4805 | 0.006866, -0.002304, -0.001221, 0.008057, -0.005463, 0.01648, -0.008667, 0.004608 |
| 1077 | `model.language_model.layers.25.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.18 | -2.732e-05 | 0.0258 | -0.05859 | 0.05859 | 0.01001, 0.003799, -0.009949, 0.01318, -0.02661, 0.005493, 0.005371, 0.03687 |
| 1078 | `model.language_model.layers.26.input_layernorm.weight` | 1536 | bfloat16 | 229.5 | 3.939 | 4.335 | -8.688 | 93.5 | 2.672, 3.141, 2.359, 3, 7.375, 4.031, 2.531, 3.5 |
| 1079 | `model.language_model.layers.26.layer_scalar` | 1 | bfloat16 | 0.8242 | 0.8242 | 0 | 0.8242 | 0.8242 | 0.8242 |
| 1080 | `model.language_model.layers.26.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 106.1 | -2.457e-06 | 0.02448 | -0.5352 | 0.457 | -0.03125, 0.003113, -0.0282, 0.04346, 0.000927, 0.01178, -0.03271, 0.01208 |
| 1081 | `model.language_model.layers.26.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 142.1 | -2.427e-05 | 0.03281 | -0.4219 | 0.625 | 0.008484, -0.009949, 0.006134, 0.03125, 0.007324, 0.04248, -0.01221, 0.04639 |
| 1082 | `model.language_model.layers.26.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 143.4 | -4.061e-06 | 0.03309 | -0.5586 | 0.5352 | -0.01746, 0.01422, -0.006134, 0.05005, 0.04663, 0.004272, 0.004456, -0.01337 |
| 1083 | `model.language_model.layers.26.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.67 | -0.0001474 | 0.02658 | -0.3652 | 0.3145 | -0.02771, -0.02917, -0.03418, -0.0188, 0.01093, 0.0004864, -0.02234, -0.003342 |
| 1084 | `model.language_model.layers.26.per_layer_projection.weight` | 1536×256 | bfloat16 | 40.25 | -8.954e-05 | 0.06419 | -0.7578 | 0.6367 | -0.02795, 0.08984, 0.02917, 0.05469, -0.052, -0.09424, 0.05981, 0.08594 |
| 1085 | `model.language_model.layers.26.post_attention_layernorm.weight` | 1536 | bfloat16 | 22.9 | 0.578 | 0.08534 | 0.0005722 | 0.6406 | 0.6406, 0.6406, 0.5938, 0.6406, 0.3223, 0.5859, 0.6406, 0.6133 |
| 1086 | `model.language_model.layers.26.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 41.92 | 1.058 | 0.1601 | 0.0009689 | 1.125 | 1.125, 1.125, 1.125, 1.031, 0.8672, 1.125, 1.125, 1.125 |
| 1087 | `model.language_model.layers.26.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 15.97 | 0.375 | 0.1596 | 0.0004005 | 0.4863 | 0.4863, 0.0004463, 0.3477, 0.0004654, 0.4805, 0.3574, 0.4863, 0.3652 |
| 1088 | `model.language_model.layers.26.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 36.47 | 0.8639 | 0.3462 | 0.1553 | 5.531 | 0.6172, 0.7773, 0.6875, 0.5938, 1.953, 0.8711, 0.6445, 0.957 |
| 1089 | `model.language_model.layers.26.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1090 | `model.language_model.layers.26.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.78 | -2.267e-05 | 0.02676 | -0.06079 | 0.06079 | 0.04028, 0.03027, 0.005951, -0.004547, -0.02356, -0.02258, -0.006073, 0.006958 |
| 1091 | `model.language_model.layers.26.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 59.21 | -4.988e-06 | 0.03339 | -0.4551 | 0.5156 | 0.03516, 0.06177, -0.05811, 0.01721, 0.03015, 0.004364, -0.001221, 0.03442 |
| 1092 | `model.language_model.layers.26.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1093 | `model.language_model.layers.26.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 63.28 | 3.28e-05 | 0.03568 | -0.5312 | 0.5078 | -0.01013, -0.005859, 0.01099, -0.003738, 0.008057, -0.002914, -0.01331, 0.006683 |
| 1094 | `model.language_model.layers.26.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.78 | 3.911e-05 | 0.02676 | -0.06079 | 0.06079 | 0.006805, 0.03369, 0.0238, 0.04272, -0.003815, 0.01892, -0.03662, 0.02051 |
| 1095 | `model.language_model.layers.27.input_layernorm.weight` | 1536 | bfloat16 | 462.8 | 7.394 | 9.21 | -17 | 234 | 5.562, 7.281, -5.469, 7.219, 8.562, 7.656, 5.25, 6.031 |
| 1096 | `model.language_model.layers.27.layer_scalar` | 1 | bfloat16 | 0.8203 | 0.8203 | 0 | 0.8203 | 0.8203 | 0.8203 |
| 1097 | `model.language_model.layers.27.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 105.6 | 6.27e-06 | 0.02437 | -0.5977 | 0.6484 | 0.00824, 0.001869, -0.0009499, -0.04004, 0.052, 0.03345, -0.01648, 0.02563 |
| 1098 | `model.language_model.layers.27.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 152.5 | -1.997e-05 | 0.03521 | -0.5234 | 0.5391 | 0.03345, 0.0007286, 0.00589, 0.02881, -0.0564, 0.04419, -0.02148, 0.03564 |
| 1099 | `model.language_model.layers.27.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 153.8 | -2.813e-06 | 0.0355 | -0.4043 | 0.4062 | -0.01538, 0.08008, 0.03809, -0.06299, -0.01282, -0.01178, -0.06787, -0.03589 |
| 1100 | `model.language_model.layers.27.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.37 | -0.0002616 | 0.0261 | -0.3164 | 0.3047 | -0.002228, -0.02173, 0.02502, 0.001289, -5.531e-05, 0.0009613, -0.04102, -0.006653 |
| 1101 | `model.language_model.layers.27.per_layer_projection.weight` | 1536×256 | bfloat16 | 40.36 | 0.0001003 | 0.06436 | -0.4102 | 0.377 | -0.03955, -0.07129, 0.125, -0.09814, -0.05273, -0.02271, 0.05273, -0.03955 |
| 1102 | `model.language_model.layers.27.post_attention_layernorm.weight` | 1536 | bfloat16 | 23.92 | 0.6042 | 0.08605 | 0.0005951 | 0.6602 | 0.6602, 0.6602, 0.6602, 0.6602, 0.3984, 0.6328, 0.6602, 0.6602 |
| 1103 | `model.language_model.layers.27.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 42.99 | 1.081 | 0.1846 | 0.001007 | 1.156 | 1.156, 1.156, 1.156, 0.957, 1.156, 1.156, 1.156, 1.156 |
| 1104 | `model.language_model.layers.27.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 14.09 | 0.3298 | 0.1431 | 0.0003319 | 0.4082 | 0.4082, 0.0003815, 0.0003719, 0.0003738, 0.4082, 0.3438, 0.4082, 0.3965 |
| 1105 | `model.language_model.layers.27.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 32.96 | 0.7971 | 0.2682 | 0.09473 | 5.75 | 0.625, 0.7812, 0.6719, 0.5977, 1.344, 0.8281, 0.6367, 0.8359 |
| 1106 | `model.language_model.layers.27.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1107 | `model.language_model.layers.27.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.23 | 3.09e-05 | 0.02589 | -0.05884 | 0.05884 | 0.00473, 0.01257, -0.02942, 0.02637, 0.0116, -0.02698, 0.01794, -0.01184 |
| 1108 | `model.language_model.layers.27.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 56.82 | 1.784e-05 | 0.03204 | -0.7383 | 0.9062 | 0.01892, 0.0005798, 0.03711, -0.02173, 0.03394, 0.0249, 0.01312, -0.01434 |
| 1109 | `model.language_model.layers.27.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1110 | `model.language_model.layers.27.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 60.18 | 2.029e-05 | 0.03393 | -0.418 | 0.4219 | 0.02502, 0.02441, 0.03467, -0.01086, 0.0144, 0.007751, 0.03198, -0.0007401 |
| 1111 | `model.language_model.layers.27.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.23 | 1.757e-05 | 0.02589 | -0.05884 | 0.05884 | -0.006897, -0.0007973, -0.0166, 0.03857, -0.01166, 0.03687, -0.03638, -0.02417 |
| 1112 | `model.language_model.layers.28.input_layernorm.weight` | 1536 | bfloat16 | 446.7 | 5.61 | 9.925 | -17.75 | 253 | 4.219, 3.891, 3.203, 5.781, 5.531, 3.953, 3.953, 3.391 |
| 1113 | `model.language_model.layers.28.layer_scalar` | 1 | bfloat16 | 0.8203 | 0.8203 | 0 | 0.8203 | 0.8203 | 0.8203 |
| 1114 | `model.language_model.layers.28.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 105.6 | -1.435e-06 | 0.02437 | -0.5156 | 0.4805 | -0.05859, -0.009827, -0.01697, 0.004578, -0.008606, 0.007477, 0.03369, 0.01154 |
| 1115 | `model.language_model.layers.28.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 146.7 | 5.079e-06 | 0.03388 | -0.4766 | 0.8359 | 0.04541, -0.0249, -0.01129, 0.01453, -0.01428, -0.01453, 0.00206, 0.006744 |
| 1116 | `model.language_model.layers.28.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 147.2 | 7.025e-06 | 0.03397 | -0.4629 | 0.4863 | 0.04004, 0.001633, 0.02075, 0.007477, 0.03564, 0.007446, -0.008057, -0.01398 |
| 1117 | `model.language_model.layers.28.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.43 | -0.0002688 | 0.0262 | -0.334 | 0.3027 | 0.02905, -0.002365, -0.01721, -0.001999, 0.003052, -0.001472, -0.01306, -0.03589 |
| 1118 | `model.language_model.layers.28.per_layer_projection.weight` | 1536×256 | bfloat16 | 39.68 | -0.0001186 | 0.06327 | -0.6953 | 0.4434 | 0.01166, 0.1221, -0.06348, 0.05762, 0.03467, -0.01733, 0.04614, -0.06299 |
| 1119 | `model.language_model.layers.28.post_attention_layernorm.weight` | 1536 | bfloat16 | 21.64 | 0.5475 | 0.07151 | 0.0005341 | 0.6016 | 0.5703, 0.6016, 0.5195, 0.6016, 0.459, 0.5508, 0.6016, 0.4746 |
| 1120 | `model.language_model.layers.28.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 43.56 | 1.097 | 0.1816 | 0.001099 | 1.172 | 1.172, 1.172, 1.164, 0.9531, 1.172, 1.172, 1.172, 1.172 |
| 1121 | `model.language_model.layers.28.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 15.54 | 0.3625 | 0.1609 | 0.000349 | 0.4609 | 0.4434, 0.0004139, 0.3203, 0.0004311, 0.4609, 0.4609, 0.4609, 0.4395 |
| 1122 | `model.language_model.layers.28.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 30.28 | 0.7453 | 0.2037 | 0.001816 | 4.25 | 0.6133, 0.75, 0.6758, 0.6094, 1, 0.7617, 0.6211, 0.7656 |
| 1123 | `model.language_model.layers.28.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1124 | `model.language_model.layers.28.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.51 | 3.216e-05 | 0.02633 | -0.06006 | 0.06006 | 0.006287, -0.05615, -0.0498, 0.009094, 0.008118, 0.01868, 0.03088, 0.05713 |
| 1125 | `model.language_model.layers.28.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 53.95 | -9.216e-06 | 0.03042 | -0.5664 | 0.4707 | 0.009277, -0.01636, -0.05127, -0.001648, 0.00589, -0.001602, -0.01306, -0.002991 |
| 1126 | `model.language_model.layers.28.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1127 | `model.language_model.layers.28.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 62.01 | -3.013e-06 | 0.03496 | -0.5039 | 0.416 | 0.009521, 0.007935, 0.003937, 0.00824, -0.0006409, 0.006409, 0.0119, 0.003876 |
| 1128 | `model.language_model.layers.28.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.51 | -1.923e-05 | 0.02633 | -0.05981 | 0.05981 | -0.04663, 0.02844, 0.03735, -0.002945, -0.03955, 0.04126, 0.04175, 0.03613 |
| 1129 | `model.language_model.layers.29.input_layernorm.weight` | 1536 | bfloat16 | 365 | 5.339 | 7.635 | -17 | 122.5 | 5.031, 3.469, 2.812, 5.688, 3.297, 4.5, 3.797, 2.812 |
| 1130 | `model.language_model.layers.29.layer_scalar` | 1 | bfloat16 | 0.8125 | 0.8125 | 0 | 0.8125 | 0.8125 | 0.8125 |
| 1131 | `model.language_model.layers.29.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 109.6 | 3.454e-06 | 0.0253 | -0.5898 | 0.5781 | -0.00135, 0.01202, -0.01123, -0.04053, -0.0332, 0.004089, -0.0199, -0.0144 |
| 1132 | `model.language_model.layers.29.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 152.4 | -7.453e-05 | 0.03518 | -0.7031 | 0.6445 | -0.04126, -0.02258, -0.007172, 0.03345, -0.02917, 0.01672, 0.003586, -0.07324 |
| 1133 | `model.language_model.layers.29.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 154.6 | -1.062e-05 | 0.03568 | -0.3477 | 0.3789 | -0.002029, -0.0415, 0.0005074, -0.0004787, -0.01624, -0.006775, -0.02417, 0.02954 |
| 1134 | `model.language_model.layers.29.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.5 | -0.0002217 | 0.02632 | -0.3691 | 0.3281 | -0.0199, 0.01044, 0.01099, 0.002823, -0.006622, 0.03857, -0.001389, -0.02148 |
| 1135 | `model.language_model.layers.29.per_layer_projection.weight` | 1536×256 | bfloat16 | 43.2 | 3.104e-05 | 0.06889 | -0.8008 | 0.8008 | 0.07373, -0.03174, -0.02661, 0.05347, 0.01648, -0.0708, 0.0603, 0.02063 |
| 1136 | `model.language_model.layers.29.post_attention_layernorm.weight` | 1536 | bfloat16 | 26.91 | 0.6834 | 0.06636 | 0.0006371 | 0.707 | 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707 |
| 1137 | `model.language_model.layers.29.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 43.12 | 1.084 | 0.1914 | 0.00106 | 1.164 | 1.164, 1.164, 1.164, 0.9453, 1.125, 1.164, 1.164, 1.164 |
| 1138 | `model.language_model.layers.29.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 13.42 | 0.2896 | 0.1828 | 0.000351 | 0.4746 | 0.3555, 0.000433, 0.2539, 0.0004501, 0.4746, 0.2871, 0.3027, 0.2734 |
| 1139 | `model.language_model.layers.29.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 29.02 | 0.7047 | 0.2277 | 0.001823 | 5.219 | 0.6133, 0.7266, 0.707, 0.6484, 0.7891, 0.6953, 0.5977, 0.7266 |
| 1140 | `model.language_model.layers.29.self_attn.k_norm.weight` | 512 | bfloat16 | 1.381 | 0.06104 | 0 | 0.06104 | 0.06104 | 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104 |
| 1141 | `model.language_model.layers.29.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 22.78 | -4.221e-05 | 0.02568 | -0.05835 | 0.05835 | -0.04004, -0.008789, 0.01733, 0.03589, 0.02942, -0.05005, 0.03735, -0.03809 |
| 1142 | `model.language_model.layers.29.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 67.19 | -5.078e-06 | 0.0268 | -1.211 | 0.6055 | 0.05249, -0.0002689, 0.02185, -0.02319, 0.06787, -0.01575, -0.01331, 0.02466 |
| 1143 | `model.language_model.layers.29.self_attn.q_norm.weight` | 512 | bfloat16 | 23.16 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 1144 | `model.language_model.layers.29.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 85.58 | -2.062e-06 | 0.03413 | -0.3945 | 0.4141 | 0.006744, -0.001671, -0.0008278, -0.007874, 1.365e-05, -0.001472, 0.002563, 0.002274 |
| 1145 | `model.language_model.layers.29.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 22.78 | 1.72e-05 | 0.02568 | -0.05835 | 0.05835 | 0.03711, 0.00946, 0.00235, -0.003906, 0.02454, 0.02026, -0.004089, 0.04907 |
| 1146 | `model.language_model.layers.3.input_layernorm.weight` | 1536 | bfloat16 | 1258 | 15.92 | 27.87 | -4.469 | 204 | 7.969, 0.3223, 0.6172, 12.75, 24.5, 0.7734, 41.25, 13.81 |
| 1147 | `model.language_model.layers.3.layer_scalar` | 1 | bfloat16 | 0.2871 | 0.2871 | 0 | 0.2871 | 0.2871 | 0.2871 |
| 1148 | `model.language_model.layers.3.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 86.95 | 1.002e-05 | 0.02832 | -0.707 | 0.6914 | 0.01111, -0.01929, -0.004913, -0.01379, 0.01904, 0.02734, -0.06299, -0.001259 |
| 1149 | `model.language_model.layers.3.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 106.2 | 3.496e-05 | 0.03462 | -0.3887 | 0.4004 | 0.02136, -0.01263, 0.05493, 0.003906, 0.03857, -0.08203, 0.0481, 0.0791 |
| 1150 | `model.language_model.layers.3.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 108.1 | 3.992e-06 | 0.03524 | -0.4023 | 0.3828 | 0.01428, 0.002762, 0.03931, 0.01239, 0.01324, 0.01917, 0.003464, -0.04297 |
| 1151 | `model.language_model.layers.3.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 6.496 | -5.016e-05 | 0.01036 | -0.1191 | 0.1143 | 0.007629, 0.006409, 0.006866, 0.01843, 0.009766, 0.009827, -0.0003185, 0.005035 |
| 1152 | `model.language_model.layers.3.per_layer_projection.weight` | 1536×256 | bfloat16 | 43.43 | 0.0002131 | 0.06926 | -0.7891 | 0.8086 | 0.05273, 0.02332, 0.00647, -0.07129, -0.07324, 0.01917, -0.02454, -0.009521 |
| 1153 | `model.language_model.layers.3.post_attention_layernorm.weight` | 1536 | bfloat16 | 48.47 | 0.8005 | 0.9432 | 0.001801 | 2.484 | 0.04053, 2.484, 0.7422, 0.02979, 0.01074, 1.633, 0.008423, 0.0177 |
| 1154 | `model.language_model.layers.3.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 163.9 | 3.049 | 2.863 | 0.004028 | 6.188 | 0.2285, 6.188, 6.188, 0.1235, 0.04468, 6.188, 0.02893, 0.08057 |
| 1155 | `model.language_model.layers.3.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 7.941 | 0.1333 | 0.1527 | 0.0001955 | 0.3145 | 0.3145, 0.3145, 0.3145, 0.08008, 0.0002193, 0.3145, 0.0002766, 0.0002499 |
| 1156 | `model.language_model.layers.3.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 867.2 | 12.87 | 18.01 | -0.3613 | 123.5 | 11.44, 0.3496, 0.6523, 13, 29, 0.5234, 42.75, 17.75 |
| 1157 | `model.language_model.layers.3.self_attn.k_norm.weight` | 256 | bfloat16 | 1.945 | 0.1216 | 0 | 0.1216 | 0.1216 | 0.1216, 0.1216, 0.1216, 0.1216, 0.1216, 0.1216, 0.1216, 0.1216 |
| 1158 | `model.language_model.layers.3.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 23.01 | 2.197e-06 | 0.03669 | -0.3457 | 0.4199 | -0.005524, 0.005554, 0.02808, -0.007385, 0.003708, -0.002625, 0.001785, 0.01965 |
| 1159 | `model.language_model.layers.3.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 57.82 | -1.369e-06 | 0.0326 | -0.6602 | 0.625 | -0.004486, -0.01172, -0.01843, 0.06348, -0.003082, 0.0376, -0.005524, 0.008728 |
| 1160 | `model.language_model.layers.3.self_attn.q_norm.weight` | 256 | bfloat16 | 16.5 | 1.031 | 0 | 1.031 | 1.031 | 1.031, 1.031, 1.031, 1.031, 1.031, 1.031, 1.031, 1.031 |
| 1161 | `model.language_model.layers.3.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 64.37 | 2.582e-05 | 0.0363 | -0.5352 | 0.5 | 0.005981, 0.005066, 0.052, 0.002014, 0.01062, 0.01361, 0.01105, -0.01239 |
| 1162 | `model.language_model.layers.3.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 23.5 | -1.256e-05 | 0.03748 | -0.3105 | 0.332 | -0.004059, -8.059e-05, 0.01361, -0.02209, -0.02051, -0.03735, 0.03149, -0.003433 |
| 1163 | `model.language_model.layers.30.input_layernorm.weight` | 1536 | bfloat16 | 664.9 | 8.645 | 14.6 | -30 | 354 | 6.688, -7.438, 6.062, 8.625, 7.219, 6.312, 5.844, 5.219 |
| 1164 | `model.language_model.layers.30.layer_scalar` | 1 | bfloat16 | 0.8711 | 0.8711 | 0 | 0.8711 | 0.8711 | 0.8711 |
| 1165 | `model.language_model.layers.30.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 113.2 | 4.178e-06 | 0.02613 | -0.4668 | 0.5664 | -0.06641, 0.005554, -0.008423, -0.002792, 0.001877, 0.006958, 0.01172, 0.002991 |
| 1166 | `model.language_model.layers.30.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 148 | -9.349e-05 | 0.03416 | -0.8711 | 0.5703 | -0.04443, 0.06445, -0.0271, 0.0008812, -0.006256, 0.0003757, 0.03149, 0.0271 |
| 1167 | `model.language_model.layers.30.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 150.5 | -7.527e-06 | 0.03472 | -0.3672 | 0.3867 | -0.01453, -0.04321, -0.0019, -0.04126, -0.007874, 0.05957, 0.02173, 0.009399 |
| 1168 | `model.language_model.layers.30.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 18.01 | -0.0001915 | 0.02872 | -0.3887 | 0.4082 | 0.009399, -0.04102, -0.02588, 0.02576, -0.01965, 0.01587, 0.006287, -0.01978 |
| 1169 | `model.language_model.layers.30.per_layer_projection.weight` | 1536×256 | bfloat16 | 39.62 | 0.0001232 | 0.06319 | -0.7383 | 0.7266 | 0.0005836, -0.001945, 0.00592, 0.0001621, -0.02002, 0.003754, 0.001869, -0.0019 |
| 1170 | `model.language_model.layers.30.post_attention_layernorm.weight` | 1536 | bfloat16 | 12.41 | 0.3128 | 0.0497 | 0.0002937 | 0.3281 | 0.3281, 0.3281, 0.3281, 0.3281, 0.3281, 0.3281, 0.3281, 0.3281 |
| 1171 | `model.language_model.layers.30.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 35.04 | 0.874 | 0.1883 | 0.0008621 | 0.9609 | 0.9375, 0.9609, 0.957, 0.6836, 0.9609, 0.9609, 0.9609, 0.9609 |
| 1172 | `model.language_model.layers.30.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 10.83 | 0.1932 | 0.1977 | 0.0003204 | 0.4863 | 0.0003452, 0.0004559, 0.0004444, 0.0004749, 0.4863, 0.4863, 0.000412, 0.2285 |
| 1173 | `model.language_model.layers.30.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 24.69 | 0.6177 | 0.1233 | 0.001656 | 3.797 | 0.5547, 0.6016, 0.5977, 0.5781, 0.6953, 0.625, 0.5586, 0.6289 |
| 1174 | `model.language_model.layers.30.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1175 | `model.language_model.layers.30.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 15.71 | -2.411e-05 | 0.02505 | -0.05713 | 0.05713 | -0.01318, -0.00209, -0.02344, -0.02124, -0.02625, -0.005768, 0.01245, 0.01245 |
| 1176 | `model.language_model.layers.30.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 56.1 | 1.424e-05 | 0.03163 | -0.6094 | 0.3984 | -0.01965, -0.0188, -0.02649, 0.01501, -0.02307, -0.03516, 0.01611, -0.005219 |
| 1177 | `model.language_model.layers.30.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1178 | `model.language_model.layers.30.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 56.11 | -2.782e-05 | 0.03164 | -0.375 | 0.4316 | -0.005798, 0.002686, 0.003403, 0.01227, -0.01135, -0.006042, -0.008179, 0.007111 |
| 1179 | `model.language_model.layers.30.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 15.71 | -1.529e-05 | 0.02505 | -0.05688 | 0.05688 | -0.01239, -0.02539, -0.04932, 0.02307, -0.001778, -0.002121, 0.001472, 0.007935 |
| 1180 | `model.language_model.layers.31.input_layernorm.weight` | 1536 | bfloat16 | 1064 | 15.17 | 22.54 | -44 | 588 | 12, 11.81, 10.94, 16.25, 12.56, 10.94, 11.56, 9.75 |
| 1181 | `model.language_model.layers.31.layer_scalar` | 1 | bfloat16 | 0.8281 | 0.8281 | 0 | 0.8281 | 0.8281 | 0.8281 |
| 1182 | `model.language_model.layers.31.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 107.3 | 1.771e-07 | 0.02476 | -0.4766 | 0.4922 | -0.0542, -0.0271, 0.01337, 0.01044, 0.01672, 0.02258, 0.0271, -0.01062 |
| 1183 | `model.language_model.layers.31.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 141 | -4.829e-05 | 0.03255 | -0.4883 | 0.4375 | 0.03101, 0.004303, 0.03516, 0.01141, -0.007568, -0.03064, -0.02979, 0.007996 |
| 1184 | `model.language_model.layers.31.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 143 | 7.409e-06 | 0.033 | -0.3379 | 0.3789 | -0.01917, 0.01099, -0.04248, 0.06348, 0.004639, 0.02075, 0.02112, 0.0708 |
| 1185 | `model.language_model.layers.31.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.95 | -0.0002367 | 0.02702 | -0.3125 | 0.377 | -0.05493, 0.01746, -0.009033, 0.02576, -0.006744, -0.01624, 0.02148, 0.006256 |
| 1186 | `model.language_model.layers.31.per_layer_projection.weight` | 1536×256 | bfloat16 | 41.12 | 4.866e-05 | 0.06558 | -0.793 | 0.8008 | 0.004791, -0.0007591, -0.02026, 0.004028, -0.004181, -0.002655, -0.002563, 0.002686 |
| 1187 | `model.language_model.layers.31.post_attention_layernorm.weight` | 1536 | bfloat16 | 19.07 | 0.4774 | 0.09346 | 0.0004311 | 0.5117 | 0.5117, 0.5117, 0.5117, 0.4395, 0.5117, 0.5117, 0.5117, 0.5117 |
| 1188 | `model.language_model.layers.31.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 42.54 | 1.061 | 0.229 | 0.001091 | 1.164 | 1.125, 1.164, 1.148, 0.8398, 1.164, 1.164, 1.164, 1.164 |
| 1189 | `model.language_model.layers.31.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 10.53 | 0.1715 | 0.2069 | 0.0002766 | 0.4707 | 0.0004387, 0.0004272, 0.0004444, 0.000433, 0.4707, 0.4707, 0.0004368, 0.4707 |
| 1190 | `model.language_model.layers.31.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 22.17 | 0.5535 | 0.1168 | 0.0006561 | 3.625 | 0.5117, 0.5312, 0.5352, 0.5469, 0.5742, 0.543, 0.5117, 0.5469 |
| 1191 | `model.language_model.layers.31.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1192 | `model.language_model.layers.31.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.59 | 4.177e-05 | 0.02645 | -0.0603 | 0.0603 | -0.01172, -0.04419, -0.02454, -0.0006638, -0.0238, 0.01196, -0.01648, 0.05811 |
| 1193 | `model.language_model.layers.31.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 56.43 | -2.367e-05 | 0.03182 | -0.6094 | 0.6016 | -0.008057, -0.01324, -0.003433, -0.007324, -0.01819, 0.006134, 0.04199, -0.006409 |
| 1194 | `model.language_model.layers.31.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1195 | `model.language_model.layers.31.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 59.61 | 1.417e-05 | 0.03361 | -0.4395 | 0.3965 | -0.004913, -0.01575, 0.02478, -0.02173, 0.01154, 0.009583, -0.01001, -0.0293 |
| 1196 | `model.language_model.layers.31.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.59 | -2.216e-05 | 0.02645 | -0.06006 | 0.06006 | 0.03857, -0.02368, 0.00412, -0.00325, 0.0304, -0.01477, 0.002136, -0.02466 |
| 1197 | `model.language_model.layers.32.input_layernorm.weight` | 1536 | bfloat16 | 740.5 | 8.658 | 16.8 | -21.5 | 420 | 6.156, 6.375, 5.875, 9.062, 6.062, 5.531, 5.688, 4.625 |
| 1198 | `model.language_model.layers.32.layer_scalar` | 1 | bfloat16 | 0.8711 | 0.8711 | 0 | 0.8711 | 0.8711 | 0.8711 |
| 1199 | `model.language_model.layers.32.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 105.9 | 1.087e-05 | 0.02444 | -0.7852 | 0.4785 | -0.01544, -0.02625, 0.01917, 0.0177, 0.03735, 0.002151, -0.007141, 0.04712 |
| 1200 | `model.language_model.layers.32.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 148.5 | -0.000119 | 0.03427 | -0.5547 | 0.6133 | 0.03076, 0.02185, -0.03613, 0.000288, 0.02527, -0.02344, 0.02441, 0.03882 |
| 1201 | `model.language_model.layers.32.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 150.6 | 6.756e-06 | 0.03477 | -0.4453 | 0.5117 | -0.01746, -0.00383, 0.05762, -0.01538, 0.04565, -0.021, 0.07275, -0.0008469 |
| 1202 | `model.language_model.layers.32.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 18.29 | -0.0002871 | 0.02917 | -0.291 | 0.3203 | -0.001595, -0.01904, -0.008484, -0.02881, 0.03198, 0.03442, -0.05713, -0.00531 |
| 1203 | `model.language_model.layers.32.per_layer_projection.weight` | 1536×256 | bfloat16 | 43.37 | -0.0001112 | 0.06916 | -0.9453 | 0.9375 | -0.001396, -0.00322, 6.032e-05, -0.009094, 0.006958, 0.006653, -0.006378, 0.001137 |
| 1204 | `model.language_model.layers.32.post_attention_layernorm.weight` | 1536 | bfloat16 | 12.42 | 0.3138 | 0.04406 | 0.0002937 | 0.3281 | 0.3281, 0.3281, 0.3281, 0.3281, 0.2793, 0.3281, 0.3281, 0.3281 |
| 1205 | `model.language_model.layers.32.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 34.5 | 0.856 | 0.206 | 0.0008736 | 0.9648 | 0.875, 0.9648, 0.9258, 0.6328, 0.9648, 0.9648, 0.9141, 0.9258 |
| 1206 | `model.language_model.layers.32.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 10.06 | 0.1606 | 0.2002 | 0.0003147 | 0.4707 | 0.0004234, 0.0004368, 0.2969, 0.0004463, 0.0003929, 0.4707, 0.0004292, 0.0004368 |
| 1207 | `model.language_model.layers.32.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 18.81 | 0.4673 | 0.1094 | 0.001671 | 3.203 | 0.4355, 0.4395, 0.4551, 0.4707, 0.4668, 0.4395, 0.4219, 0.4434 |
| 1208 | `model.language_model.layers.32.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1209 | `model.language_model.layers.32.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.8 | 6.214e-05 | 0.02679 | -0.06104 | 0.06104 | 0.02258, -0.01184, -0.0293, -0.004852, -0.03589, -0.02332, 0.04956, 0.007629 |
| 1210 | `model.language_model.layers.32.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 56.58 | 4.316e-06 | 0.0319 | -0.4375 | 0.4004 | -0.005615, -0.05688, -0.02197, -0.01746, 0.008911, -0.03711, 0.008911, 0.0119 |
| 1211 | `model.language_model.layers.32.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1212 | `model.language_model.layers.32.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 59.11 | 2.725e-05 | 0.03333 | -0.3984 | 0.4297 | 0.01465, 0.001007, 0.01599, 0.004456, -0.006744, 0.0155, 0.02002, -0.01318 |
| 1213 | `model.language_model.layers.32.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.8 | 1.723e-05 | 0.02679 | -0.06104 | 0.06104 | 0.03491, 0.02649, -0.03271, -0.0166, -0.02148, -0.003296, 0.02686, -0.01068 |
| 1214 | `model.language_model.layers.33.input_layernorm.weight` | 1536 | bfloat16 | 405.2 | 3.911 | 9.575 | -5.438 | 237 | 2.297, 2.812, 2.312, 4.062, 1.859, 1.828, 2.172, 1.883 |
| 1215 | `model.language_model.layers.33.layer_scalar` | 1 | bfloat16 | 0.6953 | 0.6953 | 0 | 0.6953 | 0.6953 | 0.6953 |
| 1216 | `model.language_model.layers.33.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 111.6 | -7.731e-06 | 0.02577 | -0.5781 | 0.5195 | 0.03015, -0.0282, -0.02759, 0.006012, 0.0199, 0.01086, 0.05859, -0.02148 |
| 1217 | `model.language_model.layers.33.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 142.5 | -7.36e-05 | 0.03289 | -0.5234 | 0.6016 | -0.01562, -0.03931, -0.03687, -0.05933, -0.01013, -0.02368, -0.006866, 0.01892 |
| 1218 | `model.language_model.layers.33.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 144.8 | -1.677e-05 | 0.03343 | -0.6094 | 0.4453 | -0.006042, -0.01624, -0.01489, -0.02808, -0.0542, 0.01276, 0.02539, -0.04858 |
| 1219 | `model.language_model.layers.33.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 17.08 | -0.0001259 | 0.02723 | -0.3535 | 0.3828 | -0.01105, 0.005463, -0.0498, -0.00412, -0.0009346, -0.01416, -0.04419, 0.02576 |
| 1220 | `model.language_model.layers.33.per_layer_projection.weight` | 1536×256 | bfloat16 | 42.56 | 0.0001523 | 0.06787 | -0.5234 | 0.6523 | 0.08252, -0.05811, 0.07666, 0.2354, -0.1836, -0.04907, 0.05957, 0.1523 |
| 1221 | `model.language_model.layers.33.post_attention_layernorm.weight` | 1536 | bfloat16 | 13.62 | 0.3459 | 0.03374 | 0.0003376 | 0.3574 | 0.3574, 0.3574, 0.3574, 0.3574, 0.2812, 0.3574, 0.3574, 0.3574 |
| 1222 | `model.language_model.layers.33.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 46.58 | 1.172 | 0.1951 | 0.001198 | 1.258 | 1.258, 1.258, 1.258, 1.148, 1.141, 1.258, 1.211, 1.234 |
| 1223 | `model.language_model.layers.33.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 48.39 | 1.083 | 0.5924 | 0.001266 | 1.609 | 0.6289, 0.001465, 0.9141, 0.8438, 0.001427, 0.75, 0.9102, 0.9258 |
| 1224 | `model.language_model.layers.33.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 23.01 | 0.542 | 0.2258 | -0.0007591 | 3.859 | 0.4863, 0.5039, 0.4922, 0.5859, 0.498, 0.4473, 0.4629, 0.4648 |
| 1225 | `model.language_model.layers.33.self_attn.k_norm.weight` | 256 | bfloat16 | 2.016 | 0.126 | 0 | 0.126 | 0.126 | 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126 |
| 1226 | `model.language_model.layers.33.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 16.36 | -4.462e-05 | 0.02609 | -0.05933 | 0.05933 | 0.03833, 0.004639, -0.00177, -0.01843, 0.006073, 0.0005264, -0.006012, -0.008911 |
| 1227 | `model.language_model.layers.33.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 55.14 | 7.972e-06 | 0.03109 | -0.3125 | 0.2832 | -0.009888, 0.01733, -0.02966, -0.009766, 0.003891, 0.05029, 0.04199, -0.0564 |
| 1228 | `model.language_model.layers.33.self_attn.q_norm.weight` | 256 | bfloat16 | 15.88 | 0.9922 | 0 | 0.9922 | 0.9922 | 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922, 0.9922 |
| 1229 | `model.language_model.layers.33.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 58.04 | 4.982e-06 | 0.03273 | -0.4414 | 0.4922 | 0.008789, 0.004974, -0.004425, -0.001808, -0.01044, -0.01123, -0.00766, 0.01068 |
| 1230 | `model.language_model.layers.33.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 16.36 | 7.857e-05 | 0.02609 | -0.05933 | 0.05933 | -0.03223, -0.005676, -0.0304, -0.04053, -0.01483, -0.02332, 0.05225, 0.03442 |
| 1231 | `model.language_model.layers.34.input_layernorm.weight` | 1536 | bfloat16 | 614.4 | 8.036 | 13.47 | -21.5 | 229 | 6.375, 5.562, 4.094, 9.875, 3.703, 3.5, 4.75, 3.188 |
| 1232 | `model.language_model.layers.34.layer_scalar` | 1 | bfloat16 | 0.167 | 0.167 | 0 | 0.167 | 0.167 | 0.167 |
| 1233 | `model.language_model.layers.34.mlp.down_proj.weight` | 1536×12288 | bfloat16 | 107.5 | -5.969e-06 | 0.02482 | -0.5859 | 0.6992 | -0.04077, 0.004425, -0.0009613, -0.02209, 0.01514, -0.01117, 0.01263, 0.03345 |
| 1234 | `model.language_model.layers.34.mlp.gate_proj.weight` | 12288×1536 | bfloat16 | 147.5 | -8.961e-05 | 0.03405 | -0.5547 | 0.4961 | -0.03076, -0.0282, -0.04199, -0.04077, 0.02722, 0.001472, -0.009827, 0.02649 |
| 1235 | `model.language_model.layers.34.mlp.up_proj.weight` | 12288×1536 | bfloat16 | 152.9 | 2.021e-06 | 0.03528 | -0.5664 | 0.5156 | -0.06348, -0.01672, 0.005859, -0.01093, -0.05444, 0.001427, -0.01465, -0.02136 |
| 1236 | `model.language_model.layers.34.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 3.043 | 1.092e-05 | 0.004853 | -0.06592 | 0.06055 | -0.001488, 0.000124, -0.002106, 0.001038, 0.0001054, -0.0001411, -0.001778, -0.0005951 |
| 1237 | `model.language_model.layers.34.per_layer_projection.weight` | 1536×256 | bfloat16 | 44.47 | -0.0001463 | 0.07091 | -0.7852 | 0.7812 | 0.05762, -0.01965, 0.03564, -0.03125, -0.2383, -0.05566, -0.0282, 0.06494 |
| 1238 | `model.language_model.layers.34.post_attention_layernorm.weight` | 1536 | bfloat16 | 20.28 | 0.462 | 0.233 | 0.001213 | 1.383 | 0.3945, 1.383, 0.3711, 0.3301, 0.4512, 0.457, 0.4199, 0.4121 |
| 1239 | `model.language_model.layers.34.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 70.67 | 1.216 | 1.332 | 0.01105 | 11.69 | 1.023, 2.219, 1.133, 1.023, 0.8477, 0.875, 1.016, 0.9727 |
| 1240 | `model.language_model.layers.34.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 26.55 | 0.6054 | 0.3042 | 0.0008087 | 0.9141 | 0.9141, 0.9141, 0.9141, 0.4551, 0.3984, 0.6797, 0.6562, 0.543 |
| 1241 | `model.language_model.layers.34.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 30.46 | 0.6823 | 0.3724 | 0.2041 | 5.844 | 0.6172, 0.6602, 0.6914, 0.8359, 0.5938, 0.5156, 0.582, 0.6445 |
| 1242 | `model.language_model.layers.34.self_attn.k_norm.weight` | 512 | bfloat16 | 1.381 | 0.06104 | 0 | 0.06104 | 0.06104 | 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104, 0.06104 |
| 1243 | `model.language_model.layers.34.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 23.33 | -2.969e-05 | 0.02631 | -0.05981 | 0.05981 | -0.005585, 0.005157, 0.007874, -0.03589, 0.02246, -0.04443, 0.02002, -0.01044 |
| 1244 | `model.language_model.layers.34.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 75.71 | 1.118e-05 | 0.0302 | -0.3398 | 0.3301 | -0.007172, 0.009216, 0.06738, 0.0127, -0.03198, -0.01587, -0.004089, 0.04639 |
| 1245 | `model.language_model.layers.34.self_attn.q_norm.weight` | 512 | bfloat16 | 23.16 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 1246 | `model.language_model.layers.34.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 85.8 | -2.175e-05 | 0.03422 | -0.3379 | 0.3281 | 0.004883, -0.00238, 0.001144, -0.003296, -0.00177, 0.004333, 0.0001035, 0.003326 |
| 1247 | `model.language_model.layers.34.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 23.33 | 1.432e-05 | 0.02631 | -0.05981 | 0.05981 | -0.001122, -0.00412, -0.02124, 0.02783, -0.04419, -0.0177, -0.001884, -0.04956 |
| 1248 | `model.language_model.layers.4.input_layernorm.weight` | 1536 | bfloat16 | 1576 | 18.64 | 35.66 | -0.7305 | 236 | 9.562, 0.1426, 0.3379, 14.5, 33, 0.3594, 63, 21 |
| 1249 | `model.language_model.layers.4.layer_scalar` | 1 | bfloat16 | 0.498 | 0.498 | 0 | 0.498 | 0.498 | 0.498 |
| 1250 | `model.language_model.layers.4.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 87.09 | 1.025e-05 | 0.02837 | -0.5781 | 0.5 | -0.01453, -0.01044, -0.001381, 0.007111, -0.0249, 0.04395, 0.007385, -0.06689 |
| 1251 | `model.language_model.layers.4.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 116.7 | -8.104e-05 | 0.03802 | -0.375 | 0.3926 | -0.02234, -0.02747, -0.06885, 0.04883, -0.02966, 0.0625, -0.003738, 0.05615 |
| 1252 | `model.language_model.layers.4.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 120.9 | -5.521e-06 | 0.03937 | -0.4629 | 0.4277 | -0.008606, -0.06836, 0.01501, 0.08398, -0.02222, -0.04565, -0.1011, -0.03125 |
| 1253 | `model.language_model.layers.4.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 12.92 | -7.656e-05 | 0.0206 | -0.1455 | 0.1582 | 0.002487, -0.01831, 0.008911, 0.002914, 0.002075, -0.008545, 0.001633, 0.06689 |
| 1254 | `model.language_model.layers.4.per_layer_projection.weight` | 1536×256 | bfloat16 | 40.72 | -8.095e-05 | 0.06494 | -1.07 | 0.9922 | 0.003403, -0.02234, -0.1377, -0.02747, 0.09131, -0.126, 0.01709, 0.2217 |
| 1255 | `model.language_model.layers.4.post_attention_layernorm.weight` | 1536 | bfloat16 | 64.4 | 1.256 | 1.06 | 0.001892 | 2.312 | 0.1377, 2.312, 2.312, 0.04443, 0.01453, 2.312, 0.01227, 0.02954 |
| 1256 | `model.language_model.layers.4.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 67.21 | 1.375 | 1.025 | 0.002136 | 2.359 | 2.359, 2.359, 2.359, 0.09912, 0.03064, 2.359, 0.05981, 0.04541 |
| 1257 | `model.language_model.layers.4.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 28.84 | 0.5056 | 0.5349 | 0.0008163 | 1.109 | 1.109, 1.109, 1.109, 0.2617, 0.08203, 1.109, 0.02246, 0.001022 |
| 1258 | `model.language_model.layers.4.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 5016 | 66.78 | 109.2 | -1.398 | 462 | 42.75, 0.9922, 1.672, 82, 213, 1.531, 286, 114 |
| 1259 | `model.language_model.layers.4.self_attn.k_norm.weight` | 512 | bfloat16 | 1.469 | 0.06494 | 0 | 0.06494 | 0.06494 | 0.06494, 0.06494, 0.06494, 0.06494, 0.06494, 0.06494, 0.06494, 0.06494 |
| 1260 | `model.language_model.layers.4.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 30.37 | 1.281e-05 | 0.03424 | -0.2617 | 0.2695 | 0.02039, -0.04736, 0.001785, -0.01556, 0.0293, -0.0116, -0.01251, 0.03638 |
| 1261 | `model.language_model.layers.4.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 73.38 | 2.757e-07 | 0.02927 | -0.6797 | 0.5781 | 0.002747, 0.0006943, 0.007751, -0.007629, -0.04346, 0.01721, 0.008179, -0.008301 |
| 1262 | `model.language_model.layers.4.self_attn.q_norm.weight` | 512 | bfloat16 | 21.83 | 0.9648 | 0 | 0.9648 | 0.9648 | 0.9648, 0.9648, 0.9648, 0.9648, 0.9648, 0.9648, 0.9648, 0.9648 |
| 1263 | `model.language_model.layers.4.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 87.6 | -1.032e-05 | 0.03493 | -0.4883 | 0.4629 | -0.01147, 0.006989, -0.03516, -0.003433, 0.006042, 0.03088, -0.005676, 0.005463 |
| 1264 | `model.language_model.layers.4.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 32.21 | -2.168e-05 | 0.03632 | -0.3496 | 0.3359 | -0.01416, 0.02039, -0.01318, -0.04492, -0.006958, -0.02405, 0.04468, 0.07617 |
| 1265 | `model.language_model.layers.5.input_layernorm.weight` | 1536 | bfloat16 | 2334 | 30.15 | 51.37 | -10.81 | 236 | 3.078, -0.8555, 1.188, 28.25, 77.5, 1.711, 157, 84 |
| 1266 | `model.language_model.layers.5.layer_scalar` | 1 | bfloat16 | 0.6367 | 0.6367 | 0 | 0.6367 | 0.6367 | 0.6367 |
| 1267 | `model.language_model.layers.5.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 85.87 | 8.87e-07 | 0.02797 | -0.6484 | 0.457 | 0.103, 0.00824, 0.02844, -0.0376, -0.02393, 0.02246, -0.06445, -0.04468 |
| 1268 | `model.language_model.layers.5.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 109.2 | 5.416e-06 | 0.03559 | -0.4121 | 0.418 | 0.06885, 0.0376, 0.06055, -0.02332, -0.003265, 0.04517, -0.04028, -0.0282 |
| 1269 | `model.language_model.layers.5.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 111.2 | 1.738e-06 | 0.03622 | -0.4648 | 0.4023 | 0.1108, -0.04175, -0.02966, 0.005493, -0.003021, -0.01624, -0.02161, -0.0376 |
| 1270 | `model.language_model.layers.5.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 16.83 | 2.539e-05 | 0.02684 | -0.2139 | 0.2119 | 0.004089, -0.06348, 0.01532, 0.05737, 0.06982, 0.005981, -0.04785, 0.06982 |
| 1271 | `model.language_model.layers.5.per_layer_projection.weight` | 1536×256 | bfloat16 | 44.79 | 8.684e-05 | 0.07142 | -0.8945 | 0.9883 | 0.06885, 0.01337, 0.06592, 0.103, 0.002655, 0.0791, -0.01904, -0.08203 |
| 1272 | `model.language_model.layers.5.post_attention_layernorm.weight` | 1536 | bfloat16 | 18.58 | 0.3899 | 0.2698 | 0.0004501 | 0.6328 | 0.6289, 0.6328, 0.6328, 0.03076, 0.008667, 0.6328, 0.05493, 0.01227 |
| 1273 | `model.language_model.layers.5.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 60.51 | 1.332 | 0.7817 | 0.001793 | 2.062 | 2.062, 2.062, 2.062, 0.4082, 0.4023, 2.062, 0.7578, 0.04565 |
| 1274 | `model.language_model.layers.5.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 30.57 | 0.6001 | 0.4984 | 0.0008316 | 1.062 | 1.062, 1.062, 1.062, 1.062, 1.062, 1.062, 1.062, 0.0008736 |
| 1275 | `model.language_model.layers.5.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 976.6 | 11.87 | 21.92 | -0.8789 | 183 | 0.9375, 0.498, 0.8203, 20.25, 63.75, 0.7188, 24.12, 46.25 |
| 1276 | `model.language_model.layers.5.self_attn.k_norm.weight` | 256 | bfloat16 | 2.031 | 0.127 | 0 | 0.127 | 0.127 | 0.127, 0.127, 0.127, 0.127, 0.127, 0.127, 0.127, 0.127 |
| 1277 | `model.language_model.layers.5.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 22.81 | 8.928e-07 | 0.03638 | -0.3379 | 0.3418 | 0.006989, 0.03491, 0.03198, -0.02148, -0.02881, -0.02649, 0.01794, -0.0166 |
| 1278 | `model.language_model.layers.5.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 62.49 | 1.476e-05 | 0.03524 | -0.4043 | 0.457 | -0.04834, -0.003601, 0.02246, -0.01306, 0.03687, 0.0155, -0.002167, -0.07373 |
| 1279 | `model.language_model.layers.5.self_attn.q_norm.weight` | 256 | bfloat16 | 15.69 | 0.9805 | 0 | 0.9805 | 0.9805 | 0.9805, 0.9805, 0.9805, 0.9805, 0.9805, 0.9805, 0.9805, 0.9805 |
| 1280 | `model.language_model.layers.5.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 63.24 | -2.001e-05 | 0.03566 | -0.4863 | 0.4883 | 0.002853, -0.04883, -7.808e-06, 0.02222, 0.01941, 0.03418, -0.01794, 0.004486 |
| 1281 | `model.language_model.layers.5.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 22.22 | 8.788e-05 | 0.03543 | -0.4375 | 0.3926 | -0.0354, -0.02991, 0.01385, -0.01367, 0.01758, 0.01282, 0.01904, -0.04639 |
| 1282 | `model.language_model.layers.6.input_layernorm.weight` | 1536 | bfloat16 | 2158 | 22.08 | 50.46 | -9.812 | 504 | 3.266, 1.57, 3.016, 19.25, 22.88, 3.734, 9, 146 |
| 1283 | `model.language_model.layers.6.layer_scalar` | 1 | bfloat16 | 0.498 | 0.498 | 0 | 0.498 | 0.498 | 0.498 |
| 1284 | `model.language_model.layers.6.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 83.67 | -1.057e-05 | 0.02726 | -0.6602 | 0.5273 | -0.01685, 0.01294, -0.04663, 0.04224, 0.0003262, -0.02771, 0.05762, -0.02527 |
| 1285 | `model.language_model.layers.6.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 114.3 | -8.396e-06 | 0.03725 | -0.543 | 0.5742 | 0.07129, -0.05713, 0.03687, 0.004028, -0.02283, -0.04346, -0.03442, -0.003754 |
| 1286 | `model.language_model.layers.6.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 114.1 | 2.13e-05 | 0.03718 | -0.5039 | 0.4941 | 0.01562, -0.02112, 0.05908, 0.01422, 0.05957, 0.003021, 0.052, -0.03564 |
| 1287 | `model.language_model.layers.6.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 12.15 | -5.347e-05 | 0.01937 | -0.2344 | 0.1924 | -0.01392, 0.03467, 0.009094, -0.003723, 0.02417, 0.00412, 0.006226, 0.007996 |
| 1288 | `model.language_model.layers.6.per_layer_projection.weight` | 1536×256 | bfloat16 | 45.65 | -0.0001028 | 0.0728 | -0.9023 | 0.8594 | -0.01019, -0.04395, 0.1396, 0.05078, -0.1001, 0.05835, 0.02979, -0.00647 |
| 1289 | `model.language_model.layers.6.post_attention_layernorm.weight` | 1536 | bfloat16 | 41.18 | 0.9058 | 0.5327 | 0.001076 | 1.406 | 1.336, 1.406, 1.406, 0.2969, 0.377, 1.406, 0.4941, 0.0293 |
| 1290 | `model.language_model.layers.6.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 76.06 | 1.733 | 0.8741 | 0.002365 | 2.656 | 2.656, 2.656, 2.656, 0.8594, 1.102, 2.656, 1.398, 0.07666 |
| 1291 | `model.language_model.layers.6.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 52.15 | 1.081 | 0.7765 | 0.001221 | 1.758 | 1.758, 1.758, 1.758, 1.758, 1.758, 1.758, 0.7227, 0.02881 |
| 1292 | `model.language_model.layers.6.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 631.2 | 5.641 | 15.09 | -0.5273 | 149 | 0.6758, 0.375, 0.6875, 3.438, 3.281, 0.6211, 2.234, 34 |
| 1293 | `model.language_model.layers.6.self_attn.k_norm.weight` | 256 | bfloat16 | 1.953 | 0.1221 | 0 | 0.1221 | 0.1221 | 0.1221, 0.1221, 0.1221, 0.1221, 0.1221, 0.1221, 0.1221, 0.1221 |
| 1294 | `model.language_model.layers.6.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 22.72 | -2.568e-05 | 0.03623 | -0.2412 | 0.293 | -0.0166, -0.06934, -0.001823, -0.000576, 0.01917, -0.0166, -0.02515, 0.00415 |
| 1295 | `model.language_model.layers.6.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 59.39 | -1.364e-05 | 0.03348 | -0.5781 | 0.4688 | 0.03638, -0.04614, -0.02173, 0.03027, 0.0188, 0.01111, 0.003937, -0.05859 |
| 1296 | `model.language_model.layers.6.self_attn.q_norm.weight` | 256 | bfloat16 | 16.38 | 1.023 | 0 | 1.023 | 1.023 | 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023, 1.023 |
| 1297 | `model.language_model.layers.6.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 64.43 | 1.893e-05 | 0.03633 | -0.5234 | 0.5586 | 0.0005188, 0.02856, 0.02698, 0.01434, -0.021, -0.006042, -0.02637, 0.03564 |
| 1298 | `model.language_model.layers.6.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 22.52 | 4.792e-06 | 0.03591 | -0.2832 | 0.2598 | 0.01697, -0.005157, 0.0376, -0.001877, -0.06689, 0.003937, -0.01019, -0.09668 |
| 1299 | `model.language_model.layers.7.input_layernorm.weight` | 1536 | bfloat16 | 2009 | 19.04 | 47.6 | -10 | 484 | 4.375, 1.633, 3.016, 10.69, 8.25, 3.703, 6.188, 145 |
| 1300 | `model.language_model.layers.7.layer_scalar` | 1 | bfloat16 | 0.6094 | 0.6094 | 0 | 0.6094 | 0.6094 | 0.6094 |
| 1301 | `model.language_model.layers.7.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 79.89 | 3.055e-06 | 0.02603 | -0.668 | 0.5078 | 0.007477, 0.06836, 0.002304, -0.003372, 0.02014, 0.02332, -0.003632, 0.0001287 |
| 1302 | `model.language_model.layers.7.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 110.5 | -0.0001247 | 0.03599 | -0.6992 | 0.6055 | -0.01562, 0.05298, 0.005066, 0.01215, -0.004242, 0.006165, 0.06689, 0.005676 |
| 1303 | `model.language_model.layers.7.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 111.1 | -2.684e-06 | 0.0362 | -0.4863 | 0.6133 | 0.02759, -0.03064, 0.05908, 0.05713, -0.03174, -0.005707, -0.02954, -0.01648 |
| 1304 | `model.language_model.layers.7.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 12.92 | 4.81e-05 | 0.02061 | -0.2812 | 0.2988 | 0.004181, -0.04688, 0.02148, -0.002029, 0.0002193, 0.005219, 0.004883, -0.07764 |
| 1305 | `model.language_model.layers.7.per_layer_projection.weight` | 1536×256 | bfloat16 | 45.8 | -7.349e-05 | 0.07304 | -0.7344 | 0.7305 | -0.0007782, -0.05273, -0.1318, 0.2188, -0.007111, 0.03003, 0.07373, -0.07178 |
| 1306 | `model.language_model.layers.7.post_attention_layernorm.weight` | 1536 | bfloat16 | 26.36 | 0.6232 | 0.2533 | 0.0006409 | 0.8086 | 0.7891, 0.8086, 0.8086, 0.2178, 0.4102, 0.8086, 0.4824, 0.03467 |
| 1307 | `model.language_model.layers.7.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 74.24 | 1.78 | 0.6472 | 0.001808 | 2.234 | 2.203, 2.234, 2.234, 0.75, 1.727, 2.234, 2.234, 0.1621 |
| 1308 | `model.language_model.layers.7.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 32.68 | 0.6953 | 0.4604 | 0.0006943 | 1.055 | 1.055, 1.055, 1.055, 1.055, 0.7031, 1.055, 1.055, 0.000946 |
| 1309 | `model.language_model.layers.7.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 441.8 | 2.676 | 10.95 | 0.002121 | 152 | 0.3711, 0.293, 0.4414, 1.492, 1.203, 0.3926, 0.9258, 14.81 |
| 1310 | `model.language_model.layers.7.self_attn.k_norm.weight` | 256 | bfloat16 | 1.984 | 0.124 | 0 | 0.124 | 0.124 | 0.124, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124 |
| 1311 | `model.language_model.layers.7.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 24.35 | 0.0001478 | 0.03883 | -0.4043 | 0.3809 | -0.00589, -0.08496, 0.01697, -0.03613, -0.05298, -0.03076, 0.05591, -0.003296 |
| 1312 | `model.language_model.layers.7.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 61.05 | -2.196e-05 | 0.03442 | -0.6641 | 0.5625 | -0.02576, -0.01953, 0.04224, 0.004242, -0.01135, -0.03467, -0.007233, 0.02954 |
| 1313 | `model.language_model.layers.7.self_attn.q_norm.weight` | 256 | bfloat16 | 16.12 | 1.008 | 0 | 1.008 | 1.008 | 1.008, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008 |
| 1314 | `model.language_model.layers.7.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 65.9 | 4.874e-05 | 0.03716 | -0.418 | 0.4355 | -0.02246, -0.05542, 0.03467, 0.04053, 0.002472, 0.04565, -0.01306, 0.004547 |
| 1315 | `model.language_model.layers.7.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 22.92 | 0.000122 | 0.03655 | -0.4375 | 0.4102 | -0.05762, 0.02136, -0.0141, 0.00209, -0.08008, -0.04077, 0.04224, 0.009033 |
| 1316 | `model.language_model.layers.8.input_layernorm.weight` | 1536 | bfloat16 | 2282 | 9.581 | 57.45 | -2.609 | 800 | 1, 0.7344, 0.875, 3.141, 1.844, 1, 1.469, 38.75 |
| 1317 | `model.language_model.layers.8.layer_scalar` | 1 | bfloat16 | 0.377 | 0.377 | 0 | 0.377 | 0.377 | 0.377 |
| 1318 | `model.language_model.layers.8.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 76.38 | 8.566e-06 | 0.02488 | -0.5352 | 0.4707 | -0.008728, 0.1001, 0.02295, -0.0177, 0.04614, -0.01239, 0.002777, 0.008484 |
| 1319 | `model.language_model.layers.8.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 109.3 | -9.997e-05 | 0.03561 | -0.5664 | 0.7852 | -0.06494, 0.0282, -0.01697, 0.05127, -0.05811, 0.02185, -0.01746, -0.02417 |
| 1320 | `model.language_model.layers.8.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 108.4 | -1.039e-05 | 0.03531 | -0.4883 | 0.4863 | -0.03149, -0.01599, -0.02112, -0.01117, -0.02295, -0.02917, -0.0007477, -0.02356 |
| 1321 | `model.language_model.layers.8.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 10.35 | -0.00013 | 0.01651 | -0.2197 | 0.3359 | 0.0008812, 0.01239, -0.004852, 0.01965, 0.01257, 0.00412, -0.0008316, -0.01746 |
| 1322 | `model.language_model.layers.8.per_layer_projection.weight` | 1536×256 | bfloat16 | 38.86 | 8e-05 | 0.06197 | -0.8594 | 0.8633 | 0.08203, 0.02417, 0.1582, 0.1094, 0.00473, -0.02002, -0.026, -0.0006676 |
| 1323 | `model.language_model.layers.8.post_attention_layernorm.weight` | 1536 | bfloat16 | 35.96 | 0.8752 | 0.2758 | 0.0009003 | 1.016 | 0.7891, 1.016, 1.016, 0.2168, 1.016, 0.7695, 1.016, 1.016 |
| 1324 | `model.language_model.layers.8.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 106 | 2.547 | 0.9102 | 0.002502 | 3.094 | 1.984, 3.094, 3.094, 0.8633, 3.094, 2.422, 3.094, 0.4219 |
| 1325 | `model.language_model.layers.8.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 117.7 | 2.559 | 1.57 | 0.00264 | 3.688 | 3.688, 3.688, 3.688, 3.688, 3.531, 3.688, 3.688, 0.00322 |
| 1326 | `model.language_model.layers.8.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 646.5 | 2.158 | 16.36 | 0.00174 | 262 | 0.3125, 0.2812, 0.2988, 1.141, 0.4785, 0.334, 0.4121, 0.5273 |
| 1327 | `model.language_model.layers.8.self_attn.k_norm.weight` | 256 | bfloat16 | 2.109 | 0.1318 | 0 | 0.1318 | 0.1318 | 0.1318, 0.1318, 0.1318, 0.1318, 0.1318, 0.1318, 0.1318, 0.1318 |
| 1328 | `model.language_model.layers.8.self_attn.k_proj.weight` | 256×1536 | bfloat16 | 22.86 | 4.639e-05 | 0.03645 | -0.3672 | 0.3438 | 0.002975, 0.00351, -0.02014, 0.0249, 0.01636, 0.04126, -0.02356, -0.0007629 |
| 1329 | `model.language_model.layers.8.self_attn.o_proj.weight` | 1536×2048 | bfloat16 | 57.77 | -3.548e-06 | 0.03257 | -0.4023 | 0.5078 | 0.01361, 0.03809, 0.07227, 0.1235, -0.03564, -0.01855, 0.06934, 0.1138 |
| 1330 | `model.language_model.layers.8.self_attn.q_norm.weight` | 256 | bfloat16 | 15.19 | 0.9492 | 0 | 0.9492 | 0.9492 | 0.9492, 0.9492, 0.9492, 0.9492, 0.9492, 0.9492, 0.9492, 0.9492 |
| 1331 | `model.language_model.layers.8.self_attn.q_proj.weight` | 2048×1536 | bfloat16 | 64.84 | -4.664e-06 | 0.03656 | -0.4883 | 0.4258 | 0.00824, 0.0009193, 0.00132, -0.00325, -0.007324, 0.006104, -0.01953, 0.01843 |
| 1332 | `model.language_model.layers.8.self_attn.v_proj.weight` | 256×1536 | bfloat16 | 22.15 | 7.38e-05 | 0.03533 | -0.6016 | 0.4902 | -0.01135, -0.0625, 0.02478, -0.02991, -0.04468, -0.001663, -0.03442, 0.03589 |
| 1333 | `model.language_model.layers.9.input_layernorm.weight` | 1536 | bfloat16 | 1916 | 5.725 | 48.56 | -1.156 | 720 | 0.75, 0.3457, 0.3887, 1.445, 0.5508, 0.5859, 0.4434, 2.656 |
| 1334 | `model.language_model.layers.9.layer_scalar` | 1 | bfloat16 | 0.4648 | 0.4648 | 0 | 0.4648 | 0.4648 | 0.4648 |
| 1335 | `model.language_model.layers.9.mlp.down_proj.weight` | 1536×6144 | bfloat16 | 77.85 | 5.191e-06 | 0.02536 | -0.4473 | 0.4082 | -0.04297, 0.0047, -0.009277, 0.02087, -0.03467, 0.00351, -0.03418, 0.03125 |
| 1336 | `model.language_model.layers.9.mlp.gate_proj.weight` | 6144×1536 | bfloat16 | 105.4 | -4.063e-05 | 0.03436 | -0.8281 | 0.6328 | -0.0119, -0.00705, -0.009827, 0.08496, -0.06006, -0.06885, -0.02563, 0.02747 |
| 1337 | `model.language_model.layers.9.mlp.up_proj.weight` | 6144×1536 | bfloat16 | 105.1 | -6.28e-07 | 0.03425 | -0.543 | 0.5625 | -0.01465, -0.0332, -0.06177, 0.02881, -0.01453, 0.04395, -0.02698, 0.008423 |
| 1338 | `model.language_model.layers.9.per_layer_input_gate.weight` | 256×1536 | bfloat16 | 10.53 | -8.934e-05 | 0.01679 | -0.3047 | 0.3379 | 0.00354, -0.01758, 0.01239, -0.02307, -0.0007858, -0.01154, 0.007111, 0.000885 |
| 1339 | `model.language_model.layers.9.per_layer_projection.weight` | 1536×256 | bfloat16 | 45.37 | -0.0002008 | 0.07236 | -0.8555 | 1.008 | 0.08594, -0.009216, 0.09277, -0.08887, -0.007721, -0.03101, -0.05103, 0.2148 |
| 1340 | `model.language_model.layers.9.post_attention_layernorm.weight` | 1536 | bfloat16 | 49.79 | 1.207 | 0.3968 | 0.00116 | 1.414 | 1.414, 1.414, 1.188, 0.2617, 1.414, 1.414, 1.414, 1.414 |
| 1341 | `model.language_model.layers.9.post_feedforward_layernorm.weight` | 1536 | bfloat16 | 107.7 | 2.523 | 1.089 | 0.002625 | 3.297 | 0.9609, 3.297, 3.297, 0.4941, 3.297, 1.32, 3.297, 3.297 |
| 1342 | `model.language_model.layers.9.post_per_layer_input_norm.weight` | 1536 | bfloat16 | 43.3 | 1.024 | 0.4141 | 0.001038 | 1.273 | 0.2988, 1.273, 1.273, 0.6367, 1.273, 1.273, 1.273, 1.273 |
| 1343 | `model.language_model.layers.9.pre_feedforward_layernorm.weight` | 1536 | bfloat16 | 579.7 | 1.926 | 14.67 | -0.001305 | 372 | 0.377, 0.3281, 0.332, 1.391, 0.3535, 0.4473, 0.3262, 0.3594 |
| 1344 | `model.language_model.layers.9.self_attn.k_norm.weight` | 512 | bfloat16 | 1.359 | 0.06006 | 0 | 0.06006 | 0.06006 | 0.06006, 0.06006, 0.06006, 0.06006, 0.06006, 0.06006, 0.06006, 0.06006 |
| 1345 | `model.language_model.layers.9.self_attn.k_proj.weight` | 512×1536 | bfloat16 | 28.56 | -3.165e-05 | 0.03221 | -0.2832 | 0.3086 | -0.001015, 0.0108, 0.003204, -0.01196, -0.003967, -0.002426, 0.02307, -0.001938 |
| 1346 | `model.language_model.layers.9.self_attn.o_proj.weight` | 1536×4096 | bfloat16 | 69.96 | 6.627e-07 | 0.0279 | -0.4492 | 0.4512 | 0.03027, -0.04077, 0.01611, 0.03711, -0.001663, -0.02942, -0.02649, -0.003677 |
| 1347 | `model.language_model.layers.9.self_attn.q_norm.weight` | 512 | bfloat16 | 23.51 | 1.039 | 0 | 1.039 | 1.039 | 1.039, 1.039, 1.039, 1.039, 1.039, 1.039, 1.039, 1.039 |
| 1348 | `model.language_model.layers.9.self_attn.q_proj.weight` | 4096×1536 | bfloat16 | 85.3 | 6.293e-06 | 0.03402 | -0.4551 | 0.5156 | -0.001648, 0.009033, -0.0003052, 0.003876, -0.005463, -0.003754, -0.0004406, 0.004761 |
| 1349 | `model.language_model.layers.9.self_attn.v_proj.weight` | 512×1536 | bfloat16 | 28.52 | 7.423e-06 | 0.03215 | -0.3398 | 0.3574 | -0.1357, -0.02869, -0.01685, 0.008972, 0.02515, 0.02515, 0.03882, 0.01337 |
| 1350 | `model.language_model.norm.weight` | 1536 | bfloat16 | 586.7 | 14.17 | 4.838 | -0.00206 | 118.5 | 13.38, 8.625, 14.25, 16.12, 14.88, 13.12, 13.31, 14.19 |
| 1351 | `model.language_model.per_layer_model_projection.weight` | 8960×1536 | bfloat16 | 127.1 | 3.186e-06 | 0.0343 | -0.8164 | 0.4941 | 0.04468, 0.03223, -0.0127, 0.0603, 0.006073, 0.02551, -0.02673, -0.08496 |
| 1352 | `model.language_model.per_layer_projection_norm.weight` | 256 | bfloat16 | 15.91 | 0.765 | 0.6364 | -0.08105 | 5.531 | 0.09424, 0.7969, 0.6602, 0.168, 4.906, 0.1377, 1.125, 0.7734 |
| 1353 | `model.vision_tower.encoder.layers.0.input_layernorm.weight` | 768 | bfloat16 | 149.5 | 2.97 | 4.505 | -2.734 | 21.75 | 1.242, 1.07, 13.38, 0.05371, 0.09131, 13.5, 0.8203, 13.44 |
| 1354 | `model.vision_tower.encoder.layers.0.mlp.down_proj.input_max` |  | bfloat16 | 12.19 | 12.19 | 0 | 12.19 | 12.19 | 12.19 |
| 1355 | `model.vision_tower.encoder.layers.0.mlp.down_proj.input_min` |  | bfloat16 | 12.25 | -12.25 | 0 | -12.25 | -12.25 | -12.25 |
| 1356 | `model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | -2.582e-06 | 0.01804 | -0.1904 | 0.1885 | -0.00531, 0.002121, 0.002121, -0.008484, 0.002121, -0.03491, 0.01697, -0.01587 |
| 1357 | `model.vision_tower.encoder.layers.0.mlp.down_proj.output_max` |  | bfloat16 | 10.25 | 10.25 | 0 | 10.25 | 10.25 | 10.25 |
| 1358 | `model.vision_tower.encoder.layers.0.mlp.down_proj.output_min` |  | bfloat16 | 10.38 | -10.38 | 0 | -10.38 | -10.38 | -10.38 |
| 1359 | `model.vision_tower.encoder.layers.0.mlp.gate_proj.input_max` |  | bfloat16 | 2.438 | 2.438 | 0 | 2.438 | 2.438 | 2.438 |
| 1360 | `model.vision_tower.encoder.layers.0.mlp.gate_proj.input_min` |  | bfloat16 | 2.453 | -2.453 | 0 | -2.453 | -2.453 | -2.453 |
| 1361 | `model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.41 | 4.757e-06 | 0.03608 | -0.4258 | 0.3281 | -0.007751, 0.01355, -0.0155, -0.01746, 0.01355, -0.04639, -0.1045, -0.009705 |
| 1362 | `model.vision_tower.encoder.layers.0.mlp.gate_proj.output_max` |  | bfloat16 | 4.469 | 4.469 | 0 | 4.469 | 4.469 | 4.469 |
| 1363 | `model.vision_tower.encoder.layers.0.mlp.gate_proj.output_min` |  | bfloat16 | 4.5 | -4.5 | 0 | -4.5 | -4.5 | -4.5 |
| 1364 | `model.vision_tower.encoder.layers.0.mlp.up_proj.input_max` |  | bfloat16 | 2.438 | 2.438 | 0 | 2.438 | 2.438 | 2.438 |
| 1365 | `model.vision_tower.encoder.layers.0.mlp.up_proj.input_min` |  | bfloat16 | 2.453 | -2.453 | 0 | -2.453 | -2.453 | -2.453 |
| 1366 | `model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 4.523e-06 | 0.03609 | -0.4023 | 0.3359 | 0.01636, -0.01086, 0.001816, 0.001816, 0.03442, 0.005432, 0.03979, -0.0127 |
| 1367 | `model.vision_tower.encoder.layers.0.mlp.up_proj.output_max` |  | bfloat16 | 4.469 | 4.469 | 0 | 4.469 | 4.469 | 4.469 |
| 1368 | `model.vision_tower.encoder.layers.0.mlp.up_proj.output_min` |  | bfloat16 | 4.5 | -4.5 | 0 | -4.5 | -4.5 | -4.5 |
| 1369 | `model.vision_tower.encoder.layers.0.post_attention_layernorm.weight` | 768 | bfloat16 | 21.43 | 0.3262 | 0.7015 | -0.5117 | 11.75 | 5.969, 0.1089, 0.3574, -0.05615, -0.04834, 0.3613, -0.06177, 0.3281 |
| 1370 | `model.vision_tower.encoder.layers.0.post_feedforward_layernorm.weight` | 768 | bfloat16 | 46.8 | 0.9859 | 1.372 | -0.7109 | 17.62 | 2.672, 3.156, 3.062, 0.0752, 1.617, 0.4707, 0.3223, 0.4434 |
| 1371 | `model.vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 113.3 | 1.908 | 3.619 | -0.5312 | 34.25 | 0.0166, 0.459, 8.188, 0.1016, 0.3496, 5.625, 2.422, 6.812 |
| 1372 | `model.vision_tower.encoder.layers.0.self_attn.k_norm.weight` | 64 | bfloat16 | 9.875 | 1.234 | 0 | 1.234 | 1.234 | 1.234, 1.234, 1.234, 1.234, 1.234, 1.234, 1.234, 1.234 |
| 1373 | `model.vision_tower.encoder.layers.0.self_attn.k_proj.input_max` |  | bfloat16 | 6.312 | 6.312 | 0 | 6.312 | 6.312 | 6.312 |
| 1374 | `model.vision_tower.encoder.layers.0.self_attn.k_proj.input_min` |  | bfloat16 | 6.375 | -6.375 | 0 | -6.375 | -6.375 | -6.375 |
| 1375 | `model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 1.229e-05 | 0.03608 | -0.459 | 0.4434 | 0.004425, -0.01105, 0.004425, 0, -0.01105, 0.02209, 0.002213, -0.002213 |
| 1376 | `model.vision_tower.encoder.layers.0.self_attn.k_proj.output_max` |  | bfloat16 | 10.06 | 10.06 | 0 | 10.06 | 10.06 | 10.06 |
| 1377 | `model.vision_tower.encoder.layers.0.self_attn.k_proj.output_min` |  | bfloat16 | 10.12 | -10.12 | 0 | -10.12 | -10.12 | -10.12 |
| 1378 | `model.vision_tower.encoder.layers.0.self_attn.o_proj.input_max` |  | bfloat16 | 3.484 | 3.484 | 0 | 3.484 | 3.484 | 3.484 |
| 1379 | `model.vision_tower.encoder.layers.0.self_attn.o_proj.input_min` |  | bfloat16 | 3.5 | -3.5 | 0 | -3.5 | -3.5 | -3.5 |
| 1380 | `model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 3.558e-05 | 0.03609 | -0.4727 | 0.4375 | -0.01404, 0.003998, 0.02209, -0.03198, 0.003998, 0.01001, -0.01202, 0.001999 |
| 1381 | `model.vision_tower.encoder.layers.0.self_attn.o_proj.output_max` |  | bfloat16 | 21.25 | 21.25 | 0 | 21.25 | 21.25 | 21.25 |
| 1382 | `model.vision_tower.encoder.layers.0.self_attn.o_proj.output_min` |  | bfloat16 | 21.38 | -21.38 | 0 | -21.38 | -21.38 | -21.38 |
| 1383 | `model.vision_tower.encoder.layers.0.self_attn.q_norm.weight` | 64 | bfloat16 | 3.25 | 0.4062 | 0 | 0.4062 | 0.4062 | 0.4062, 0.4062, 0.4062, 0.4062, 0.4062, 0.4062, 0.4062, 0.4062 |
| 1384 | `model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max` |  | bfloat16 | 6.312 | 6.312 | 0 | 6.312 | 6.312 | 6.312 |
| 1385 | `model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min` |  | bfloat16 | 6.375 | -6.375 | 0 | -6.375 | -6.375 | -6.375 |
| 1386 | `model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.7 | -4.482e-05 | 0.03607 | -0.6445 | 0.6445 | -0.01172, -0.005859, -0.005859, -0.01373, 0.0625, -0.01953, 0.04297, 0.01373 |
| 1387 | `model.vision_tower.encoder.layers.0.self_attn.q_proj.output_max` |  | bfloat16 | 11.19 | 11.19 | 0 | 11.19 | 11.19 | 11.19 |
| 1388 | `model.vision_tower.encoder.layers.0.self_attn.q_proj.output_min` |  | bfloat16 | 11.31 | -11.31 | 0 | -11.31 | -11.31 | -11.31 |
| 1389 | `model.vision_tower.encoder.layers.0.self_attn.v_proj.input_max` |  | bfloat16 | 6.312 | 6.312 | 0 | 6.312 | 6.312 | 6.312 |
| 1390 | `model.vision_tower.encoder.layers.0.self_attn.v_proj.input_min` |  | bfloat16 | 6.375 | -6.375 | 0 | -6.375 | -6.375 | -6.375 |
| 1391 | `model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.67 | -5.875e-05 | 0.03603 | -0.3906 | 0.3848 | -0.01465, 0, -0.01465, -0.03467, -0.05298, 0.05127, 0.02563, 0.06396 |
| 1392 | `model.vision_tower.encoder.layers.0.self_attn.v_proj.output_max` |  | bfloat16 | 10.06 | 10.06 | 0 | 10.06 | 10.06 | 10.06 |
| 1393 | `model.vision_tower.encoder.layers.0.self_attn.v_proj.output_min` |  | bfloat16 | 10.12 | -10.12 | 0 | -10.12 | -10.12 | -10.12 |
| 1394 | `model.vision_tower.encoder.layers.1.input_layernorm.weight` | 768 | bfloat16 | 120.8 | 2.813 | 3.332 | -3.703 | 33.5 | -0.04272, 1.586, -0.1367, -3.703, 1.633, 9.25, 3.031, 10.5 |
| 1395 | `model.vision_tower.encoder.layers.1.mlp.down_proj.input_max` |  | bfloat16 | 30 | 30 | 0 | 30 | 30 | 30 |
| 1396 | `model.vision_tower.encoder.layers.1.mlp.down_proj.input_min` |  | bfloat16 | 30.25 | -30.25 | 0 | -30.25 | -30.25 | -30.25 |
| 1397 | `model.vision_tower.encoder.layers.1.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 1.033e-05 | 0.01804 | -0.1895 | 0.2139 | -0.01147, -0.0271, 0.006256, -0.005219, 0.005219, -0.005219, 0.01868, 0.008301 |
| 1398 | `model.vision_tower.encoder.layers.1.mlp.down_proj.output_max` |  | bfloat16 | 18.25 | 18.25 | 0 | 18.25 | 18.25 | 18.25 |
| 1399 | `model.vision_tower.encoder.layers.1.mlp.down_proj.output_min` |  | bfloat16 | 18.38 | -18.38 | 0 | -18.38 | -18.38 | -18.38 |
| 1400 | `model.vision_tower.encoder.layers.1.mlp.gate_proj.input_max` |  | bfloat16 | 4.312 | 4.312 | 0 | 4.312 | 4.312 | 4.312 |
| 1401 | `model.vision_tower.encoder.layers.1.mlp.gate_proj.input_min` |  | bfloat16 | 4.344 | -4.344 | 0 | -4.344 | -4.344 | -4.344 |
| 1402 | `model.vision_tower.encoder.layers.1.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -9.342e-06 | 0.03608 | -0.3262 | 0.3223 | -0.03076, -0.01343, 0.01538, -0.0481, -0.01343, -0.001923, 0.003845, 0.04419 |
| 1403 | `model.vision_tower.encoder.layers.1.mlp.gate_proj.output_max` |  | bfloat16 | 5.969 | 5.969 | 0 | 5.969 | 5.969 | 5.969 |
| 1404 | `model.vision_tower.encoder.layers.1.mlp.gate_proj.output_min` |  | bfloat16 | 6.031 | -6.031 | 0 | -6.031 | -6.031 | -6.031 |
| 1405 | `model.vision_tower.encoder.layers.1.mlp.up_proj.input_max` |  | bfloat16 | 4.312 | 4.312 | 0 | 4.312 | 4.312 | 4.312 |
| 1406 | `model.vision_tower.encoder.layers.1.mlp.up_proj.input_min` |  | bfloat16 | 4.344 | -4.344 | 0 | -4.344 | -4.344 | -4.344 |
| 1407 | `model.vision_tower.encoder.layers.1.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -5.022e-06 | 0.03608 | -0.2891 | 0.3047 | -0.02271, -0.01855, -0.00824, 0.09473, 0.06787, -0.02063, -0.03906, 0.05371 |
| 1408 | `model.vision_tower.encoder.layers.1.mlp.up_proj.output_max` |  | bfloat16 | 5.969 | 5.969 | 0 | 5.969 | 5.969 | 5.969 |
| 1409 | `model.vision_tower.encoder.layers.1.mlp.up_proj.output_min` |  | bfloat16 | 6.031 | -6.031 | 0 | -6.031 | -6.031 | -6.031 |
| 1410 | `model.vision_tower.encoder.layers.1.post_attention_layernorm.weight` | 768 | bfloat16 | 17.06 | 0.2821 | 0.5473 | -1.375 | 8.562 | -0.006073, 0.6094, 0.04614, -0.1191, 0.01471, 0.2871, 0.2656, 0.2012 |
| 1411 | `model.vision_tower.encoder.layers.1.post_feedforward_layernorm.weight` | 768 | bfloat16 | 28.73 | 0.6966 | 0.7681 | -0.7227 | 9.812 | 1.539, 1.906, 1.789, 0.2441, 0.9883, 0.4727, 0.1099, 0.4961 |
| 1412 | `model.vision_tower.encoder.layers.1.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 123 | 2.913 | 3.351 | -1.648 | 14.75 | 0.01599, 1.625, 0.875, 1.148, 1.43, 10.69, 2.516, 12.75 |
| 1413 | `model.vision_tower.encoder.layers.1.self_attn.k_norm.weight` | 64 | bfloat16 | 11.25 | 1.406 | 0 | 1.406 | 1.406 | 1.406, 1.406, 1.406, 1.406, 1.406, 1.406, 1.406, 1.406 |
| 1414 | `model.vision_tower.encoder.layers.1.self_attn.k_proj.input_max` |  | bfloat16 | 5.531 | 5.531 | 0 | 5.531 | 5.531 | 5.531 |
| 1415 | `model.vision_tower.encoder.layers.1.self_attn.k_proj.input_min` |  | bfloat16 | 5.562 | -5.562 | 0 | -5.562 | -5.562 | -5.562 |
| 1416 | `model.vision_tower.encoder.layers.1.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -2.682e-05 | 0.03609 | -0.5508 | 0.5508 | -0.001938, -0.0155, 0.01361, 0.07959, 0.0835, -0.001938, -0.02917, -0.03101 |
| 1417 | `model.vision_tower.encoder.layers.1.self_attn.k_proj.output_max` |  | bfloat16 | 11.12 | 11.12 | 0 | 11.12 | 11.12 | 11.12 |
| 1418 | `model.vision_tower.encoder.layers.1.self_attn.k_proj.output_min` |  | bfloat16 | 11.19 | -11.19 | 0 | -11.19 | -11.19 | -11.19 |
| 1419 | `model.vision_tower.encoder.layers.1.self_attn.o_proj.input_max` |  | bfloat16 | 3.109 | 3.109 | 0 | 3.109 | 3.109 | 3.109 |
| 1420 | `model.vision_tower.encoder.layers.1.self_attn.o_proj.input_min` |  | bfloat16 | 3.141 | -3.141 | 0 | -3.141 | -3.141 | -3.141 |
| 1421 | `model.vision_tower.encoder.layers.1.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 4.407e-05 | 0.03609 | -0.3965 | 0.4102 | 0.07373, -0.1426, 0.09814, -0.25, 0.03931, -0.05884, -0.06396, -0.1177 |
| 1422 | `model.vision_tower.encoder.layers.1.self_attn.o_proj.output_max` |  | bfloat16 | 15.12 | 15.12 | 0 | 15.12 | 15.12 | 15.12 |
| 1423 | `model.vision_tower.encoder.layers.1.self_attn.o_proj.output_min` |  | bfloat16 | 15.25 | -15.25 | 0 | -15.25 | -15.25 | -15.25 |
| 1424 | `model.vision_tower.encoder.layers.1.self_attn.q_norm.weight` | 64 | bfloat16 | 2.844 | 0.3555 | 0 | 0.3555 | 0.3555 | 0.3555, 0.3555, 0.3555, 0.3555, 0.3555, 0.3555, 0.3555, 0.3555 |
| 1425 | `model.vision_tower.encoder.layers.1.self_attn.q_proj.input_max` |  | bfloat16 | 5.531 | 5.531 | 0 | 5.531 | 5.531 | 5.531 |
| 1426 | `model.vision_tower.encoder.layers.1.self_attn.q_proj.input_min` |  | bfloat16 | 5.562 | -5.562 | 0 | -5.562 | -5.562 | -5.562 |
| 1427 | `model.vision_tower.encoder.layers.1.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -6.484e-05 | 0.03609 | -0.5664 | 0.5664 | 0.02173, -0.008179, 0.05151, -0.05444, -0.0354, -0.002716, 0.008179, 0.002716 |
| 1428 | `model.vision_tower.encoder.layers.1.self_attn.q_proj.output_max` |  | bfloat16 | 12.38 | 12.38 | 0 | 12.38 | 12.38 | 12.38 |
| 1429 | `model.vision_tower.encoder.layers.1.self_attn.q_proj.output_min` |  | bfloat16 | 12.44 | -12.44 | 0 | -12.44 | -12.44 | -12.44 |
| 1430 | `model.vision_tower.encoder.layers.1.self_attn.v_proj.input_max` |  | bfloat16 | 5.531 | 5.531 | 0 | 5.531 | 5.531 | 5.531 |
| 1431 | `model.vision_tower.encoder.layers.1.self_attn.v_proj.input_min` |  | bfloat16 | 5.562 | -5.562 | 0 | -5.562 | -5.562 | -5.562 |
| 1432 | `model.vision_tower.encoder.layers.1.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 4.435e-05 | 0.03608 | -0.4941 | 0.457 | 0.004456, 0.01044, 0.005951, 0.01044, -0.01044, 0.03125, 0.04907, -0.0238 |
| 1433 | `model.vision_tower.encoder.layers.1.self_attn.v_proj.output_max` |  | bfloat16 | 11.12 | 11.12 | 0 | 11.12 | 11.12 | 11.12 |
| 1434 | `model.vision_tower.encoder.layers.1.self_attn.v_proj.output_min` |  | bfloat16 | 11.19 | -11.19 | 0 | -11.19 | -11.19 | -11.19 |
| 1435 | `model.vision_tower.encoder.layers.10.input_layernorm.weight` | 768 | bfloat16 | 139.7 | 4.602 | 2.058 | -0.1045 | 17.5 | 0.02759, 3.734, 6.406, 4.469, 6.094, 5.281, 7.719, 2.344 |
| 1436 | `model.vision_tower.encoder.layers.10.mlp.down_proj.input_max` |  | bfloat16 | 18.38 | 18.38 | 0 | 18.38 | 18.38 | 18.38 |
| 1437 | `model.vision_tower.encoder.layers.10.mlp.down_proj.input_min` |  | bfloat16 | 18.5 | -18.5 | 0 | -18.5 | -18.5 | -18.5 |
| 1438 | `model.vision_tower.encoder.layers.10.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 2.404e-05 | 0.01804 | -0.2188 | 0.2285 | -0.003723, -0.007446, 0.03174, -0.003723, -0.01489, -0.02417, 0.02612, 0.03345 |
| 1439 | `model.vision_tower.encoder.layers.10.mlp.down_proj.output_max` |  | bfloat16 | 5.562 | 5.562 | 0 | 5.562 | 5.562 | 5.562 |
| 1440 | `model.vision_tower.encoder.layers.10.mlp.down_proj.output_min` |  | bfloat16 | 5.594 | -5.594 | 0 | -5.594 | -5.594 | -5.594 |
| 1441 | `model.vision_tower.encoder.layers.10.mlp.gate_proj.input_max` |  | bfloat16 | 8.875 | 8.875 | 0 | 8.875 | 8.875 | 8.875 |
| 1442 | `model.vision_tower.encoder.layers.10.mlp.gate_proj.input_min` |  | bfloat16 | 8.938 | -8.938 | 0 | -8.938 | -8.938 | -8.938 |
| 1443 | `model.vision_tower.encoder.layers.10.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0003478 | 0.03608 | -0.3223 | 0.3184 | -0.04272, 0.05786, 0.03271, -0.02515, 0.007538, 0.09814, -0.005035, 0.05786 |
| 1444 | `model.vision_tower.encoder.layers.10.mlp.gate_proj.output_max` |  | bfloat16 | 9.688 | 9.688 | 0 | 9.688 | 9.688 | 9.688 |
| 1445 | `model.vision_tower.encoder.layers.10.mlp.gate_proj.output_min` |  | bfloat16 | 9.75 | -9.75 | 0 | -9.75 | -9.75 | -9.75 |
| 1446 | `model.vision_tower.encoder.layers.10.mlp.up_proj.input_max` |  | bfloat16 | 8.875 | 8.875 | 0 | 8.875 | 8.875 | 8.875 |
| 1447 | `model.vision_tower.encoder.layers.10.mlp.up_proj.input_min` |  | bfloat16 | 8.938 | -8.938 | 0 | -8.938 | -8.938 | -8.938 |
| 1448 | `model.vision_tower.encoder.layers.10.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 1.191e-05 | 0.03609 | -0.2852 | 0.2852 | -0.03223, 0.01611, -0.02625, 0.01611, 0.06445, -0.05469, 0.01013, 0.02026 |
| 1449 | `model.vision_tower.encoder.layers.10.mlp.up_proj.output_max` |  | bfloat16 | 9.688 | 9.688 | 0 | 9.688 | 9.688 | 9.688 |
| 1450 | `model.vision_tower.encoder.layers.10.mlp.up_proj.output_min` |  | bfloat16 | 9.75 | -9.75 | 0 | -9.75 | -9.75 | -9.75 |
| 1451 | `model.vision_tower.encoder.layers.10.post_attention_layernorm.weight` | 768 | bfloat16 | 60.84 | 1.894 | 1.111 | -4.125 | 14.06 | -0.1079, 1.25, 0.9688, 1.25, 1.641, 2.641, 0.8828, 1.328 |
| 1452 | `model.vision_tower.encoder.layers.10.post_feedforward_layernorm.weight` | 768 | bfloat16 | 102.1 | 3.302 | 1.637 | -1.133 | 19.38 | 0.2061, 3.078, 2.719, 2.859, 2.234, 3.141, 2.609, 2.891 |
| 1453 | `model.vision_tower.encoder.layers.10.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 101.3 | 3.366 | 1.422 | -0.1582 | 9.062 | 0.165, 3.422, 4.469, 3.406, 5.375, 3.766, 4.312, 1.453 |
| 1454 | `model.vision_tower.encoder.layers.10.self_attn.k_norm.weight` | 64 | bfloat16 | 10.62 | 1.328 | 0 | 1.328 | 1.328 | 1.328, 1.328, 1.328, 1.328, 1.328, 1.328, 1.328, 1.328 |
| 1455 | `model.vision_tower.encoder.layers.10.self_attn.k_proj.input_max` |  | bfloat16 | 13.62 | 13.62 | 0 | 13.62 | 13.62 | 13.62 |
| 1456 | `model.vision_tower.encoder.layers.10.self_attn.k_proj.input_min` |  | bfloat16 | 13.69 | -13.69 | 0 | -13.69 | -13.69 | -13.69 |
| 1457 | `model.vision_tower.encoder.layers.10.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | -2.053e-05 | 0.03608 | -0.4316 | 0.375 | -0.03369, 0.01123, -0.03857, -0.02246, -0.004822, 0.0144, 0.02563, 0.004822 |
| 1458 | `model.vision_tower.encoder.layers.10.self_attn.k_proj.output_max` |  | bfloat16 | 21 | 21 | 0 | 21 | 21 | 21 |
| 1459 | `model.vision_tower.encoder.layers.10.self_attn.k_proj.output_min` |  | bfloat16 | 21.12 | -21.12 | 0 | -21.12 | -21.12 | -21.12 |
| 1460 | `model.vision_tower.encoder.layers.10.self_attn.o_proj.input_max` |  | bfloat16 | 1.938 | 1.938 | 0 | 1.938 | 1.938 | 1.938 |
| 1461 | `model.vision_tower.encoder.layers.10.self_attn.o_proj.input_min` |  | bfloat16 | 1.945 | -1.945 | 0 | -1.945 | -1.945 | -1.945 |
| 1462 | `model.vision_tower.encoder.layers.10.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.506e-05 | 0.03609 | -0.2754 | 0.2432 | -0.03394, 0.07129, -0.0791, -0.05273, 0.04517, -0.003754, 0.1426, 0.01129 |
| 1463 | `model.vision_tower.encoder.layers.10.self_attn.o_proj.output_max` |  | bfloat16 | 2.359 | 2.359 | 0 | 2.359 | 2.359 | 2.359 |
| 1464 | `model.vision_tower.encoder.layers.10.self_attn.o_proj.output_min` |  | bfloat16 | 2.375 | -2.375 | 0 | -2.375 | -2.375 | -2.375 |
| 1465 | `model.vision_tower.encoder.layers.10.self_attn.q_norm.weight` | 64 | bfloat16 | 3.016 | 0.377 | 0 | 0.377 | 0.377 | 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377 |
| 1466 | `model.vision_tower.encoder.layers.10.self_attn.q_proj.input_max` |  | bfloat16 | 13.62 | 13.62 | 0 | 13.62 | 13.62 | 13.62 |
| 1467 | `model.vision_tower.encoder.layers.10.self_attn.q_proj.input_min` |  | bfloat16 | 13.69 | -13.69 | 0 | -13.69 | -13.69 | -13.69 |
| 1468 | `model.vision_tower.encoder.layers.10.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.869e-05 | 0.03609 | -0.3867 | 0.3379 | -0.02979, -0.01917, 0.03394, -0.006378, 0.01489, -0.01489, 0.01276, 0.04028 |
| 1469 | `model.vision_tower.encoder.layers.10.self_attn.q_proj.output_max` |  | bfloat16 | 15.19 | 15.19 | 0 | 15.19 | 15.19 | 15.19 |
| 1470 | `model.vision_tower.encoder.layers.10.self_attn.q_proj.output_min` |  | bfloat16 | 15.31 | -15.31 | 0 | -15.31 | -15.31 | -15.31 |
| 1471 | `model.vision_tower.encoder.layers.10.self_attn.v_proj.input_max` |  | bfloat16 | 13.62 | 13.62 | 0 | 13.62 | 13.62 | 13.62 |
| 1472 | `model.vision_tower.encoder.layers.10.self_attn.v_proj.input_min` |  | bfloat16 | 13.69 | -13.69 | 0 | -13.69 | -13.69 | -13.69 |
| 1473 | `model.vision_tower.encoder.layers.10.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.957e-05 | 0.03609 | -0.2695 | 0.293 | -0.04395, -0.02393, 0.0498, 0.0498, 0.007996, 0.05176, -0.009949, 0.02588 |
| 1474 | `model.vision_tower.encoder.layers.10.self_attn.v_proj.output_max` |  | bfloat16 | 21 | 21 | 0 | 21 | 21 | 21 |
| 1475 | `model.vision_tower.encoder.layers.10.self_attn.v_proj.output_min` |  | bfloat16 | 21.12 | -21.12 | 0 | -21.12 | -21.12 | -21.12 |
| 1476 | `model.vision_tower.encoder.layers.11.input_layernorm.weight` | 768 | bfloat16 | 119.1 | 3.89 | 1.828 | -1.289 | 9.875 | 0.03979, 3.562, 4.219, 3.5, 6.312, 5.062, 5.938, 2.438 |
| 1477 | `model.vision_tower.encoder.layers.11.mlp.down_proj.input_max` |  | bfloat16 | 9.375 | 9.375 | 0 | 9.375 | 9.375 | 9.375 |
| 1478 | `model.vision_tower.encoder.layers.11.mlp.down_proj.input_min` |  | bfloat16 | 9.438 | -9.438 | 0 | -9.438 | -9.438 | -9.438 |
| 1479 | `model.vision_tower.encoder.layers.11.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 9.915e-06 | 0.01804 | -0.2119 | 0.2383 | 0.005615, 0.003738, -0.02625, -0.04126, -0.01312, -0.06543, 0.03369, 0.07666 |
| 1480 | `model.vision_tower.encoder.layers.11.mlp.down_proj.output_max` |  | bfloat16 | 2.453 | 2.453 | 0 | 2.453 | 2.453 | 2.453 |
| 1481 | `model.vision_tower.encoder.layers.11.mlp.down_proj.output_min` |  | bfloat16 | 2.469 | -2.469 | 0 | -2.469 | -2.469 | -2.469 |
| 1482 | `model.vision_tower.encoder.layers.11.mlp.gate_proj.input_max` |  | bfloat16 | 6.031 | 6.031 | 0 | 6.031 | 6.031 | 6.031 |
| 1483 | `model.vision_tower.encoder.layers.11.mlp.gate_proj.input_min` |  | bfloat16 | 6.062 | -6.062 | 0 | -6.062 | -6.062 | -6.062 |
| 1484 | `model.vision_tower.encoder.layers.11.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0002743 | 0.03609 | -0.2832 | 0.291 | 0.03027, -0.06982, -0.0415, -0.02832, -0.03223, -0.03027, 0.03394, 0.00946 |
| 1485 | `model.vision_tower.encoder.layers.11.mlp.gate_proj.output_max` |  | bfloat16 | 5.969 | 5.969 | 0 | 5.969 | 5.969 | 5.969 |
| 1486 | `model.vision_tower.encoder.layers.11.mlp.gate_proj.output_min` |  | bfloat16 | 6 | -6 | 0 | -6 | -6 | -6 |
| 1487 | `model.vision_tower.encoder.layers.11.mlp.up_proj.input_max` |  | bfloat16 | 6.031 | 6.031 | 0 | 6.031 | 6.031 | 6.031 |
| 1488 | `model.vision_tower.encoder.layers.11.mlp.up_proj.input_min` |  | bfloat16 | 6.062 | -6.062 | 0 | -6.062 | -6.062 | -6.062 |
| 1489 | `model.vision_tower.encoder.layers.11.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -2.036e-05 | 0.03609 | -0.2695 | 0.2695 | 0.02429, 0.03369, 0.07812, 0.04858, -0.02234, -0.01495, 0.02612, 0.04858 |
| 1490 | `model.vision_tower.encoder.layers.11.mlp.up_proj.output_max` |  | bfloat16 | 5.969 | 5.969 | 0 | 5.969 | 5.969 | 5.969 |
| 1491 | `model.vision_tower.encoder.layers.11.mlp.up_proj.output_min` |  | bfloat16 | 6 | -6 | 0 | -6 | -6 | -6 |
| 1492 | `model.vision_tower.encoder.layers.11.post_attention_layernorm.weight` | 768 | bfloat16 | 103.2 | 3.031 | 2.165 | -5 | 16.5 | 0.02698, 2.656, 2.109, 2, 4.219, 5.844, 1.719, 1.141 |
| 1493 | `model.vision_tower.encoder.layers.11.post_feedforward_layernorm.weight` | 768 | bfloat16 | 128.2 | 4.306 | 1.694 | -5.594 | 17.62 | 0.1523, 4.469, 4.531, 3.938, 3.562, 4.5, 3.828, 3.109 |
| 1494 | `model.vision_tower.encoder.layers.11.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 68.84 | 2.252 | 1.048 | -0.106 | 5.312 | 0.04468, 2.688, 3.406, 2.719, 3.781, 2.547, 2.609, 1.336 |
| 1495 | `model.vision_tower.encoder.layers.11.self_attn.k_norm.weight` | 64 | bfloat16 | 9.625 | 1.203 | 0 | 1.203 | 1.203 | 1.203, 1.203, 1.203, 1.203, 1.203, 1.203, 1.203, 1.203 |
| 1496 | `model.vision_tower.encoder.layers.11.self_attn.k_proj.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 1497 | `model.vision_tower.encoder.layers.11.self_attn.k_proj.input_min` |  | bfloat16 | 11.88 | -11.88 | 0 | -11.88 | -11.88 | -11.88 |
| 1498 | `model.vision_tower.encoder.layers.11.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 5.076e-05 | 0.03609 | -0.3555 | 0.3477 | 0.06934, -0.01007, 0.01385, -0.01385, 0.01892, 0.01007, -0.01636, 0.0354 |
| 1499 | `model.vision_tower.encoder.layers.11.self_attn.k_proj.output_max` |  | bfloat16 | 13.75 | 13.75 | 0 | 13.75 | 13.75 | 13.75 |
| 1500 | `model.vision_tower.encoder.layers.11.self_attn.k_proj.output_min` |  | bfloat16 | 13.81 | -13.81 | 0 | -13.81 | -13.81 | -13.81 |
| 1501 | `model.vision_tower.encoder.layers.11.self_attn.o_proj.input_max` |  | bfloat16 | 2.391 | 2.391 | 0 | 2.391 | 2.391 | 2.391 |
| 1502 | `model.vision_tower.encoder.layers.11.self_attn.o_proj.input_min` |  | bfloat16 | 2.406 | -2.406 | 0 | -2.406 | -2.406 | -2.406 |
| 1503 | `model.vision_tower.encoder.layers.11.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 6.81e-06 | 0.03609 | -0.2988 | 0.3047 | 0.04956, 0.02625, -0.06128, -0.01459, 0.02625, 0.005829, 0.02332, -0.03198 |
| 1504 | `model.vision_tower.encoder.layers.11.self_attn.o_proj.output_max` |  | bfloat16 | 3.703 | 3.703 | 0 | 3.703 | 3.703 | 3.703 |
| 1505 | `model.vision_tower.encoder.layers.11.self_attn.o_proj.output_min` |  | bfloat16 | 3.734 | -3.734 | 0 | -3.734 | -3.734 | -3.734 |
| 1506 | `model.vision_tower.encoder.layers.11.self_attn.q_norm.weight` | 64 | bfloat16 | 3.328 | 0.416 | 0 | 0.416 | 0.416 | 0.416, 0.416, 0.416, 0.416, 0.416, 0.416, 0.416, 0.416 |
| 1507 | `model.vision_tower.encoder.layers.11.self_attn.q_proj.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 1508 | `model.vision_tower.encoder.layers.11.self_attn.q_proj.input_min` |  | bfloat16 | 11.88 | -11.88 | 0 | -11.88 | -11.88 | -11.88 |
| 1509 | `model.vision_tower.encoder.layers.11.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 8.991e-06 | 0.03608 | -0.707 | 0.4941 | 0.01819, -0.003632, 0.0127, 0.001816, 0.007263, 0.0437, 0.0437, 0.001816 |
| 1510 | `model.vision_tower.encoder.layers.11.self_attn.q_proj.output_max` |  | bfloat16 | 14.38 | 14.38 | 0 | 14.38 | 14.38 | 14.38 |
| 1511 | `model.vision_tower.encoder.layers.11.self_attn.q_proj.output_min` |  | bfloat16 | 14.5 | -14.5 | 0 | -14.5 | -14.5 | -14.5 |
| 1512 | `model.vision_tower.encoder.layers.11.self_attn.v_proj.input_max` |  | bfloat16 | 11.75 | 11.75 | 0 | 11.75 | 11.75 | 11.75 |
| 1513 | `model.vision_tower.encoder.layers.11.self_attn.v_proj.input_min` |  | bfloat16 | 11.88 | -11.88 | 0 | -11.88 | -11.88 | -11.88 |
| 1514 | `model.vision_tower.encoder.layers.11.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -1.157e-05 | 0.03609 | -0.3496 | 0.3945 | -0.04321, -0.001801, -0.0271, 0.0415, 0.03064, 0.09033, -0.09912, 0.0144 |
| 1515 | `model.vision_tower.encoder.layers.11.self_attn.v_proj.output_max` |  | bfloat16 | 13.75 | 13.75 | 0 | 13.75 | 13.75 | 13.75 |
| 1516 | `model.vision_tower.encoder.layers.11.self_attn.v_proj.output_min` |  | bfloat16 | 13.81 | -13.81 | 0 | -13.81 | -13.81 | -13.81 |
| 1517 | `model.vision_tower.encoder.layers.12.input_layernorm.weight` | 768 | bfloat16 | 121.5 | 4.035 | 1.711 | -0.04932 | 9.062 | 0.1089, 3.531, 5.062, 4.5, 5.781, 4.469, 5.719, 2.938 |
| 1518 | `model.vision_tower.encoder.layers.12.mlp.down_proj.input_max` |  | bfloat16 | 13 | 13 | 0 | 13 | 13 | 13 |
| 1519 | `model.vision_tower.encoder.layers.12.mlp.down_proj.input_min` |  | bfloat16 | 13.12 | -13.12 | 0 | -13.12 | -13.12 | -13.12 |
| 1520 | `model.vision_tower.encoder.layers.12.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 1.366e-05 | 0.01804 | -0.2129 | 0.2188 | 0.001968, 0.01575, -0.01373, -0.03149, 0.0177, 0.04907, -0.04712, -0.0354 |
| 1521 | `model.vision_tower.encoder.layers.12.mlp.down_proj.output_max` |  | bfloat16 | 4.406 | 4.406 | 0 | 4.406 | 4.406 | 4.406 |
| 1522 | `model.vision_tower.encoder.layers.12.mlp.down_proj.output_min` |  | bfloat16 | 4.438 | -4.438 | 0 | -4.438 | -4.438 | -4.438 |
| 1523 | `model.vision_tower.encoder.layers.12.mlp.gate_proj.input_max` |  | bfloat16 | 6.938 | 6.938 | 0 | 6.938 | 6.938 | 6.938 |
| 1524 | `model.vision_tower.encoder.layers.12.mlp.gate_proj.input_min` |  | bfloat16 | 7 | -7 | 0 | -7 | -7 | -7 |
| 1525 | `model.vision_tower.encoder.layers.12.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0002254 | 0.03609 | -0.332 | 0.2949 | 0.01746, 0.0437, 0.01398, -0.0105, 0.01746, 0.01398, 0.0105, -0.03662 |
| 1526 | `model.vision_tower.encoder.layers.12.mlp.gate_proj.output_max` |  | bfloat16 | 7.094 | 7.094 | 0 | 7.094 | 7.094 | 7.094 |
| 1527 | `model.vision_tower.encoder.layers.12.mlp.gate_proj.output_min` |  | bfloat16 | 7.156 | -7.156 | 0 | -7.156 | -7.156 | -7.156 |
| 1528 | `model.vision_tower.encoder.layers.12.mlp.up_proj.input_max` |  | bfloat16 | 6.938 | 6.938 | 0 | 6.938 | 6.938 | 6.938 |
| 1529 | `model.vision_tower.encoder.layers.12.mlp.up_proj.input_min` |  | bfloat16 | 7 | -7 | 0 | -7 | -7 | -7 |
| 1530 | `model.vision_tower.encoder.layers.12.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 1.075e-05 | 0.03609 | -0.2676 | 0.2891 | 0.02686, -0.03589, -0.003586, -0.03955, 0.005371, 0.04663, 0.03418, 0.01257 |
| 1531 | `model.vision_tower.encoder.layers.12.mlp.up_proj.output_max` |  | bfloat16 | 7.094 | 7.094 | 0 | 7.094 | 7.094 | 7.094 |
| 1532 | `model.vision_tower.encoder.layers.12.mlp.up_proj.output_min` |  | bfloat16 | 7.156 | -7.156 | 0 | -7.156 | -7.156 | -7.156 |
| 1533 | `model.vision_tower.encoder.layers.12.post_attention_layernorm.weight` | 768 | bfloat16 | 97.05 | 3.105 | 1.62 | -2.641 | 8.25 | -0.03076, 2.219, 1.656, 1.938, 3.797, 5.219, 1.258, 1.305 |
| 1534 | `model.vision_tower.encoder.layers.12.post_feedforward_layernorm.weight` | 768 | bfloat16 | 186.3 | 6.271 | 2.423 | -9.25 | 18.5 | -0.02002, 6.312, 6.469, 5.656, 5.906, 6.781, 6.344, -1.484 |
| 1535 | `model.vision_tower.encoder.layers.12.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 64.89 | 2.161 | 0.9014 | -0.07666 | 5.062 | 0.1074, 2.375, 2.969, 2.719, 3.078, 2.203, 2.781, 1.57 |
| 1536 | `model.vision_tower.encoder.layers.12.self_attn.k_norm.weight` | 64 | bfloat16 | 11.19 | 1.398 | 0 | 1.398 | 1.398 | 1.398, 1.398, 1.398, 1.398, 1.398, 1.398, 1.398, 1.398 |
| 1537 | `model.vision_tower.encoder.layers.12.self_attn.k_proj.input_max` |  | bfloat16 | 14.19 | 14.19 | 0 | 14.19 | 14.19 | 14.19 |
| 1538 | `model.vision_tower.encoder.layers.12.self_attn.k_proj.input_min` |  | bfloat16 | 14.31 | -14.31 | 0 | -14.31 | -14.31 | -14.31 |
| 1539 | `model.vision_tower.encoder.layers.12.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -4.788e-05 | 0.03609 | -0.3457 | 0.3477 | 0, -0.0459, -0.01038, -0.02222, -0.004456, 0.0238, 0.01337, -0.02527 |
| 1540 | `model.vision_tower.encoder.layers.12.self_attn.k_proj.output_max` |  | bfloat16 | 17.88 | 17.88 | 0 | 17.88 | 17.88 | 17.88 |
| 1541 | `model.vision_tower.encoder.layers.12.self_attn.k_proj.output_min` |  | bfloat16 | 18 | -18 | 0 | -18 | -18 | -18 |
| 1542 | `model.vision_tower.encoder.layers.12.self_attn.o_proj.input_max` |  | bfloat16 | 1.922 | 1.922 | 0 | 1.922 | 1.922 | 1.922 |
| 1543 | `model.vision_tower.encoder.layers.12.self_attn.o_proj.input_min` |  | bfloat16 | 1.93 | -1.93 | 0 | -1.93 | -1.93 | -1.93 |
| 1544 | `model.vision_tower.encoder.layers.12.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -6.539e-06 | 0.03609 | -0.3086 | 0.3125 | 0.0108, 0.08643, 0.04321, 0.1406, 0.05029, 0.0791, -0.1045, -0.05396 |
| 1545 | `model.vision_tower.encoder.layers.12.self_attn.o_proj.output_max` |  | bfloat16 | 2.453 | 2.453 | 0 | 2.453 | 2.453 | 2.453 |
| 1546 | `model.vision_tower.encoder.layers.12.self_attn.o_proj.output_min` |  | bfloat16 | 2.469 | -2.469 | 0 | -2.469 | -2.469 | -2.469 |
| 1547 | `model.vision_tower.encoder.layers.12.self_attn.q_norm.weight` | 64 | bfloat16 | 2.859 | 0.3574 | 0 | 0.3574 | 0.3574 | 0.3574, 0.3574, 0.3574, 0.3574, 0.3574, 0.3574, 0.3574, 0.3574 |
| 1548 | `model.vision_tower.encoder.layers.12.self_attn.q_proj.input_max` |  | bfloat16 | 14.19 | 14.19 | 0 | 14.19 | 14.19 | 14.19 |
| 1549 | `model.vision_tower.encoder.layers.12.self_attn.q_proj.input_min` |  | bfloat16 | 14.31 | -14.31 | 0 | -14.31 | -14.31 | -14.31 |
| 1550 | `model.vision_tower.encoder.layers.12.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 3.248e-05 | 0.03609 | -0.2559 | 0.3516 | -0.001755, 0.02637, 0.02454, 0.02991, 0.04395, -0.01575, -0.02637, -0.001755 |
| 1551 | `model.vision_tower.encoder.layers.12.self_attn.q_proj.output_max` |  | bfloat16 | 15.62 | 15.62 | 0 | 15.62 | 15.62 | 15.62 |
| 1552 | `model.vision_tower.encoder.layers.12.self_attn.q_proj.output_min` |  | bfloat16 | 15.75 | -15.75 | 0 | -15.75 | -15.75 | -15.75 |
| 1553 | `model.vision_tower.encoder.layers.12.self_attn.v_proj.input_max` |  | bfloat16 | 14.19 | 14.19 | 0 | 14.19 | 14.19 | 14.19 |
| 1554 | `model.vision_tower.encoder.layers.12.self_attn.v_proj.input_min` |  | bfloat16 | 14.31 | -14.31 | 0 | -14.31 | -14.31 | -14.31 |
| 1555 | `model.vision_tower.encoder.layers.12.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -8.533e-05 | 0.03609 | -0.2773 | 0.2393 | -0.01562, 0.0177, -0.02942, -0.0332, -0.0177, 0.04517, -0.001953, 0.01562 |
| 1556 | `model.vision_tower.encoder.layers.12.self_attn.v_proj.output_max` |  | bfloat16 | 17.88 | 17.88 | 0 | 17.88 | 17.88 | 17.88 |
| 1557 | `model.vision_tower.encoder.layers.12.self_attn.v_proj.output_min` |  | bfloat16 | 18 | -18 | 0 | -18 | -18 | -18 |
| 1558 | `model.vision_tower.encoder.layers.13.input_layernorm.weight` | 768 | bfloat16 | 116 | 3.885 | 1.563 | -0.1738 | 8.875 | 0.1846, 3.828, 4.906, 4.781, 4.375, 4.312, 5.75, 3.906 |
| 1559 | `model.vision_tower.encoder.layers.13.mlp.down_proj.input_max` |  | bfloat16 | 12.19 | 12.19 | 0 | 12.19 | 12.19 | 12.19 |
| 1560 | `model.vision_tower.encoder.layers.13.mlp.down_proj.input_min` |  | bfloat16 | 12.31 | -12.31 | 0 | -12.31 | -12.31 | -12.31 |
| 1561 | `model.vision_tower.encoder.layers.13.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.72 | -5.192e-06 | 0.01804 | -0.2275 | 0.2148 | -0.02039, 0.001701, -0.003403, -0.02039, 0.005096, -0.01019, -0.03735, -0.05615 |
| 1562 | `model.vision_tower.encoder.layers.13.mlp.down_proj.output_max` |  | bfloat16 | 3.875 | 3.875 | 0 | 3.875 | 3.875 | 3.875 |
| 1563 | `model.vision_tower.encoder.layers.13.mlp.down_proj.output_min` |  | bfloat16 | 3.906 | -3.906 | 0 | -3.906 | -3.906 | -3.906 |
| 1564 | `model.vision_tower.encoder.layers.13.mlp.gate_proj.input_max` |  | bfloat16 | 7.938 | 7.938 | 0 | 7.938 | 7.938 | 7.938 |
| 1565 | `model.vision_tower.encoder.layers.13.mlp.gate_proj.input_min` |  | bfloat16 | 8 | -8 | 0 | -8 | -8 | -8 |
| 1566 | `model.vision_tower.encoder.layers.13.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -3.472e-05 | 0.03609 | -0.3066 | 0.3066 | 0, -0.06128, -0.02869, 0.03064, 0.04028, -0.03247, 0.01532, -0.001915 |
| 1567 | `model.vision_tower.encoder.layers.13.mlp.gate_proj.output_max` |  | bfloat16 | 8.188 | 8.188 | 0 | 8.188 | 8.188 | 8.188 |
| 1568 | `model.vision_tower.encoder.layers.13.mlp.gate_proj.output_min` |  | bfloat16 | 8.25 | -8.25 | 0 | -8.25 | -8.25 | -8.25 |
| 1569 | `model.vision_tower.encoder.layers.13.mlp.up_proj.input_max` |  | bfloat16 | 7.938 | 7.938 | 0 | 7.938 | 7.938 | 7.938 |
| 1570 | `model.vision_tower.encoder.layers.13.mlp.up_proj.input_min` |  | bfloat16 | 8 | -8 | 0 | -8 | -8 | -8 |
| 1571 | `model.vision_tower.encoder.layers.13.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -4.817e-05 | 0.03609 | -0.3926 | 0.3926 | -0.0177, -0.001968, 0.03931, -0.1216, -0.001968, 0.04126, 0.00589, 0.0177 |
| 1572 | `model.vision_tower.encoder.layers.13.mlp.up_proj.output_max` |  | bfloat16 | 8.188 | 8.188 | 0 | 8.188 | 8.188 | 8.188 |
| 1573 | `model.vision_tower.encoder.layers.13.mlp.up_proj.output_min` |  | bfloat16 | 8.25 | -8.25 | 0 | -8.25 | -8.25 | -8.25 |
| 1574 | `model.vision_tower.encoder.layers.13.post_attention_layernorm.weight` | 768 | bfloat16 | 154.9 | 4.988 | 2.522 | -3.641 | 11.88 | -0.004974, 4.375, 3.062, 4.312, 5.906, 7.531, 2.703, 2.344 |
| 1575 | `model.vision_tower.encoder.layers.13.post_feedforward_layernorm.weight` | 768 | bfloat16 | 271.7 | 9.081 | 3.693 | -10.44 | 19.5 | -0.1797, 9.938, 8.75, 9.375, 8.812, 9.938, 8.438, 6.094 |
| 1576 | `model.vision_tower.encoder.layers.13.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 55.66 | 1.886 | 0.6902 | -0.2471 | 4.406 | 0.1377, 2.375, 2.297, 2.344, 2.391, 1.914, 2.359, 2.266 |
| 1577 | `model.vision_tower.encoder.layers.13.self_attn.k_norm.weight` | 64 | bfloat16 | 11.25 | 1.406 | 0 | 1.406 | 1.406 | 1.406, 1.406, 1.406, 1.406, 1.406, 1.406, 1.406, 1.406 |
| 1578 | `model.vision_tower.encoder.layers.13.self_attn.k_proj.input_max` |  | bfloat16 | 16.88 | 16.88 | 0 | 16.88 | 16.88 | 16.88 |
| 1579 | `model.vision_tower.encoder.layers.13.self_attn.k_proj.input_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 1580 | `model.vision_tower.encoder.layers.13.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 8.303e-05 | 0.03609 | -0.4238 | 0.4961 | 0.007874, 0.04736, 0.02893, -0.01843, 0.08154, 0.03687, 0.05542, 0.03955 |
| 1581 | `model.vision_tower.encoder.layers.13.self_attn.k_proj.output_max` |  | bfloat16 | 21 | 21 | 0 | 21 | 21 | 21 |
| 1582 | `model.vision_tower.encoder.layers.13.self_attn.k_proj.output_min` |  | bfloat16 | 21.12 | -21.12 | 0 | -21.12 | -21.12 | -21.12 |
| 1583 | `model.vision_tower.encoder.layers.13.self_attn.o_proj.input_max` |  | bfloat16 | 1.953 | 1.953 | 0 | 1.953 | 1.953 | 1.953 |
| 1584 | `model.vision_tower.encoder.layers.13.self_attn.o_proj.input_min` |  | bfloat16 | 1.969 | -1.969 | 0 | -1.969 | -1.969 | -1.969 |
| 1585 | `model.vision_tower.encoder.layers.13.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.462e-06 | 0.03609 | -0.3555 | 0.3359 | -0.08936, 0.1787, 0.004486, 0.04932, 0.01794, -0.08496, -0.1299, 0.008972 |
| 1586 | `model.vision_tower.encoder.layers.13.self_attn.o_proj.output_max` |  | bfloat16 | 2.828 | 2.828 | 0 | 2.828 | 2.828 | 2.828 |
| 1587 | `model.vision_tower.encoder.layers.13.self_attn.o_proj.output_min` |  | bfloat16 | 2.859 | -2.859 | 0 | -2.859 | -2.859 | -2.859 |
| 1588 | `model.vision_tower.encoder.layers.13.self_attn.q_norm.weight` | 64 | bfloat16 | 2.859 | 0.3574 | 0 | 0.3574 | 0.3574 | 0.3574, 0.3574, 0.3574, 0.3574, 0.3574, 0.3574, 0.3574, 0.3574 |
| 1589 | `model.vision_tower.encoder.layers.13.self_attn.q_proj.input_max` |  | bfloat16 | 16.88 | 16.88 | 0 | 16.88 | 16.88 | 16.88 |
| 1590 | `model.vision_tower.encoder.layers.13.self_attn.q_proj.input_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 1591 | `model.vision_tower.encoder.layers.13.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 5.875e-06 | 0.03609 | -0.3125 | 0.3027 | -0.03833, -0.005676, -0.01422, 0.01709, -0.0498, -0.02417, -0.005676, -0.09082 |
| 1592 | `model.vision_tower.encoder.layers.13.self_attn.q_proj.output_max` |  | bfloat16 | 17.38 | 17.38 | 0 | 17.38 | 17.38 | 17.38 |
| 1593 | `model.vision_tower.encoder.layers.13.self_attn.q_proj.output_min` |  | bfloat16 | 17.5 | -17.5 | 0 | -17.5 | -17.5 | -17.5 |
| 1594 | `model.vision_tower.encoder.layers.13.self_attn.v_proj.input_max` |  | bfloat16 | 16.88 | 16.88 | 0 | 16.88 | 16.88 | 16.88 |
| 1595 | `model.vision_tower.encoder.layers.13.self_attn.v_proj.input_min` |  | bfloat16 | 17.12 | -17.12 | 0 | -17.12 | -17.12 | -17.12 |
| 1596 | `model.vision_tower.encoder.layers.13.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -5.353e-05 | 0.03609 | -0.2988 | 0.3281 | -0.02795, -0.01196, 0.007996, 0.1099, 0.01196, -0.01398, 0.01196, -0.03198 |
| 1597 | `model.vision_tower.encoder.layers.13.self_attn.v_proj.output_max` |  | bfloat16 | 21 | 21 | 0 | 21 | 21 | 21 |
| 1598 | `model.vision_tower.encoder.layers.13.self_attn.v_proj.output_min` |  | bfloat16 | 21.12 | -21.12 | 0 | -21.12 | -21.12 | -21.12 |
| 1599 | `model.vision_tower.encoder.layers.14.input_layernorm.weight` | 768 | bfloat16 | 116.1 | 3.715 | 1.94 | -1.641 | 24.38 | 3.656, 3.188, 4.906, 3.203, 4, 4.625, 4.375, 4.656 |
| 1600 | `model.vision_tower.encoder.layers.14.mlp.down_proj.input_max` |  | bfloat16 | 13.44 | 13.44 | 0 | 13.44 | 13.44 | 13.44 |
| 1601 | `model.vision_tower.encoder.layers.14.mlp.down_proj.input_min` |  | bfloat16 | 13.5 | -13.5 | 0 | -13.5 | -13.5 | -13.5 |
| 1602 | `model.vision_tower.encoder.layers.14.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | -2.738e-06 | 0.01804 | -0.1953 | 0.1826 | 0.01251, 0.002274, 0.02051, 0.01819, -0.02161, 0.01709, -0.006836, -0.01941 |
| 1603 | `model.vision_tower.encoder.layers.14.mlp.down_proj.output_max` |  | bfloat16 | 4.719 | 4.719 | 0 | 4.719 | 4.719 | 4.719 |
| 1604 | `model.vision_tower.encoder.layers.14.mlp.down_proj.output_min` |  | bfloat16 | 4.75 | -4.75 | 0 | -4.75 | -4.75 | -4.75 |
| 1605 | `model.vision_tower.encoder.layers.14.mlp.gate_proj.input_max` |  | bfloat16 | 8.75 | 8.75 | 0 | 8.75 | 8.75 | 8.75 |
| 1606 | `model.vision_tower.encoder.layers.14.mlp.gate_proj.input_min` |  | bfloat16 | 8.812 | -8.812 | 0 | -8.812 | -8.812 | -8.812 |
| 1607 | `model.vision_tower.encoder.layers.14.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.44 | -0.0001084 | 0.03609 | -0.2969 | 0.2969 | 0.005402, 0.01624, -0.03979, -0.0271, -0.08496, -0.003601, 0.02527, 0.02881 |
| 1608 | `model.vision_tower.encoder.layers.14.mlp.gate_proj.output_max` |  | bfloat16 | 9.188 | 9.188 | 0 | 9.188 | 9.188 | 9.188 |
| 1609 | `model.vision_tower.encoder.layers.14.mlp.gate_proj.output_min` |  | bfloat16 | 9.25 | -9.25 | 0 | -9.25 | -9.25 | -9.25 |
| 1610 | `model.vision_tower.encoder.layers.14.mlp.up_proj.input_max` |  | bfloat16 | 8.75 | 8.75 | 0 | 8.75 | 8.75 | 8.75 |
| 1611 | `model.vision_tower.encoder.layers.14.mlp.up_proj.input_min` |  | bfloat16 | 8.812 | -8.812 | 0 | -8.812 | -8.812 | -8.812 |
| 1612 | `model.vision_tower.encoder.layers.14.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.44 | 1.583e-05 | 0.03609 | -0.2559 | 0.2578 | -0.09766, -0.03003, 0.03369, 0.001877, 0.02063, 0.02625, -0.01312, -0.03564 |
| 1613 | `model.vision_tower.encoder.layers.14.mlp.up_proj.output_max` |  | bfloat16 | 9.188 | 9.188 | 0 | 9.188 | 9.188 | 9.188 |
| 1614 | `model.vision_tower.encoder.layers.14.mlp.up_proj.output_min` |  | bfloat16 | 9.25 | -9.25 | 0 | -9.25 | -9.25 | -9.25 |
| 1615 | `model.vision_tower.encoder.layers.14.post_attention_layernorm.weight` | 768 | bfloat16 | 289.1 | 7.763 | 6.972 | -12.19 | 49 | -6.344, 5.344, 2.672, 4.688, 10.62, 12.56, 3.953, 4.25 |
| 1616 | `model.vision_tower.encoder.layers.14.post_feedforward_layernorm.weight` | 768 | bfloat16 | 455.3 | 15.25 | 6.105 | -3.672 | 32.75 | -0.1787, 17.38, 13.56, 16.88, 15.94, 17.62, 16.5, 11.62 |
| 1617 | `model.vision_tower.encoder.layers.14.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 49.87 | 1.688 | 0.6252 | -0.02661 | 3.562 | 0.1279, 2.109, 2.281, 2.297, 2.047, 1.594, 2.203, 2.484 |
| 1618 | `model.vision_tower.encoder.layers.14.self_attn.k_norm.weight` | 64 | bfloat16 | 10.31 | 1.289 | 0 | 1.289 | 1.289 | 1.289, 1.289, 1.289, 1.289, 1.289, 1.289, 1.289, 1.289 |
| 1619 | `model.vision_tower.encoder.layers.14.self_attn.k_proj.input_max` |  | bfloat16 | 19.5 | 19.5 | 0 | 19.5 | 19.5 | 19.5 |
| 1620 | `model.vision_tower.encoder.layers.14.self_attn.k_proj.input_min` |  | bfloat16 | 19.62 | -19.62 | 0 | -19.62 | -19.62 | -19.62 |
| 1621 | `model.vision_tower.encoder.layers.14.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.62 | -1.838e-05 | 0.03596 | -0.3887 | 0.3848 | -0.01086, 0.02905, -0.04175, 0, -0.06689, -0.02905, 0.01819, -0.05615 |
| 1622 | `model.vision_tower.encoder.layers.14.self_attn.k_proj.output_max` |  | bfloat16 | 22.88 | 22.88 | 0 | 22.88 | 22.88 | 22.88 |
| 1623 | `model.vision_tower.encoder.layers.14.self_attn.k_proj.output_min` |  | bfloat16 | 23 | -23 | 0 | -23 | -23 | -23 |
| 1624 | `model.vision_tower.encoder.layers.14.self_attn.o_proj.input_max` |  | bfloat16 | 2.328 | 2.328 | 0 | 2.328 | 2.328 | 2.328 |
| 1625 | `model.vision_tower.encoder.layers.14.self_attn.o_proj.input_min` |  | bfloat16 | 2.344 | -2.344 | 0 | -2.344 | -2.344 | -2.344 |
| 1626 | `model.vision_tower.encoder.layers.14.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | -7.492e-05 | 0.03607 | -0.4258 | 0.4121 | -0.01202, -0.007996, -0.001999, 0.003998, -0.003998, 0.007996, 0.01398, -0.01202 |
| 1627 | `model.vision_tower.encoder.layers.14.self_attn.o_proj.output_max` |  | bfloat16 | 2.875 | 2.875 | 0 | 2.875 | 2.875 | 2.875 |
| 1628 | `model.vision_tower.encoder.layers.14.self_attn.o_proj.output_min` |  | bfloat16 | 2.906 | -2.906 | 0 | -2.906 | -2.906 | -2.906 |
| 1629 | `model.vision_tower.encoder.layers.14.self_attn.q_norm.weight` | 64 | bfloat16 | 3.094 | 0.3867 | 0 | 0.3867 | 0.3867 | 0.3867, 0.3867, 0.3867, 0.3867, 0.3867, 0.3867, 0.3867, 0.3867 |
| 1630 | `model.vision_tower.encoder.layers.14.self_attn.q_proj.input_max` |  | bfloat16 | 19.5 | 19.5 | 0 | 19.5 | 19.5 | 19.5 |
| 1631 | `model.vision_tower.encoder.layers.14.self_attn.q_proj.input_min` |  | bfloat16 | 19.62 | -19.62 | 0 | -19.62 | -19.62 | -19.62 |
| 1632 | `model.vision_tower.encoder.layers.14.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | -6.326e-05 | 0.03608 | -0.373 | 0.3203 | 0.02319, -0.005798, 0.01929, -0.01733, 0.00193, -0.00193, 0.05225, -0.05615 |
| 1633 | `model.vision_tower.encoder.layers.14.self_attn.q_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 1634 | `model.vision_tower.encoder.layers.14.self_attn.q_proj.output_min` |  | bfloat16 | 22.12 | -22.12 | 0 | -22.12 | -22.12 | -22.12 |
| 1635 | `model.vision_tower.encoder.layers.14.self_attn.v_proj.input_max` |  | bfloat16 | 19.5 | 19.5 | 0 | 19.5 | 19.5 | 19.5 |
| 1636 | `model.vision_tower.encoder.layers.14.self_attn.v_proj.input_min` |  | bfloat16 | 19.62 | -19.62 | 0 | -19.62 | -19.62 | -19.62 |
| 1637 | `model.vision_tower.encoder.layers.14.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.5 | 7.313e-06 | 0.03581 | -0.4023 | 0.4023 | -0.0155, -0.02124, -0.01355, 0, 0.009705, 0.007751, 0.06201, -0.03687 |
| 1638 | `model.vision_tower.encoder.layers.14.self_attn.v_proj.output_max` |  | bfloat16 | 22.88 | 22.88 | 0 | 22.88 | 22.88 | 22.88 |
| 1639 | `model.vision_tower.encoder.layers.14.self_attn.v_proj.output_min` |  | bfloat16 | 23 | -23 | 0 | -23 | -23 | -23 |
| 1640 | `model.vision_tower.encoder.layers.15.input_layernorm.weight` | 768 | bfloat16 | 119.9 | 3.533 | 2.502 | -2.516 | 31.12 | 6.438, 3.5, 4.281, 4.281, 3.078, 3.266, 4.688, 5.312 |
| 1641 | `model.vision_tower.encoder.layers.15.mlp.down_proj.input_max` |  | bfloat16 | 11.94 | 11.94 | 0 | 11.94 | 11.94 | 11.94 |
| 1642 | `model.vision_tower.encoder.layers.15.mlp.down_proj.input_min` |  | bfloat16 | 12.06 | -12.06 | 0 | -12.06 | -12.06 | -12.06 |
| 1643 | `model.vision_tower.encoder.layers.15.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.48 | -1.367e-06 | 0.01789 | -0.2949 | 0.2754 | 0.003403, 0.03174, -0.002274, -0.005676, -0.007935, -0.01477, 0.01709, 0.01019 |
| 1644 | `model.vision_tower.encoder.layers.15.mlp.down_proj.output_max` |  | bfloat16 | 3.781 | 3.781 | 0 | 3.781 | 3.781 | 3.781 |
| 1645 | `model.vision_tower.encoder.layers.15.mlp.down_proj.output_min` |  | bfloat16 | 3.812 | -3.812 | 0 | -3.812 | -3.812 | -3.812 |
| 1646 | `model.vision_tower.encoder.layers.15.mlp.gate_proj.input_max` |  | bfloat16 | 7.844 | 7.844 | 0 | 7.844 | 7.844 | 7.844 |
| 1647 | `model.vision_tower.encoder.layers.15.mlp.gate_proj.input_min` |  | bfloat16 | 7.906 | -7.906 | 0 | -7.906 | -7.906 | -7.906 |
| 1648 | `model.vision_tower.encoder.layers.15.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0001319 | 0.03609 | -0.2871 | 0.2656 | -0.04126, -0.07422, -0.06104, 0.02173, -0.07617, 0.002182, -0.02612, -0.03906 |
| 1649 | `model.vision_tower.encoder.layers.15.mlp.gate_proj.output_max` |  | bfloat16 | 8.688 | 8.688 | 0 | 8.688 | 8.688 | 8.688 |
| 1650 | `model.vision_tower.encoder.layers.15.mlp.gate_proj.output_min` |  | bfloat16 | 8.75 | -8.75 | 0 | -8.75 | -8.75 | -8.75 |
| 1651 | `model.vision_tower.encoder.layers.15.mlp.up_proj.input_max` |  | bfloat16 | 7.844 | 7.844 | 0 | 7.844 | 7.844 | 7.844 |
| 1652 | `model.vision_tower.encoder.layers.15.mlp.up_proj.input_min` |  | bfloat16 | 7.906 | -7.906 | 0 | -7.906 | -7.906 | -7.906 |
| 1653 | `model.vision_tower.encoder.layers.15.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.44 | -8.164e-06 | 0.03609 | -0.2676 | 0.2656 | 0.02002, -0.05005, 0.03809, -0.01599, -0.001999, -0.0459, 0.003998, -0.01398 |
| 1654 | `model.vision_tower.encoder.layers.15.mlp.up_proj.output_max` |  | bfloat16 | 8.688 | 8.688 | 0 | 8.688 | 8.688 | 8.688 |
| 1655 | `model.vision_tower.encoder.layers.15.mlp.up_proj.output_min` |  | bfloat16 | 8.75 | -8.75 | 0 | -8.75 | -8.75 | -8.75 |
| 1656 | `model.vision_tower.encoder.layers.15.post_attention_layernorm.weight` | 768 | bfloat16 | 776.9 | 22.79 | 16.34 | -18.38 | 67.5 | -0.02991, 22, 17, 25.75, 26.25, 61.25, 13.25, 13.12 |
| 1657 | `model.vision_tower.encoder.layers.15.post_feedforward_layernorm.weight` | 768 | bfloat16 | 1314 | 43.35 | 19.23 | -5.125 | 71.5 | -0.08643, 61.5, 40.25, 56.5, 50, 34.25, 50, 38.75 |
| 1658 | `model.vision_tower.encoder.layers.15.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 33.02 | 1.068 | 0.5293 | -1.156 | 4.562 | 0.418, 1.477, 1.336, 1.203, 1.43, 0.1006, 1.453, 2.453 |
| 1659 | `model.vision_tower.encoder.layers.15.self_attn.k_norm.weight` | 64 | bfloat16 | 10.5 | 1.312 | 0 | 1.312 | 1.312 | 1.312, 1.312, 1.312, 1.312, 1.312, 1.312, 1.312, 1.312 |
| 1660 | `model.vision_tower.encoder.layers.15.self_attn.k_proj.input_max` |  | bfloat16 | 20.62 | 20.62 | 0 | 20.62 | 20.62 | 20.62 |
| 1661 | `model.vision_tower.encoder.layers.15.self_attn.k_proj.input_min` |  | bfloat16 | 20.75 | -20.75 | 0 | -20.75 | -20.75 | -20.75 |
| 1662 | `model.vision_tower.encoder.layers.15.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.68 | 7.118e-05 | 0.03604 | -0.4043 | 0.3984 | -0.002518, 0.001892, 0.006927, 0.003784, -0.002518, -0.01135, -0.005676, 0.02588 |
| 1663 | `model.vision_tower.encoder.layers.15.self_attn.k_proj.output_max` |  | bfloat16 | 24.88 | 24.88 | 0 | 24.88 | 24.88 | 24.88 |
| 1664 | `model.vision_tower.encoder.layers.15.self_attn.k_proj.output_min` |  | bfloat16 | 25 | -25 | 0 | -25 | -25 | -25 |
| 1665 | `model.vision_tower.encoder.layers.15.self_attn.o_proj.input_max` |  | bfloat16 | 2.25 | 2.25 | 0 | 2.25 | 2.25 | 2.25 |
| 1666 | `model.vision_tower.encoder.layers.15.self_attn.o_proj.input_min` |  | bfloat16 | 2.266 | -2.266 | 0 | -2.266 | -2.266 | -2.266 |
| 1667 | `model.vision_tower.encoder.layers.15.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.7 | 7.51e-05 | 0.03607 | -0.4023 | 0.4668 | 0.03833, 0.005737, 0.01337, -0.001915, -0.03442, 0.009583, -0.07666, -0.05347 |
| 1668 | `model.vision_tower.encoder.layers.15.self_attn.o_proj.output_max` |  | bfloat16 | 3.641 | 3.641 | 0 | 3.641 | 3.641 | 3.641 |
| 1669 | `model.vision_tower.encoder.layers.15.self_attn.o_proj.output_min` |  | bfloat16 | 3.656 | -3.656 | 0 | -3.656 | -3.656 | -3.656 |
| 1670 | `model.vision_tower.encoder.layers.15.self_attn.q_norm.weight` | 64 | bfloat16 | 3.047 | 0.3809 | 0 | 0.3809 | 0.3809 | 0.3809, 0.3809, 0.3809, 0.3809, 0.3809, 0.3809, 0.3809, 0.3809 |
| 1671 | `model.vision_tower.encoder.layers.15.self_attn.q_proj.input_max` |  | bfloat16 | 20.62 | 20.62 | 0 | 20.62 | 20.62 | 20.62 |
| 1672 | `model.vision_tower.encoder.layers.15.self_attn.q_proj.input_min` |  | bfloat16 | 20.75 | -20.75 | 0 | -20.75 | -20.75 | -20.75 |
| 1673 | `model.vision_tower.encoder.layers.15.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -7.106e-06 | 0.03609 | -0.4062 | 0.4355 | 0.004822, -0.004822, -0.1016, -0.1016, -0.02905, -0.009644, 0.05811, -0.0918 |
| 1674 | `model.vision_tower.encoder.layers.15.self_attn.q_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 1675 | `model.vision_tower.encoder.layers.15.self_attn.q_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 1676 | `model.vision_tower.encoder.layers.15.self_attn.v_proj.input_max` |  | bfloat16 | 20.62 | 20.62 | 0 | 20.62 | 20.62 | 20.62 |
| 1677 | `model.vision_tower.encoder.layers.15.self_attn.v_proj.input_min` |  | bfloat16 | 20.75 | -20.75 | 0 | -20.75 | -20.75 | -20.75 |
| 1678 | `model.vision_tower.encoder.layers.15.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.49 | -2.034e-06 | 0.0358 | -0.4199 | 0.4121 | 0.0199, -0.06641, -0.00885, -0.0177, 0.04639, -0.0199, 0.04199, -0.0155 |
| 1679 | `model.vision_tower.encoder.layers.15.self_attn.v_proj.output_max` |  | bfloat16 | 24.88 | 24.88 | 0 | 24.88 | 24.88 | 24.88 |
| 1680 | `model.vision_tower.encoder.layers.15.self_attn.v_proj.output_min` |  | bfloat16 | 25 | -25 | 0 | -25 | -25 | -25 |
| 1681 | `model.vision_tower.encoder.layers.2.input_layernorm.weight` | 768 | bfloat16 | 130.2 | 3.597 | 3.026 | -0.1973 | 24.88 | 0.1748, 1.859, 0.0708, 6.531, 1.758, 9.125, 1.469, 7.406 |
| 1682 | `model.vision_tower.encoder.layers.2.mlp.down_proj.input_max` |  | bfloat16 | 81 | 81 | 0 | 81 | 81 | 81 |
| 1683 | `model.vision_tower.encoder.layers.2.mlp.down_proj.input_min` |  | bfloat16 | 81.5 | -81.5 | 0 | -81.5 | -81.5 | -81.5 |
| 1684 | `model.vision_tower.encoder.layers.2.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 6.35e-06 | 0.01804 | -0.2002 | 0.1973 | -0.03271, -0.004517, -0.002258, -0.002258, -0.01587, -0.001129, 0.03174, -0.002258 |
| 1685 | `model.vision_tower.encoder.layers.2.mlp.down_proj.output_max` |  | bfloat16 | 87.5 | 87.5 | 0 | 87.5 | 87.5 | 87.5 |
| 1686 | `model.vision_tower.encoder.layers.2.mlp.down_proj.output_min` |  | bfloat16 | 88 | -88 | 0 | -88 | -88 | -88 |
| 1687 | `model.vision_tower.encoder.layers.2.mlp.gate_proj.input_max` |  | bfloat16 | 9.625 | 9.625 | 0 | 9.625 | 9.625 | 9.625 |
| 1688 | `model.vision_tower.encoder.layers.2.mlp.gate_proj.input_min` |  | bfloat16 | 9.75 | -9.75 | 0 | -9.75 | -9.75 | -9.75 |
| 1689 | `model.vision_tower.encoder.layers.2.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -1.669e-05 | 0.03608 | -0.3281 | 0.3555 | 0.03394, 0.02649, 0.005676, -0.007568, -0.04541, 0.01514, -0.06055, -0.02075 |
| 1690 | `model.vision_tower.encoder.layers.2.mlp.gate_proj.output_max` |  | bfloat16 | 12.75 | 12.75 | 0 | 12.75 | 12.75 | 12.75 |
| 1691 | `model.vision_tower.encoder.layers.2.mlp.gate_proj.output_min` |  | bfloat16 | 12.88 | -12.88 | 0 | -12.88 | -12.88 | -12.88 |
| 1692 | `model.vision_tower.encoder.layers.2.mlp.up_proj.input_max` |  | bfloat16 | 9.625 | 9.625 | 0 | 9.625 | 9.625 | 9.625 |
| 1693 | `model.vision_tower.encoder.layers.2.mlp.up_proj.input_min` |  | bfloat16 | 9.75 | -9.75 | 0 | -9.75 | -9.75 | -9.75 |
| 1694 | `model.vision_tower.encoder.layers.2.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -2.718e-05 | 0.03608 | -0.3223 | 0.3438 | -0.03296, 0.005829, -0.02332, 0.009705, 0.003891, -0.02722, 0.01746, 0.04663 |
| 1695 | `model.vision_tower.encoder.layers.2.mlp.up_proj.output_max` |  | bfloat16 | 12.75 | 12.75 | 0 | 12.75 | 12.75 | 12.75 |
| 1696 | `model.vision_tower.encoder.layers.2.mlp.up_proj.output_min` |  | bfloat16 | 12.88 | -12.88 | 0 | -12.88 | -12.88 | -12.88 |
| 1697 | `model.vision_tower.encoder.layers.2.post_attention_layernorm.weight` | 768 | bfloat16 | 13.69 | 0.2765 | 0.4094 | -1.656 | 5.438 | 0.2256, 0.5273, 0.3379, 0.1445, 0.1318, 0.2461, 0.02148, 0.2656 |
| 1698 | `model.vision_tower.encoder.layers.2.post_feedforward_layernorm.weight` | 768 | bfloat16 | 26.91 | 0.6065 | 0.7589 | -0.5352 | 11.69 | 0.8789, 0.9414, 1.18, 0.3164, 0.5469, 0.2773, 0.07861, 0.5625 |
| 1699 | `model.vision_tower.encoder.layers.2.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 147.6 | 4.091 | 3.414 | -1.914 | 21.25 | 0.01917, 2.25, 0.5703, 6.531, 3.344, 10.69, 2.469, 11.81 |
| 1700 | `model.vision_tower.encoder.layers.2.self_attn.k_norm.weight` | 64 | bfloat16 | 11.44 | 1.43 | 0 | 1.43 | 1.43 | 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43, 1.43 |
| 1701 | `model.vision_tower.encoder.layers.2.self_attn.k_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1702 | `model.vision_tower.encoder.layers.2.self_attn.k_proj.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1703 | `model.vision_tower.encoder.layers.2.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -3.237e-05 | 0.03609 | -0.377 | 0.416 | -0.04419, 0.01154, 0, 0.04224, -0.01538, 0.01733, -0.08252, -0.05762 |
| 1704 | `model.vision_tower.encoder.layers.2.self_attn.k_proj.output_max` |  | bfloat16 | 19.88 | 19.88 | 0 | 19.88 | 19.88 | 19.88 |
| 1705 | `model.vision_tower.encoder.layers.2.self_attn.k_proj.output_min` |  | bfloat16 | 20.12 | -20.12 | 0 | -20.12 | -20.12 | -20.12 |
| 1706 | `model.vision_tower.encoder.layers.2.self_attn.o_proj.input_max` |  | bfloat16 | 2.891 | 2.891 | 0 | 2.891 | 2.891 | 2.891 |
| 1707 | `model.vision_tower.encoder.layers.2.self_attn.o_proj.input_min` |  | bfloat16 | 2.922 | -2.922 | 0 | -2.922 | -2.922 | -2.922 |
| 1708 | `model.vision_tower.encoder.layers.2.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.306e-05 | 0.03609 | -0.5273 | 0.543 | -0.1138, 0.01794, -0.01196, 0.03589, 0.04785, 0.06592, -0.07178, 0.1138 |
| 1709 | `model.vision_tower.encoder.layers.2.self_attn.o_proj.output_max` |  | bfloat16 | 13.38 | 13.38 | 0 | 13.38 | 13.38 | 13.38 |
| 1710 | `model.vision_tower.encoder.layers.2.self_attn.o_proj.output_min` |  | bfloat16 | 13.44 | -13.44 | 0 | -13.44 | -13.44 | -13.44 |
| 1711 | `model.vision_tower.encoder.layers.2.self_attn.q_norm.weight` | 64 | bfloat16 | 2.797 | 0.3496 | 0 | 0.3496 | 0.3496 | 0.3496, 0.3496, 0.3496, 0.3496, 0.3496, 0.3496, 0.3496, 0.3496 |
| 1712 | `model.vision_tower.encoder.layers.2.self_attn.q_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1713 | `model.vision_tower.encoder.layers.2.self_attn.q_proj.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1714 | `model.vision_tower.encoder.layers.2.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.034e-05 | 0.03609 | -0.3066 | 0.2852 | 0.007935, -0.05151, 0.0415, 0.0415, 0.01782, -0.01184, 0.01385, 0.02966 |
| 1715 | `model.vision_tower.encoder.layers.2.self_attn.q_proj.output_max` |  | bfloat16 | 22.88 | 22.88 | 0 | 22.88 | 22.88 | 22.88 |
| 1716 | `model.vision_tower.encoder.layers.2.self_attn.q_proj.output_min` |  | bfloat16 | 23.12 | -23.12 | 0 | -23.12 | -23.12 | -23.12 |
| 1717 | `model.vision_tower.encoder.layers.2.self_attn.v_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1718 | `model.vision_tower.encoder.layers.2.self_attn.v_proj.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1719 | `model.vision_tower.encoder.layers.2.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | -0.0001032 | 0.03609 | -0.3906 | 0.3633 | 0.02087, -0.03638, 0.008667, 0.026, 0.005219, -0.01904, 0.07471, -0.04517 |
| 1720 | `model.vision_tower.encoder.layers.2.self_attn.v_proj.output_max` |  | bfloat16 | 19.88 | 19.88 | 0 | 19.88 | 19.88 | 19.88 |
| 1721 | `model.vision_tower.encoder.layers.2.self_attn.v_proj.output_min` |  | bfloat16 | 20.12 | -20.12 | 0 | -20.12 | -20.12 | -20.12 |
| 1722 | `model.vision_tower.encoder.layers.3.input_layernorm.weight` | 768 | bfloat16 | 153.8 | 3.917 | 3.936 | -1.57 | 50.25 | 0.09326, 1.836, -0.04443, 13.19, 2.344, 6, 2.141, 5.312 |
| 1723 | `model.vision_tower.encoder.layers.3.mlp.down_proj.input_max` |  | bfloat16 | 87 | 87 | 0 | 87 | 87 | 87 |
| 1724 | `model.vision_tower.encoder.layers.3.mlp.down_proj.input_min` |  | bfloat16 | 87.5 | -87.5 | 0 | -87.5 | -87.5 | -87.5 |
| 1725 | `model.vision_tower.encoder.layers.3.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | -6.641e-06 | 0.01804 | -0.1855 | 0.1816 | -0.02942, 0.002258, -0.007935, 0.003387, -0.01355, 0.01355, -0.004517, 0.01019 |
| 1726 | `model.vision_tower.encoder.layers.3.mlp.down_proj.output_max` |  | bfloat16 | 62.25 | 62.25 | 0 | 62.25 | 62.25 | 62.25 |
| 1727 | `model.vision_tower.encoder.layers.3.mlp.down_proj.output_min` |  | bfloat16 | 62.75 | -62.75 | 0 | -62.75 | -62.75 | -62.75 |
| 1728 | `model.vision_tower.encoder.layers.3.mlp.gate_proj.input_max` |  | bfloat16 | 12.25 | 12.25 | 0 | 12.25 | 12.25 | 12.25 |
| 1729 | `model.vision_tower.encoder.layers.3.mlp.gate_proj.input_min` |  | bfloat16 | 12.38 | -12.38 | 0 | -12.38 | -12.38 | -12.38 |
| 1730 | `model.vision_tower.encoder.layers.3.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.000126 | 0.03609 | -0.3672 | 0.373 | 0.007446, -0.007446, -0.02234, -0.02979, -0.02051, 0.03345, -0.0354, 0.02051 |
| 1731 | `model.vision_tower.encoder.layers.3.mlp.gate_proj.output_max` |  | bfloat16 | 16 | 16 | 0 | 16 | 16 | 16 |
| 1732 | `model.vision_tower.encoder.layers.3.mlp.gate_proj.output_min` |  | bfloat16 | 16.12 | -16.12 | 0 | -16.12 | -16.12 | -16.12 |
| 1733 | `model.vision_tower.encoder.layers.3.mlp.up_proj.input_max` |  | bfloat16 | 12.25 | 12.25 | 0 | 12.25 | 12.25 | 12.25 |
| 1734 | `model.vision_tower.encoder.layers.3.mlp.up_proj.input_min` |  | bfloat16 | 12.38 | -12.38 | 0 | -12.38 | -12.38 | -12.38 |
| 1735 | `model.vision_tower.encoder.layers.3.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 1.956e-05 | 0.03609 | -0.3125 | 0.3262 | -0.01202, 0.003433, 0.03442, 0.03955, 0.03613, 0.0498, 0.05493, 0.05322 |
| 1736 | `model.vision_tower.encoder.layers.3.mlp.up_proj.output_max` |  | bfloat16 | 16 | 16 | 0 | 16 | 16 | 16 |
| 1737 | `model.vision_tower.encoder.layers.3.mlp.up_proj.output_min` |  | bfloat16 | 16.12 | -16.12 | 0 | -16.12 | -16.12 | -16.12 |
| 1738 | `model.vision_tower.encoder.layers.3.post_attention_layernorm.weight` | 768 | bfloat16 | 32.87 | 0.583 | 1.034 | -0.8516 | 8.688 | -0.01733, 0.06348, -0.007935, 0.5156, 0.125, 0.3027, 0.03015, 0.3398 |
| 1739 | `model.vision_tower.encoder.layers.3.post_feedforward_layernorm.weight` | 768 | bfloat16 | 31.65 | 0.6825 | 0.9165 | -2.453 | 13 | 0.4062, 0.9961, 0.7266, 0.9883, 0.4824, 0.375, 0.03516, 1.078 |
| 1740 | `model.vision_tower.encoder.layers.3.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 162.5 | 4.89 | 3.234 | -4.375 | 16.88 | 0.06787, 3.016, 0.8672, 12.38, 5.906, 11.44, 4.844, 7.5 |
| 1741 | `model.vision_tower.encoder.layers.3.self_attn.k_norm.weight` | 64 | bfloat16 | 11.06 | 1.383 | 0 | 1.383 | 1.383 | 1.383, 1.383, 1.383, 1.383, 1.383, 1.383, 1.383, 1.383 |
| 1742 | `model.vision_tower.encoder.layers.3.self_attn.k_proj.input_max` |  | bfloat16 | 13.06 | 13.06 | 0 | 13.06 | 13.06 | 13.06 |
| 1743 | `model.vision_tower.encoder.layers.3.self_attn.k_proj.input_min` |  | bfloat16 | 13.12 | -13.12 | 0 | -13.12 | -13.12 | -13.12 |
| 1744 | `model.vision_tower.encoder.layers.3.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.913e-05 | 0.03609 | -0.416 | 0.3691 | 0.03931, 0.04346, -0.01855, 0.02271, -0.01855, 0.01031, -0.02063, -0.05981 |
| 1745 | `model.vision_tower.encoder.layers.3.self_attn.k_proj.output_max` |  | bfloat16 | 23.25 | 23.25 | 0 | 23.25 | 23.25 | 23.25 |
| 1746 | `model.vision_tower.encoder.layers.3.self_attn.k_proj.output_min` |  | bfloat16 | 23.38 | -23.38 | 0 | -23.38 | -23.38 | -23.38 |
| 1747 | `model.vision_tower.encoder.layers.3.self_attn.o_proj.input_max` |  | bfloat16 | 2.828 | 2.828 | 0 | 2.828 | 2.828 | 2.828 |
| 1748 | `model.vision_tower.encoder.layers.3.self_attn.o_proj.input_min` |  | bfloat16 | 2.844 | -2.844 | 0 | -2.844 | -2.844 | -2.844 |
| 1749 | `model.vision_tower.encoder.layers.3.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -4.957e-05 | 0.03609 | -0.4316 | 0.4277 | 0.09668, 0.004028, -0.02417, 0.008057, 0.1172, -0.01208, 0.0564, 0.04443 |
| 1750 | `model.vision_tower.encoder.layers.3.self_attn.o_proj.output_max` |  | bfloat16 | 8 | 8 | 0 | 8 | 8 | 8 |
| 1751 | `model.vision_tower.encoder.layers.3.self_attn.o_proj.output_min` |  | bfloat16 | 8.062 | -8.062 | 0 | -8.062 | -8.062 | -8.062 |
| 1752 | `model.vision_tower.encoder.layers.3.self_attn.q_norm.weight` | 64 | bfloat16 | 2.891 | 0.3613 | 0 | 0.3613 | 0.3613 | 0.3613, 0.3613, 0.3613, 0.3613, 0.3613, 0.3613, 0.3613, 0.3613 |
| 1753 | `model.vision_tower.encoder.layers.3.self_attn.q_proj.input_max` |  | bfloat16 | 13.06 | 13.06 | 0 | 13.06 | 13.06 | 13.06 |
| 1754 | `model.vision_tower.encoder.layers.3.self_attn.q_proj.input_min` |  | bfloat16 | 13.12 | -13.12 | 0 | -13.12 | -13.12 | -13.12 |
| 1755 | `model.vision_tower.encoder.layers.3.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 7.166e-06 | 0.03609 | -0.543 | 0.5039 | -0.01819, -0.03113, 0.01556, 0.06494, 0.02075, 0.01294, 0.007782, 0.03113 |
| 1756 | `model.vision_tower.encoder.layers.3.self_attn.q_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 1757 | `model.vision_tower.encoder.layers.3.self_attn.q_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 1758 | `model.vision_tower.encoder.layers.3.self_attn.v_proj.input_max` |  | bfloat16 | 13.06 | 13.06 | 0 | 13.06 | 13.06 | 13.06 |
| 1759 | `model.vision_tower.encoder.layers.3.self_attn.v_proj.input_min` |  | bfloat16 | 13.12 | -13.12 | 0 | -13.12 | -13.12 | -13.12 |
| 1760 | `model.vision_tower.encoder.layers.3.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.756e-05 | 0.03609 | -0.2695 | 0.2949 | 0.06201, 0.06738, 0.003647, -0.02551, -0.009094, -0.03638, 0.01636, 0.04199 |
| 1761 | `model.vision_tower.encoder.layers.3.self_attn.v_proj.output_max` |  | bfloat16 | 23.25 | 23.25 | 0 | 23.25 | 23.25 | 23.25 |
| 1762 | `model.vision_tower.encoder.layers.3.self_attn.v_proj.output_min` |  | bfloat16 | 23.38 | -23.38 | 0 | -23.38 | -23.38 | -23.38 |
| 1763 | `model.vision_tower.encoder.layers.4.input_layernorm.weight` | 768 | bfloat16 | 163.4 | 4.006 | 4.33 | -1.883 | 50 | 0.04346, 1.359, 0.6211, 12.56, 2, 5.188, 0.4902, 7.594 |
| 1764 | `model.vision_tower.encoder.layers.4.mlp.down_proj.input_max` |  | bfloat16 | 90 | 90 | 0 | 90 | 90 | 90 |
| 1765 | `model.vision_tower.encoder.layers.4.mlp.down_proj.input_min` |  | bfloat16 | 91 | -91 | 0 | -91 | -91 | -91 |
| 1766 | `model.vision_tower.encoder.layers.4.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.69 | -1.053e-05 | 0.01803 | -0.2178 | 0.2197 | 0, -0.02441, -0.007111, 0.03149, -0.004059, 0.003052, 0.01831, -0.006104 |
| 1767 | `model.vision_tower.encoder.layers.4.mlp.down_proj.output_max` |  | bfloat16 | 73.5 | 73.5 | 0 | 73.5 | 73.5 | 73.5 |
| 1768 | `model.vision_tower.encoder.layers.4.mlp.down_proj.output_min` |  | bfloat16 | 74 | -74 | 0 | -74 | -74 | -74 |
| 1769 | `model.vision_tower.encoder.layers.4.mlp.gate_proj.input_max` |  | bfloat16 | 12.69 | 12.69 | 0 | 12.69 | 12.69 | 12.69 |
| 1770 | `model.vision_tower.encoder.layers.4.mlp.gate_proj.input_min` |  | bfloat16 | 12.81 | -12.81 | 0 | -12.81 | -12.81 | -12.81 |
| 1771 | `model.vision_tower.encoder.layers.4.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.41 | -0.00022 | 0.03608 | -0.3438 | 0.3262 | -0.02295, 0, 0.021, 0.06689, 0.01337, 0.001915, 0.01721, -0.005737 |
| 1772 | `model.vision_tower.encoder.layers.4.mlp.gate_proj.output_max` |  | bfloat16 | 17.25 | 17.25 | 0 | 17.25 | 17.25 | 17.25 |
| 1773 | `model.vision_tower.encoder.layers.4.mlp.gate_proj.output_min` |  | bfloat16 | 17.38 | -17.38 | 0 | -17.38 | -17.38 | -17.38 |
| 1774 | `model.vision_tower.encoder.layers.4.mlp.up_proj.input_max` |  | bfloat16 | 12.69 | 12.69 | 0 | 12.69 | 12.69 | 12.69 |
| 1775 | `model.vision_tower.encoder.layers.4.mlp.up_proj.input_min` |  | bfloat16 | 12.81 | -12.81 | 0 | -12.81 | -12.81 | -12.81 |
| 1776 | `model.vision_tower.encoder.layers.4.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -6.929e-06 | 0.03608 | -0.3066 | 0.3008 | -0.003754, 0.02441, 0.02808, -0.0188, -0.01685, -0.01123, 0.001877, -0.02625 |
| 1777 | `model.vision_tower.encoder.layers.4.mlp.up_proj.output_max` |  | bfloat16 | 17.25 | 17.25 | 0 | 17.25 | 17.25 | 17.25 |
| 1778 | `model.vision_tower.encoder.layers.4.mlp.up_proj.output_min` |  | bfloat16 | 17.38 | -17.38 | 0 | -17.38 | -17.38 | -17.38 |
| 1779 | `model.vision_tower.encoder.layers.4.post_attention_layernorm.weight` | 768 | bfloat16 | 25.07 | 0.5451 | 0.7223 | -0.2559 | 5.688 | 0.123, 0.09668, 0.1191, 0.832, 0.1279, 0.3301, 0.05176, 1.219 |
| 1780 | `model.vision_tower.encoder.layers.4.post_feedforward_layernorm.weight` | 768 | bfloat16 | 34.86 | 0.8016 | 0.9703 | -0.8164 | 12.75 | 0.6133, 0.5039, 0.6445, 1.078, 0.2832, 0.3945, 0.2773, 1.883 |
| 1781 | `model.vision_tower.encoder.layers.4.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 166.7 | 4.914 | 3.473 | -0.1826 | 29.12 | -0.01349, 1.852, 1.07, 7.469, 3.625, 6.938, 9.812, 4.719 |
| 1782 | `model.vision_tower.encoder.layers.4.self_attn.k_norm.weight` | 64 | bfloat16 | 10.06 | 1.258 | 0 | 1.258 | 1.258 | 1.258, 1.258, 1.258, 1.258, 1.258, 1.258, 1.258, 1.258 |
| 1783 | `model.vision_tower.encoder.layers.4.self_attn.k_proj.input_max` |  | bfloat16 | 12.69 | 12.69 | 0 | 12.69 | 12.69 | 12.69 |
| 1784 | `model.vision_tower.encoder.layers.4.self_attn.k_proj.input_min` |  | bfloat16 | 12.81 | -12.81 | 0 | -12.81 | -12.81 | -12.81 |
| 1785 | `model.vision_tower.encoder.layers.4.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 5.918e-06 | 0.03609 | -0.4414 | 0.4609 | -0.04492, 0.001801, 0.01257, -0.01257, 0.0144, 0.0918, -0.03052, 0.003601 |
| 1786 | `model.vision_tower.encoder.layers.4.self_attn.k_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 1787 | `model.vision_tower.encoder.layers.4.self_attn.k_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 1788 | `model.vision_tower.encoder.layers.4.self_attn.o_proj.input_max` |  | bfloat16 | 2.375 | 2.375 | 0 | 2.375 | 2.375 | 2.375 |
| 1789 | `model.vision_tower.encoder.layers.4.self_attn.o_proj.input_min` |  | bfloat16 | 2.391 | -2.391 | 0 | -2.391 | -2.391 | -2.391 |
| 1790 | `model.vision_tower.encoder.layers.4.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -1.468e-05 | 0.03609 | -0.3379 | 0.3555 | -0.004181, -0.09229, 0.05444, -0.05029, 0.021, -0.02515, 0.03784, 0.03345 |
| 1791 | `model.vision_tower.encoder.layers.4.self_attn.o_proj.output_max` |  | bfloat16 | 5.719 | 5.719 | 0 | 5.719 | 5.719 | 5.719 |
| 1792 | `model.vision_tower.encoder.layers.4.self_attn.o_proj.output_min` |  | bfloat16 | 5.781 | -5.781 | 0 | -5.781 | -5.781 | -5.781 |
| 1793 | `model.vision_tower.encoder.layers.4.self_attn.q_norm.weight` | 64 | bfloat16 | 3.172 | 0.3965 | 0 | 0.3965 | 0.3965 | 0.3965, 0.3965, 0.3965, 0.3965, 0.3965, 0.3965, 0.3965, 0.3965 |
| 1794 | `model.vision_tower.encoder.layers.4.self_attn.q_proj.input_max` |  | bfloat16 | 12.69 | 12.69 | 0 | 12.69 | 12.69 | 12.69 |
| 1795 | `model.vision_tower.encoder.layers.4.self_attn.q_proj.input_min` |  | bfloat16 | 12.81 | -12.81 | 0 | -12.81 | -12.81 | -12.81 |
| 1796 | `model.vision_tower.encoder.layers.4.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 5.53e-05 | 0.03609 | -0.5391 | 0.5039 | 0.07031, 0, 0.001602, -0.009644, 0.03857, 0.004822, 0.06396, 0.04175 |
| 1797 | `model.vision_tower.encoder.layers.4.self_attn.q_proj.output_max` |  | bfloat16 | 19 | 19 | 0 | 19 | 19 | 19 |
| 1798 | `model.vision_tower.encoder.layers.4.self_attn.q_proj.output_min` |  | bfloat16 | 19.12 | -19.12 | 0 | -19.12 | -19.12 | -19.12 |
| 1799 | `model.vision_tower.encoder.layers.4.self_attn.v_proj.input_max` |  | bfloat16 | 12.69 | 12.69 | 0 | 12.69 | 12.69 | 12.69 |
| 1800 | `model.vision_tower.encoder.layers.4.self_attn.v_proj.input_min` |  | bfloat16 | 12.81 | -12.81 | 0 | -12.81 | -12.81 | -12.81 |
| 1801 | `model.vision_tower.encoder.layers.4.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.406e-05 | 0.03609 | -0.2832 | 0.2734 | -0.005493, -0.06592, 0.001831, 0.005493, 0.02747, -0.005493, -0.001831, 0.04932 |
| 1802 | `model.vision_tower.encoder.layers.4.self_attn.v_proj.output_max` |  | bfloat16 | 21.88 | 21.88 | 0 | 21.88 | 21.88 | 21.88 |
| 1803 | `model.vision_tower.encoder.layers.4.self_attn.v_proj.output_min` |  | bfloat16 | 22 | -22 | 0 | -22 | -22 | -22 |
| 1804 | `model.vision_tower.encoder.layers.5.input_layernorm.weight` | 768 | bfloat16 | 157.6 | 3.973 | 4.074 | -17.5 | 45.75 | -0.009094, 0.793, -0.08398, 4.031, 2.203, 5.531, 11.62, 3.172 |
| 1805 | `model.vision_tower.encoder.layers.5.mlp.down_proj.input_max` |  | bfloat16 | 47.75 | 47.75 | 0 | 47.75 | 47.75 | 47.75 |
| 1806 | `model.vision_tower.encoder.layers.5.mlp.down_proj.input_min` |  | bfloat16 | 48.25 | -48.25 | 0 | -48.25 | -48.25 | -48.25 |
| 1807 | `model.vision_tower.encoder.layers.5.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.68 | -5.314e-06 | 0.01802 | -0.1572 | 0.1533 | -0.004242, 0.00106, -0.008484, -0.03198, -0.01276, 0.03613, 0.004242, 0.009583 |
| 1808 | `model.vision_tower.encoder.layers.5.mlp.down_proj.output_max` |  | bfloat16 | 30.88 | 30.88 | 0 | 30.88 | 30.88 | 30.88 |
| 1809 | `model.vision_tower.encoder.layers.5.mlp.down_proj.output_min` |  | bfloat16 | 31.12 | -31.12 | 0 | -31.12 | -31.12 | -31.12 |
| 1810 | `model.vision_tower.encoder.layers.5.mlp.gate_proj.input_max` |  | bfloat16 | 9.812 | 9.812 | 0 | 9.812 | 9.812 | 9.812 |
| 1811 | `model.vision_tower.encoder.layers.5.mlp.gate_proj.input_min` |  | bfloat16 | 9.875 | -9.875 | 0 | -9.875 | -9.875 | -9.875 |
| 1812 | `model.vision_tower.encoder.layers.5.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -5.827e-05 | 0.03608 | -0.3398 | 0.3359 | 0.01312, -0.02258, 0.04517, 0.01501, -0.0376, 0, 0.005646, -0.03955 |
| 1813 | `model.vision_tower.encoder.layers.5.mlp.gate_proj.output_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1814 | `model.vision_tower.encoder.layers.5.mlp.gate_proj.output_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1815 | `model.vision_tower.encoder.layers.5.mlp.up_proj.input_max` |  | bfloat16 | 9.812 | 9.812 | 0 | 9.812 | 9.812 | 9.812 |
| 1816 | `model.vision_tower.encoder.layers.5.mlp.up_proj.input_min` |  | bfloat16 | 9.875 | -9.875 | 0 | -9.875 | -9.875 | -9.875 |
| 1817 | `model.vision_tower.encoder.layers.5.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.42 | -1.647e-05 | 0.03608 | -0.2969 | 0.2871 | 0.03809, 0.01202, -0.05225, -0.001999, 0.01202, 0.01202, 0.003998, 0.04004 |
| 1818 | `model.vision_tower.encoder.layers.5.mlp.up_proj.output_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1819 | `model.vision_tower.encoder.layers.5.mlp.up_proj.output_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1820 | `model.vision_tower.encoder.layers.5.post_attention_layernorm.weight` | 768 | bfloat16 | 33.64 | 0.7304 | 0.9702 | -0.7344 | 13.5 | -0.7344, 0.125, -0.03613, 0.5547, 0.1816, 0.8516, 0.4238, 1.008 |
| 1821 | `model.vision_tower.encoder.layers.5.post_feedforward_layernorm.weight` | 768 | bfloat16 | 49 | 1.248 | 1.253 | -1.32 | 21 | 1.906, 1.25, 0.832, 1.047, 0.8477, 0.8984, 0.8086, 1.773 |
| 1822 | `model.vision_tower.encoder.layers.5.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 176.6 | 4.718 | 4.285 | -0.1465 | 31.75 | 0.07373, 2.141, 0.668, 3.266, 4.562, 6.594, 17.62, 2.25 |
| 1823 | `model.vision_tower.encoder.layers.5.self_attn.k_norm.weight` | 64 | bfloat16 | 10.88 | 1.359 | 0 | 1.359 | 1.359 | 1.359, 1.359, 1.359, 1.359, 1.359, 1.359, 1.359, 1.359 |
| 1824 | `model.vision_tower.encoder.layers.5.self_attn.k_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1825 | `model.vision_tower.encoder.layers.5.self_attn.k_proj.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1826 | `model.vision_tower.encoder.layers.5.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -3.776e-05 | 0.03609 | -0.4648 | 0.4609 | 0.02808, 0.00296, 0.03247, -0.02063, 0.0177, -0.06055, -0.00148, -0.00885 |
| 1827 | `model.vision_tower.encoder.layers.5.self_attn.k_proj.output_max` |  | bfloat16 | 19.25 | 19.25 | 0 | 19.25 | 19.25 | 19.25 |
| 1828 | `model.vision_tower.encoder.layers.5.self_attn.k_proj.output_min` |  | bfloat16 | 19.38 | -19.38 | 0 | -19.38 | -19.38 | -19.38 |
| 1829 | `model.vision_tower.encoder.layers.5.self_attn.o_proj.input_max` |  | bfloat16 | 2.828 | 2.828 | 0 | 2.828 | 2.828 | 2.828 |
| 1830 | `model.vision_tower.encoder.layers.5.self_attn.o_proj.input_min` |  | bfloat16 | 2.844 | -2.844 | 0 | -2.844 | -2.844 | -2.844 |
| 1831 | `model.vision_tower.encoder.layers.5.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -4.174e-05 | 0.03609 | -0.3516 | 0.3301 | 0.02441, -0.09082, 0.03833, -0.04541, -0.0874, -0.01044, -0.09424, -0.07666 |
| 1832 | `model.vision_tower.encoder.layers.5.self_attn.o_proj.output_max` |  | bfloat16 | 4.75 | 4.75 | 0 | 4.75 | 4.75 | 4.75 |
| 1833 | `model.vision_tower.encoder.layers.5.self_attn.o_proj.output_min` |  | bfloat16 | 4.781 | -4.781 | 0 | -4.781 | -4.781 | -4.781 |
| 1834 | `model.vision_tower.encoder.layers.5.self_attn.q_norm.weight` | 64 | bfloat16 | 2.938 | 0.3672 | 0 | 0.3672 | 0.3672 | 0.3672, 0.3672, 0.3672, 0.3672, 0.3672, 0.3672, 0.3672, 0.3672 |
| 1835 | `model.vision_tower.encoder.layers.5.self_attn.q_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1836 | `model.vision_tower.encoder.layers.5.self_attn.q_proj.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1837 | `model.vision_tower.encoder.layers.5.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.241e-05 | 0.03609 | -0.3008 | 0.4043 | 0.03955, -0.002197, 0.02856, -0.03516, 0.04834, 0.006592, -0.04614, 0.004395 |
| 1838 | `model.vision_tower.encoder.layers.5.self_attn.q_proj.output_max` |  | bfloat16 | 15.75 | 15.75 | 0 | 15.75 | 15.75 | 15.75 |
| 1839 | `model.vision_tower.encoder.layers.5.self_attn.q_proj.output_min` |  | bfloat16 | 15.88 | -15.88 | 0 | -15.88 | -15.88 | -15.88 |
| 1840 | `model.vision_tower.encoder.layers.5.self_attn.v_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1841 | `model.vision_tower.encoder.layers.5.self_attn.v_proj.input_min` |  | bfloat16 | 12.12 | -12.12 | 0 | -12.12 | -12.12 | -12.12 |
| 1842 | `model.vision_tower.encoder.layers.5.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 5.697e-05 | 0.03609 | -0.3887 | 0.3301 | -0.01556, -0.02637, -0.006226, -0.006226, 0.08691, -0.03418, 0.001556, -0.004669 |
| 1843 | `model.vision_tower.encoder.layers.5.self_attn.v_proj.output_max` |  | bfloat16 | 19.25 | 19.25 | 0 | 19.25 | 19.25 | 19.25 |
| 1844 | `model.vision_tower.encoder.layers.5.self_attn.v_proj.output_min` |  | bfloat16 | 19.38 | -19.38 | 0 | -19.38 | -19.38 | -19.38 |
| 1845 | `model.vision_tower.encoder.layers.6.input_layernorm.weight` | 768 | bfloat16 | 159.2 | 4.723 | 3.269 | -0.2383 | 44.5 | 0.1562, 2.312, 1.656, 5.031, 3.812, 6.5, 6.594, 3.906 |
| 1846 | `model.vision_tower.encoder.layers.6.mlp.down_proj.input_max` |  | bfloat16 | 71 | 71 | 0 | 71 | 71 | 71 |
| 1847 | `model.vision_tower.encoder.layers.6.mlp.down_proj.input_min` |  | bfloat16 | 71.5 | -71.5 | 0 | -71.5 | -71.5 | -71.5 |
| 1848 | `model.vision_tower.encoder.layers.6.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 1.08e-05 | 0.01804 | -0.1846 | 0.1934 | 0.005707, -0.001144, 0.02283, -0.01941, 0.04004, 0.01941, -0.009155, 0.01599 |
| 1849 | `model.vision_tower.encoder.layers.6.mlp.down_proj.output_max` |  | bfloat16 | 38 | 38 | 0 | 38 | 38 | 38 |
| 1850 | `model.vision_tower.encoder.layers.6.mlp.down_proj.output_min` |  | bfloat16 | 38.5 | -38.5 | 0 | -38.5 | -38.5 | -38.5 |
| 1851 | `model.vision_tower.encoder.layers.6.mlp.gate_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1852 | `model.vision_tower.encoder.layers.6.mlp.gate_proj.input_min` |  | bfloat16 | 12.19 | -12.19 | 0 | -12.19 | -12.19 | -12.19 |
| 1853 | `model.vision_tower.encoder.layers.6.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -8.893e-05 | 0.03608 | -0.3086 | 0.3047 | -0.01807, 0.03064, -0.009033, -0.03418, -0.007202, -0.005402, 0.08105, 0.04688 |
| 1854 | `model.vision_tower.encoder.layers.6.mlp.gate_proj.output_max` |  | bfloat16 | 15.06 | 15.06 | 0 | 15.06 | 15.06 | 15.06 |
| 1855 | `model.vision_tower.encoder.layers.6.mlp.gate_proj.output_min` |  | bfloat16 | 15.19 | -15.19 | 0 | -15.19 | -15.19 | -15.19 |
| 1856 | `model.vision_tower.encoder.layers.6.mlp.up_proj.input_max` |  | bfloat16 | 12.06 | 12.06 | 0 | 12.06 | 12.06 | 12.06 |
| 1857 | `model.vision_tower.encoder.layers.6.mlp.up_proj.input_min` |  | bfloat16 | 12.19 | -12.19 | 0 | -12.19 | -12.19 | -12.19 |
| 1858 | `model.vision_tower.encoder.layers.6.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 1.058e-05 | 0.03609 | -0.2988 | 0.293 | -0.04199, 0.009155, 0.02197, 0.0293, -0.03467, 0.01648, 0.08057, -0.1099 |
| 1859 | `model.vision_tower.encoder.layers.6.mlp.up_proj.output_max` |  | bfloat16 | 15.06 | 15.06 | 0 | 15.06 | 15.06 | 15.06 |
| 1860 | `model.vision_tower.encoder.layers.6.mlp.up_proj.output_min` |  | bfloat16 | 15.19 | -15.19 | 0 | -15.19 | -15.19 | -15.19 |
| 1861 | `model.vision_tower.encoder.layers.6.post_attention_layernorm.weight` | 768 | bfloat16 | 22.43 | 0.5618 | 0.583 | -0.6562 | 5.094 | 0.0957, 0.2773, 0.07178, 0.5977, 0.2285, 0.4062, 0.2285, 0.9219 |
| 1862 | `model.vision_tower.encoder.layers.6.post_feedforward_layernorm.weight` | 768 | bfloat16 | 33.18 | 0.8393 | 0.8543 | -0.8789 | 12.38 | -0.0918, 0.8672, 0.8672, 0.918, 0.5039, 0.6758, 0.3711, 1.555 |
| 1863 | `model.vision_tower.encoder.layers.6.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 237.7 | 7.301 | 4.505 | -2.453 | 43 | 0.2363, 5.719, 3.031, 6.125, 10.94, 9.688, 14.81, 3.297 |
| 1864 | `model.vision_tower.encoder.layers.6.self_attn.k_norm.weight` | 64 | bfloat16 | 11.31 | 1.414 | 0 | 1.414 | 1.414 | 1.414, 1.414, 1.414, 1.414, 1.414, 1.414, 1.414, 1.414 |
| 1865 | `model.vision_tower.encoder.layers.6.self_attn.k_proj.input_max` |  | bfloat16 | 10 | 10 | 0 | 10 | 10 | 10 |
| 1866 | `model.vision_tower.encoder.layers.6.self_attn.k_proj.input_min` |  | bfloat16 | 10.06 | -10.06 | 0 | -10.06 | -10.06 | -10.06 |
| 1867 | `model.vision_tower.encoder.layers.6.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | -3.094e-05 | 0.03609 | -0.3867 | 0.3926 | -0.01672, -0.02673, 0.001671, -0.01337, 0.006683, 0.04688, 0.005035, 0 |
| 1868 | `model.vision_tower.encoder.layers.6.self_attn.k_proj.output_max` |  | bfloat16 | 18.25 | 18.25 | 0 | 18.25 | 18.25 | 18.25 |
| 1869 | `model.vision_tower.encoder.layers.6.self_attn.k_proj.output_min` |  | bfloat16 | 18.38 | -18.38 | 0 | -18.38 | -18.38 | -18.38 |
| 1870 | `model.vision_tower.encoder.layers.6.self_attn.o_proj.input_max` |  | bfloat16 | 2.312 | 2.312 | 0 | 2.312 | 2.312 | 2.312 |
| 1871 | `model.vision_tower.encoder.layers.6.self_attn.o_proj.input_min` |  | bfloat16 | 2.328 | -2.328 | 0 | -2.328 | -2.328 | -2.328 |
| 1872 | `model.vision_tower.encoder.layers.6.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -2.643e-05 | 0.03609 | -0.3496 | 0.3145 | -0.04858, 0, -0.1582, 0.07666, -0.1299, 0.1133, -0.07666, 0.008118 |
| 1873 | `model.vision_tower.encoder.layers.6.self_attn.o_proj.output_max` |  | bfloat16 | 4.062 | 4.062 | 0 | 4.062 | 4.062 | 4.062 |
| 1874 | `model.vision_tower.encoder.layers.6.self_attn.o_proj.output_min` |  | bfloat16 | 4.094 | -4.094 | 0 | -4.094 | -4.094 | -4.094 |
| 1875 | `model.vision_tower.encoder.layers.6.self_attn.q_norm.weight` | 64 | bfloat16 | 2.828 | 0.3535 | 0 | 0.3535 | 0.3535 | 0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535 |
| 1876 | `model.vision_tower.encoder.layers.6.self_attn.q_proj.input_max` |  | bfloat16 | 10 | 10 | 0 | 10 | 10 | 10 |
| 1877 | `model.vision_tower.encoder.layers.6.self_attn.q_proj.input_min` |  | bfloat16 | 10.06 | -10.06 | 0 | -10.06 | -10.06 | -10.06 |
| 1878 | `model.vision_tower.encoder.layers.6.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.119e-06 | 0.03609 | -0.2754 | 0.2598 | -0.003479, 0.01733, -0.01733, -0.03833, -0.005219, 0.02258, -0.03833, 0.05737 |
| 1879 | `model.vision_tower.encoder.layers.6.self_attn.q_proj.output_max` |  | bfloat16 | 14.31 | 14.31 | 0 | 14.31 | 14.31 | 14.31 |
| 1880 | `model.vision_tower.encoder.layers.6.self_attn.q_proj.output_min` |  | bfloat16 | 14.38 | -14.38 | 0 | -14.38 | -14.38 | -14.38 |
| 1881 | `model.vision_tower.encoder.layers.6.self_attn.v_proj.input_max` |  | bfloat16 | 10 | 10 | 0 | 10 | 10 | 10 |
| 1882 | `model.vision_tower.encoder.layers.6.self_attn.v_proj.input_min` |  | bfloat16 | 10.06 | -10.06 | 0 | -10.06 | -10.06 | -10.06 |
| 1883 | `model.vision_tower.encoder.layers.6.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -4.236e-05 | 0.03609 | -0.291 | 0.2871 | -0.02478, -0.08008, -0.09717, 0.05322, 0.03247, 0.01709, 0.003799, 0.01141 |
| 1884 | `model.vision_tower.encoder.layers.6.self_attn.v_proj.output_max` |  | bfloat16 | 18.25 | 18.25 | 0 | 18.25 | 18.25 | 18.25 |
| 1885 | `model.vision_tower.encoder.layers.6.self_attn.v_proj.output_min` |  | bfloat16 | 18.38 | -18.38 | 0 | -18.38 | -18.38 | -18.38 |
| 1886 | `model.vision_tower.encoder.layers.7.input_layernorm.weight` | 768 | bfloat16 | 163 | 4.611 | 3.653 | -0.2617 | 46 | 0.21, 2.281, 1.25, 2.312, 5.531, 7.594, 10.44, 1.539 |
| 1887 | `model.vision_tower.encoder.layers.7.mlp.down_proj.input_max` |  | bfloat16 | 32 | 32 | 0 | 32 | 32 | 32 |
| 1888 | `model.vision_tower.encoder.layers.7.mlp.down_proj.input_min` |  | bfloat16 | 32.25 | -32.25 | 0 | -32.25 | -32.25 | -32.25 |
| 1889 | `model.vision_tower.encoder.layers.7.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | -3.41e-06 | 0.01804 | -0.1729 | 0.1885 | -0.02271, 0.00885, 0.01868, 0.01672, 0.01086, 0.00885, -0.0177, -0.02075 |
| 1890 | `model.vision_tower.encoder.layers.7.mlp.down_proj.output_max` |  | bfloat16 | 14.69 | 14.69 | 0 | 14.69 | 14.69 | 14.69 |
| 1891 | `model.vision_tower.encoder.layers.7.mlp.down_proj.output_min` |  | bfloat16 | 14.81 | -14.81 | 0 | -14.81 | -14.81 | -14.81 |
| 1892 | `model.vision_tower.encoder.layers.7.mlp.gate_proj.input_max` |  | bfloat16 | 8.875 | 8.875 | 0 | 8.875 | 8.875 | 8.875 |
| 1893 | `model.vision_tower.encoder.layers.7.mlp.gate_proj.input_min` |  | bfloat16 | 8.938 | -8.938 | 0 | -8.938 | -8.938 | -8.938 |
| 1894 | `model.vision_tower.encoder.layers.7.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0002514 | 0.03608 | -0.3047 | 0.3184 | -0.01172, 0.01953, -0.123, 0.01758, -0.0293, 0.01562, -0.001953, 0.08008 |
| 1895 | `model.vision_tower.encoder.layers.7.mlp.gate_proj.output_max` |  | bfloat16 | 10.19 | 10.19 | 0 | 10.19 | 10.19 | 10.19 |
| 1896 | `model.vision_tower.encoder.layers.7.mlp.gate_proj.output_min` |  | bfloat16 | 10.25 | -10.25 | 0 | -10.25 | -10.25 | -10.25 |
| 1897 | `model.vision_tower.encoder.layers.7.mlp.up_proj.input_max` |  | bfloat16 | 8.875 | 8.875 | 0 | 8.875 | 8.875 | 8.875 |
| 1898 | `model.vision_tower.encoder.layers.7.mlp.up_proj.input_min` |  | bfloat16 | 8.938 | -8.938 | 0 | -8.938 | -8.938 | -8.938 |
| 1899 | `model.vision_tower.encoder.layers.7.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -2.331e-05 | 0.03608 | -0.3047 | 0.3047 | 0.03931, -0.01178, -0.03149, -0.07861, 0.009827, -0.02356, 0.00589, 0.001968 |
| 1900 | `model.vision_tower.encoder.layers.7.mlp.up_proj.output_max` |  | bfloat16 | 10.19 | 10.19 | 0 | 10.19 | 10.19 | 10.19 |
| 1901 | `model.vision_tower.encoder.layers.7.mlp.up_proj.output_min` |  | bfloat16 | 10.25 | -10.25 | 0 | -10.25 | -10.25 | -10.25 |
| 1902 | `model.vision_tower.encoder.layers.7.post_attention_layernorm.weight` | 768 | bfloat16 | 36.59 | 0.9004 | 0.9665 | -1.531 | 12.69 | 0.1157, 0.4453, -0.04736, 0.5938, 0.7227, 1.945, 0.7812, 0.9883 |
| 1903 | `model.vision_tower.encoder.layers.7.post_feedforward_layernorm.weight` | 768 | bfloat16 | 45.23 | 1.298 | 0.9897 | -0.9766 | 15.06 | 0.3457, 1.18, 0.918, 1.164, 0.7578, 1.094, 1.133, 1.633 |
| 1904 | `model.vision_tower.encoder.layers.7.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 171.8 | 5.262 | 3.278 | -0.6133 | 24.38 | 0.6797, 4.094, 2.672, 3.719, 7.75, 7.719, 13.62, 1.211 |
| 1905 | `model.vision_tower.encoder.layers.7.self_attn.k_norm.weight` | 64 | bfloat16 | 10.12 | 1.266 | 0 | 1.266 | 1.266 | 1.266, 1.266, 1.266, 1.266, 1.266, 1.266, 1.266, 1.266 |
| 1906 | `model.vision_tower.encoder.layers.7.self_attn.k_proj.input_max` |  | bfloat16 | 9.125 | 9.125 | 0 | 9.125 | 9.125 | 9.125 |
| 1907 | `model.vision_tower.encoder.layers.7.self_attn.k_proj.input_min` |  | bfloat16 | 9.188 | -9.188 | 0 | -9.188 | -9.188 | -9.188 |
| 1908 | `model.vision_tower.encoder.layers.7.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.805e-05 | 0.03609 | -0.3457 | 0.3672 | 0.04321, 0.06934, 0.01733, 0.02808, -0.04321, -0.02161, 0.01733, 0.07764 |
| 1909 | `model.vision_tower.encoder.layers.7.self_attn.k_proj.output_max` |  | bfloat16 | 15 | 15 | 0 | 15 | 15 | 15 |
| 1910 | `model.vision_tower.encoder.layers.7.self_attn.k_proj.output_min` |  | bfloat16 | 15.12 | -15.12 | 0 | -15.12 | -15.12 | -15.12 |
| 1911 | `model.vision_tower.encoder.layers.7.self_attn.o_proj.input_max` |  | bfloat16 | 2.516 | 2.516 | 0 | 2.516 | 2.516 | 2.516 |
| 1912 | `model.vision_tower.encoder.layers.7.self_attn.o_proj.input_min` |  | bfloat16 | 2.547 | -2.547 | 0 | -2.547 | -2.547 | -2.547 |
| 1913 | `model.vision_tower.encoder.layers.7.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 1.28e-05 | 0.03609 | -0.3398 | 0.3574 | 0.04736, 0.06201, -0.05469, -0.02185, 0, 0.1279, -0.0874, -0.02551 |
| 1914 | `model.vision_tower.encoder.layers.7.self_attn.o_proj.output_max` |  | bfloat16 | 3.75 | 3.75 | 0 | 3.75 | 3.75 | 3.75 |
| 1915 | `model.vision_tower.encoder.layers.7.self_attn.o_proj.output_min` |  | bfloat16 | 3.781 | -3.781 | 0 | -3.781 | -3.781 | -3.781 |
| 1916 | `model.vision_tower.encoder.layers.7.self_attn.q_norm.weight` | 64 | bfloat16 | 3.172 | 0.3965 | 0 | 0.3965 | 0.3965 | 0.3965, 0.3965, 0.3965, 0.3965, 0.3965, 0.3965, 0.3965, 0.3965 |
| 1917 | `model.vision_tower.encoder.layers.7.self_attn.q_proj.input_max` |  | bfloat16 | 9.125 | 9.125 | 0 | 9.125 | 9.125 | 9.125 |
| 1918 | `model.vision_tower.encoder.layers.7.self_attn.q_proj.input_min` |  | bfloat16 | 9.188 | -9.188 | 0 | -9.188 | -9.188 | -9.188 |
| 1919 | `model.vision_tower.encoder.layers.7.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 2.258e-05 | 0.03609 | -0.2695 | 0.2451 | -0.04395, 0.02515, 0.07861, -0.0564, -0.01575, -0.009399, -0.05957, 0.1787 |
| 1920 | `model.vision_tower.encoder.layers.7.self_attn.q_proj.output_max` |  | bfloat16 | 12.44 | 12.44 | 0 | 12.44 | 12.44 | 12.44 |
| 1921 | `model.vision_tower.encoder.layers.7.self_attn.q_proj.output_min` |  | bfloat16 | 12.56 | -12.56 | 0 | -12.56 | -12.56 | -12.56 |
| 1922 | `model.vision_tower.encoder.layers.7.self_attn.v_proj.input_max` |  | bfloat16 | 9.125 | 9.125 | 0 | 9.125 | 9.125 | 9.125 |
| 1923 | `model.vision_tower.encoder.layers.7.self_attn.v_proj.input_min` |  | bfloat16 | 9.188 | -9.188 | 0 | -9.188 | -9.188 | -9.188 |
| 1924 | `model.vision_tower.encoder.layers.7.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 3.359e-05 | 0.03609 | -0.4004 | 0.252 | 0.06152, -0.02112, -0.01733, 0.04419, 0.03467, 0.01538, 0.01343, -0.01154 |
| 1925 | `model.vision_tower.encoder.layers.7.self_attn.v_proj.output_max` |  | bfloat16 | 15 | 15 | 0 | 15 | 15 | 15 |
| 1926 | `model.vision_tower.encoder.layers.7.self_attn.v_proj.output_min` |  | bfloat16 | 15.12 | -15.12 | 0 | -15.12 | -15.12 | -15.12 |
| 1927 | `model.vision_tower.encoder.layers.8.input_layernorm.weight` | 768 | bfloat16 | 154.8 | 4.918 | 2.651 | -0.2061 | 20.12 | 0.5352, 4, 2.438, 3.406, 8.062, 6.562, 9.812, 1.305 |
| 1928 | `model.vision_tower.encoder.layers.8.mlp.down_proj.input_max` |  | bfloat16 | 20.12 | 20.12 | 0 | 20.12 | 20.12 | 20.12 |
| 1929 | `model.vision_tower.encoder.layers.8.mlp.down_proj.input_min` |  | bfloat16 | 20.25 | -20.25 | 0 | -20.25 | -20.25 | -20.25 |
| 1930 | `model.vision_tower.encoder.layers.8.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | 5.709e-06 | 0.01804 | -0.1924 | 0.2051 | 0, 0, 0.01855, 0.01855, -0.02478, -0.02893, -0.05371, 0.03931 |
| 1931 | `model.vision_tower.encoder.layers.8.mlp.down_proj.output_max` |  | bfloat16 | 6.156 | 6.156 | 0 | 6.156 | 6.156 | 6.156 |
| 1932 | `model.vision_tower.encoder.layers.8.mlp.down_proj.output_min` |  | bfloat16 | 6.188 | -6.188 | 0 | -6.188 | -6.188 | -6.188 |
| 1933 | `model.vision_tower.encoder.layers.8.mlp.gate_proj.input_max` |  | bfloat16 | 6.938 | 6.938 | 0 | 6.938 | 6.938 | 6.938 |
| 1934 | `model.vision_tower.encoder.layers.8.mlp.gate_proj.input_min` |  | bfloat16 | 7 | -7 | 0 | -7 | -7 | -7 |
| 1935 | `model.vision_tower.encoder.layers.8.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0001904 | 0.03609 | -0.2852 | 0.3086 | 0.01331, 0.03223, -0.007599, 0.007599, -0.0019, -0.06079, 0.08936, -0.0304 |
| 1936 | `model.vision_tower.encoder.layers.8.mlp.gate_proj.output_max` |  | bfloat16 | 7.312 | 7.312 | 0 | 7.312 | 7.312 | 7.312 |
| 1937 | `model.vision_tower.encoder.layers.8.mlp.gate_proj.output_min` |  | bfloat16 | 7.375 | -7.375 | 0 | -7.375 | -7.375 | -7.375 |
| 1938 | `model.vision_tower.encoder.layers.8.mlp.up_proj.input_max` |  | bfloat16 | 6.938 | 6.938 | 0 | 6.938 | 6.938 | 6.938 |
| 1939 | `model.vision_tower.encoder.layers.8.mlp.up_proj.input_min` |  | bfloat16 | 7 | -7 | 0 | -7 | -7 | -7 |
| 1940 | `model.vision_tower.encoder.layers.8.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 1.154e-05 | 0.03609 | -0.2832 | 0.2852 | 0.02515, -0.02319, 0.009705, -0.0155, 0.0155, 0.0542, 0.005798, -0.05225 |
| 1941 | `model.vision_tower.encoder.layers.8.mlp.up_proj.output_max` |  | bfloat16 | 7.312 | 7.312 | 0 | 7.312 | 7.312 | 7.312 |
| 1942 | `model.vision_tower.encoder.layers.8.mlp.up_proj.output_min` |  | bfloat16 | 7.375 | -7.375 | 0 | -7.375 | -7.375 | -7.375 |
| 1943 | `model.vision_tower.encoder.layers.8.post_attention_layernorm.weight` | 768 | bfloat16 | 45.47 | 1.382 | 0.8853 | -0.8359 | 13.94 | -0.06787, 1.266, 0.6289, 0.8594, 1.297, 2, 0.8281, 0.9648 |
| 1944 | `model.vision_tower.encoder.layers.8.post_feedforward_layernorm.weight` | 768 | bfloat16 | 57.57 | 1.848 | 0.9494 | -2.859 | 12.44 | 0.04028, 2.109, 1.906, 1.977, 1.43, 1.594, 1.758, 1.992 |
| 1945 | `model.vision_tower.encoder.layers.8.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 114.4 | 3.645 | 1.936 | -0.08936 | 12.06 | 0.5078, 4.188, 4.281, 2.844, 6.375, 4.125, 6.031, 0.6211 |
| 1946 | `model.vision_tower.encoder.layers.8.self_attn.k_norm.weight` | 64 | bfloat16 | 11.69 | 1.461 | 0 | 1.461 | 1.461 | 1.461, 1.461, 1.461, 1.461, 1.461, 1.461, 1.461, 1.461 |
| 1947 | `model.vision_tower.encoder.layers.8.self_attn.k_proj.input_max` |  | bfloat16 | 10.44 | 10.44 | 0 | 10.44 | 10.44 | 10.44 |
| 1948 | `model.vision_tower.encoder.layers.8.self_attn.k_proj.input_min` |  | bfloat16 | 10.56 | -10.56 | 0 | -10.56 | -10.56 | -10.56 |
| 1949 | `model.vision_tower.encoder.layers.8.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 6.537e-05 | 0.03608 | -0.4824 | 0.4785 | -0.02136, 0.05518, 0.02319, -0.01953, -0.003555, 0.001778, 0.03394, 0.001778 |
| 1950 | `model.vision_tower.encoder.layers.8.self_attn.k_proj.output_max` |  | bfloat16 | 17.25 | 17.25 | 0 | 17.25 | 17.25 | 17.25 |
| 1951 | `model.vision_tower.encoder.layers.8.self_attn.k_proj.output_min` |  | bfloat16 | 17.38 | -17.38 | 0 | -17.38 | -17.38 | -17.38 |
| 1952 | `model.vision_tower.encoder.layers.8.self_attn.o_proj.input_max` |  | bfloat16 | 2.312 | 2.312 | 0 | 2.312 | 2.312 | 2.312 |
| 1953 | `model.vision_tower.encoder.layers.8.self_attn.o_proj.input_min` |  | bfloat16 | 2.344 | -2.344 | 0 | -2.344 | -2.344 | -2.344 |
| 1954 | `model.vision_tower.encoder.layers.8.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 6.933e-05 | 0.03609 | -0.4355 | 0.4277 | 0, 0.09375, 0.02783, 0.04517, -0.04175, 0.03125, 0.1318, 0.052 |
| 1955 | `model.vision_tower.encoder.layers.8.self_attn.o_proj.output_max` |  | bfloat16 | 3.484 | 3.484 | 0 | 3.484 | 3.484 | 3.484 |
| 1956 | `model.vision_tower.encoder.layers.8.self_attn.o_proj.output_min` |  | bfloat16 | 3.516 | -3.516 | 0 | -3.516 | -3.516 | -3.516 |
| 1957 | `model.vision_tower.encoder.layers.8.self_attn.q_norm.weight` | 64 | bfloat16 | 2.734 | 0.3418 | 0 | 0.3418 | 0.3418 | 0.3418, 0.3418, 0.3418, 0.3418, 0.3418, 0.3418, 0.3418, 0.3418 |
| 1958 | `model.vision_tower.encoder.layers.8.self_attn.q_proj.input_max` |  | bfloat16 | 10.44 | 10.44 | 0 | 10.44 | 10.44 | 10.44 |
| 1959 | `model.vision_tower.encoder.layers.8.self_attn.q_proj.input_min` |  | bfloat16 | 10.56 | -10.56 | 0 | -10.56 | -10.56 | -10.56 |
| 1960 | `model.vision_tower.encoder.layers.8.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | 6.04e-06 | 0.03609 | -0.4102 | 0.498 | 0.03833, 0, -0.0791, 0.008179, -0.05737, -0.02466, 0.06543, 0.01093 |
| 1961 | `model.vision_tower.encoder.layers.8.self_attn.q_proj.output_max` |  | bfloat16 | 14.75 | 14.75 | 0 | 14.75 | 14.75 | 14.75 |
| 1962 | `model.vision_tower.encoder.layers.8.self_attn.q_proj.output_min` |  | bfloat16 | 14.88 | -14.88 | 0 | -14.88 | -14.88 | -14.88 |
| 1963 | `model.vision_tower.encoder.layers.8.self_attn.v_proj.input_max` |  | bfloat16 | 10.44 | 10.44 | 0 | 10.44 | 10.44 | 10.44 |
| 1964 | `model.vision_tower.encoder.layers.8.self_attn.v_proj.input_min` |  | bfloat16 | 10.56 | -10.56 | 0 | -10.56 | -10.56 | -10.56 |
| 1965 | `model.vision_tower.encoder.layers.8.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -3.783e-06 | 0.03609 | -0.3184 | 0.3516 | 0.01263, 0.02344, 0.0271, 0.01447, 0.03076, -0.06299, -0.103, 0.0199 |
| 1966 | `model.vision_tower.encoder.layers.8.self_attn.v_proj.output_max` |  | bfloat16 | 17.25 | 17.25 | 0 | 17.25 | 17.25 | 17.25 |
| 1967 | `model.vision_tower.encoder.layers.8.self_attn.v_proj.output_min` |  | bfloat16 | 17.38 | -17.38 | 0 | -17.38 | -17.38 | -17.38 |
| 1968 | `model.vision_tower.encoder.layers.9.input_layernorm.weight` | 768 | bfloat16 | 144.2 | 4.684 | 2.27 | -0.1318 | 17.12 | 0.04663, 4.344, 3.672, 4, 7.281, 5.938, 8.062, 1.547 |
| 1969 | `model.vision_tower.encoder.layers.9.mlp.down_proj.input_max` |  | bfloat16 | 28.38 | 28.38 | 0 | 28.38 | 28.38 | 28.38 |
| 1970 | `model.vision_tower.encoder.layers.9.mlp.down_proj.input_min` |  | bfloat16 | 28.62 | -28.62 | 0 | -28.62 | -28.62 | -28.62 |
| 1971 | `model.vision_tower.encoder.layers.9.mlp.down_proj.linear.weight` | 768×3072 | bfloat16 | 27.71 | -4.022e-06 | 0.01804 | -0.2305 | 0.2539 | 0.01794, 0.02014, -0.008972, 0.03589, -0.006714, 0.008972, 0.02466, -0.04028 |
| 1972 | `model.vision_tower.encoder.layers.9.mlp.down_proj.output_max` |  | bfloat16 | 10.31 | 10.31 | 0 | 10.31 | 10.31 | 10.31 |
| 1973 | `model.vision_tower.encoder.layers.9.mlp.down_proj.output_min` |  | bfloat16 | 10.44 | -10.44 | 0 | -10.44 | -10.44 | -10.44 |
| 1974 | `model.vision_tower.encoder.layers.9.mlp.gate_proj.input_max` |  | bfloat16 | 8.625 | 8.625 | 0 | 8.625 | 8.625 | 8.625 |
| 1975 | `model.vision_tower.encoder.layers.9.mlp.gate_proj.input_min` |  | bfloat16 | 8.688 | -8.688 | 0 | -8.688 | -8.688 | -8.688 |
| 1976 | `model.vision_tower.encoder.layers.9.mlp.gate_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | -0.0003181 | 0.03608 | -0.3789 | 0.2988 | 0.01434, -0.0119, -0.01434, -0.004761, 0.08594, -0.01904, 0.06201, 0.009521 |
| 1977 | `model.vision_tower.encoder.layers.9.mlp.gate_proj.output_max` |  | bfloat16 | 9.438 | 9.438 | 0 | 9.438 | 9.438 | 9.438 |
| 1978 | `model.vision_tower.encoder.layers.9.mlp.gate_proj.output_min` |  | bfloat16 | 9.5 | -9.5 | 0 | -9.5 | -9.5 | -9.5 |
| 1979 | `model.vision_tower.encoder.layers.9.mlp.up_proj.input_max` |  | bfloat16 | 8.625 | 8.625 | 0 | 8.625 | 8.625 | 8.625 |
| 1980 | `model.vision_tower.encoder.layers.9.mlp.up_proj.input_min` |  | bfloat16 | 8.688 | -8.688 | 0 | -8.688 | -8.688 | -8.688 |
| 1981 | `model.vision_tower.encoder.layers.9.mlp.up_proj.linear.weight` | 3072×768 | bfloat16 | 55.43 | 6.968e-06 | 0.03609 | -0.3047 | 0.2754 | 0.006897, -0.01843, -0.01843, -0.009216, -0.05981, 0, 0.02295, 0.03442 |
| 1982 | `model.vision_tower.encoder.layers.9.mlp.up_proj.output_max` |  | bfloat16 | 9.438 | 9.438 | 0 | 9.438 | 9.438 | 9.438 |
| 1983 | `model.vision_tower.encoder.layers.9.mlp.up_proj.output_min` |  | bfloat16 | 9.5 | -9.5 | 0 | -9.5 | -9.5 | -9.5 |
| 1984 | `model.vision_tower.encoder.layers.9.post_attention_layernorm.weight` | 768 | bfloat16 | 34.25 | 1.081 | 0.5993 | -2.297 | 6.281 | -0.1079, 1.008, 0.3926, 0.8633, 1.227, 1.531, 0.6992, 0.5742 |
| 1985 | `model.vision_tower.encoder.layers.9.post_feedforward_layernorm.weight` | 768 | bfloat16 | 68.35 | 2.236 | 1.042 | -2.297 | 12.31 | 0.004791, 2.344, 2, 2.234, 1.609, 2.203, 2.406, 2.203 |
| 1986 | `model.vision_tower.encoder.layers.9.pre_feedforward_layernorm.weight` | 768 | bfloat16 | 117.6 | 3.85 | 1.783 | -0.08105 | 11.38 | 0.3555, 4.125, 4.469, 4.094, 6.406, 4.281, 5.031, 1.312 |
| 1987 | `model.vision_tower.encoder.layers.9.self_attn.k_norm.weight` | 64 | bfloat16 | 10.62 | 1.328 | 0 | 1.328 | 1.328 | 1.328, 1.328, 1.328, 1.328, 1.328, 1.328, 1.328, 1.328 |
| 1988 | `model.vision_tower.encoder.layers.9.self_attn.k_proj.input_max` |  | bfloat16 | 11.38 | 11.38 | 0 | 11.38 | 11.38 | 11.38 |
| 1989 | `model.vision_tower.encoder.layers.9.self_attn.k_proj.input_min` |  | bfloat16 | 11.5 | -11.5 | 0 | -11.5 | -11.5 | -11.5 |
| 1990 | `model.vision_tower.encoder.layers.9.self_attn.k_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 1.167e-05 | 0.03608 | -0.4219 | 0.4414 | 0.04736, -0.01068, 0.03662, 0.02441, -0.009155, -0.01068, 0.01672, 0.003052 |
| 1991 | `model.vision_tower.encoder.layers.9.self_attn.k_proj.output_max` |  | bfloat16 | 15.81 | 15.81 | 0 | 15.81 | 15.81 | 15.81 |
| 1992 | `model.vision_tower.encoder.layers.9.self_attn.k_proj.output_min` |  | bfloat16 | 15.94 | -15.94 | 0 | -15.94 | -15.94 | -15.94 |
| 1993 | `model.vision_tower.encoder.layers.9.self_attn.o_proj.input_max` |  | bfloat16 | 1.914 | 1.914 | 0 | 1.914 | 1.914 | 1.914 |
| 1994 | `model.vision_tower.encoder.layers.9.self_attn.o_proj.input_min` |  | bfloat16 | 1.93 | -1.93 | 0 | -1.93 | -1.93 | -1.93 |
| 1995 | `model.vision_tower.encoder.layers.9.self_attn.o_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -1.619e-06 | 0.03609 | -0.2539 | 0.2715 | -0.009888, 0.09229, -0.01978, 0.003296, 0.06934, -0.003296, -0.04956, 0.02307 |
| 1996 | `model.vision_tower.encoder.layers.9.self_attn.o_proj.output_max` |  | bfloat16 | 2.469 | 2.469 | 0 | 2.469 | 2.469 | 2.469 |
| 1997 | `model.vision_tower.encoder.layers.9.self_attn.o_proj.output_min` |  | bfloat16 | 2.484 | -2.484 | 0 | -2.484 | -2.484 | -2.484 |
| 1998 | `model.vision_tower.encoder.layers.9.self_attn.q_norm.weight` | 64 | bfloat16 | 3.016 | 0.377 | 0 | 0.377 | 0.377 | 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377, 0.377 |
| 1999 | `model.vision_tower.encoder.layers.9.self_attn.q_proj.input_max` |  | bfloat16 | 11.38 | 11.38 | 0 | 11.38 | 11.38 | 11.38 |
| 2000 | `model.vision_tower.encoder.layers.9.self_attn.q_proj.input_min` |  | bfloat16 | 11.5 | -11.5 | 0 | -11.5 | -11.5 | -11.5 |
| 2001 | `model.vision_tower.encoder.layers.9.self_attn.q_proj.linear.weight` | 768×768 | bfloat16 | 27.72 | -1.623e-05 | 0.03609 | -0.4336 | 0.4648 | -0.03369, 0.003754, 0.009399, 0.01685, -0.01501, -0.01312, 0.03003, -0.01123 |
| 2002 | `model.vision_tower.encoder.layers.9.self_attn.q_proj.output_max` |  | bfloat16 | 14.38 | 14.38 | 0 | 14.38 | 14.38 | 14.38 |
| 2003 | `model.vision_tower.encoder.layers.9.self_attn.q_proj.output_min` |  | bfloat16 | 14.5 | -14.5 | 0 | -14.5 | -14.5 | -14.5 |
| 2004 | `model.vision_tower.encoder.layers.9.self_attn.v_proj.input_max` |  | bfloat16 | 11.38 | 11.38 | 0 | 11.38 | 11.38 | 11.38 |
| 2005 | `model.vision_tower.encoder.layers.9.self_attn.v_proj.input_min` |  | bfloat16 | 11.5 | -11.5 | 0 | -11.5 | -11.5 | -11.5 |
| 2006 | `model.vision_tower.encoder.layers.9.self_attn.v_proj.linear.weight` | 768×768 | bfloat16 | 27.71 | 1.338e-05 | 0.03609 | -0.1885 | 0.1895 | -0.01892, -0.05273, 0.07715, 0.02271, -0.01892, 0.01697, -0.104, 0.01135 |
| 2007 | `model.vision_tower.encoder.layers.9.self_attn.v_proj.output_max` |  | bfloat16 | 15.81 | 15.81 | 0 | 15.81 | 15.81 | 15.81 |
| 2008 | `model.vision_tower.encoder.layers.9.self_attn.v_proj.output_min` |  | bfloat16 | 15.94 | -15.94 | 0 | -15.94 | -15.94 | -15.94 |
| 2009 | `model.vision_tower.patch_embedder.input_proj.weight` | 768×768 | bfloat16 | 21.18 | 0.0001179 | 0.02757 | -0.3145 | 0.3379 | -0.04468, 0.04736, 0.1079, -0.03149, 0.0437, 0.08936, -0.02673, 0.04102 |
| 2010 | `model.vision_tower.patch_embedder.position_embedding_table` | 2×10240×768 | bfloat16 | 104.4 | -0.000336 | 0.02635 | -0.4141 | 0.4316 | -4.015e-13, -1.528e-13, -2.678e-15, -0.05249, -0.02478, -6.606e-15, 5.332e-08, 6.745e-15 |
