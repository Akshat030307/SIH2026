# Edge inference benchmark

Model sizes: d3qn_fp32.onnx 601 KiB, d3qn_int8.onnx 171 KiB

FP32 vs INT8 greedy-action agreement on 12935 recorded states: **0.764**

## Latency per decision (µs)

| path | p50_us | p99_us |
|---|---|---|
| PyTorch CPU | 155.4 | 1169.6 |
| ONNX Runtime FP32 (1 thread) | 93.6 | 143.6 |
| ONNX Runtime INT8 (1 thread) | 197.2 | 270.2 |
| Full decision (features + INT8), S2_dense | 755.7 | 1680.2 |

## Scheduling metrics, FP32 vs INT8 (seed 150)

| scenario | model | pd | pd_weighted |
|---|---|---|---|
| S1_sparse | fp32 | 0.643 | 0.705 |
| S1_sparse | int8 | 0.929 | 0.919 |
| S3_mfr | fp32 | 0.729 | 0.762 |
| S3_mfr | int8 | 0.704 | 0.736 |
| S6_lockin | fp32 | 0.883 | 0.867 |
| S6_lockin | int8 | 0.745 | 0.793 |
**Recommendation:** deploy the FP32 ONNX policy. On this ~150k-parameter network, dynamic INT8
is slower on CPU (quantize/dequantize overhead dominates) and changes a noticeable share of greedy
actions. The decision loop is dominated by Python feature extraction, which is the part to port
to C++/FPGA for microsecond-level budgets.
