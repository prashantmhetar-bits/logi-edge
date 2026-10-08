# Task B1 – Constraint Triangle Application

## Dominant Constraint

Cold-chain deployment prioritizes:

1. Reliability
2. Latency
3. Fleet deployment cost

**Dominant Vertex:** Compute Performance balanced with Cost.

Power remains important, but electrical power is available from the truck's onboard system.

## Hardware Comparison

| Parameter | Raspberry Pi 5 + Hailo AI HAT | Jetson Orin Nano Super | STM32H7 Custom MCU |
|------------|-------------------------------|-------------------------|--------------------|
| Cost per Truck | ₹15,000 | ₹45,000 | ₹3,500 |
| Power Consumption | 7.5 W | 15 W | 0.4 W |
| Fits 10 W Power Budget? | ✅ Yes | ❌ No | ✅ Yes |
| AI Performance | Suitable for LogiEdge workload | 67 TOPS | Limited |
| Linux Support | ✅ Yes | ✅ Yes | ❌ Limited |
| Docker Support | ✅ Yes | ✅ Yes | ❌ Difficult |
| MQTT Support | ✅ Yes | ✅ Yes | ⚠ Requires custom implementation |
| OTA Deployment Support | ✅ Yes | ✅ Yes | ❌ Difficult |
| TensorFlow Lite Support | ✅ Yes | ✅ Yes | ⚠ Limited |
| Ansible Support | ✅ Yes | ✅ Yes | ❌ Not practical |
| Future MLOps Expansion | ✅ Strong | ✅ Strong | ❌ Limited |
| Memory Capacity | Sufficient for future model upgrades | Excellent | Limited |
| Pilot Fleet Cost (85 Trucks) | ₹12.75 lakh | ₹38.25 lakh | ₹2.98 lakh |
| Full Fleet Cost (265 Trucks) | ₹39.75 lakh | ₹1.19 crore | ₹9.28 lakh |
| Major Advantages | Linux ecosystem, Docker, MQTT, OTA support, balanced cost-performance ratio | Extremely high compute capability, excellent scalability | Lowest cost and lowest power consumption |
| Major Drawbacks | Higher cost than MCU solution | High cost, high power consumption, exceeds 10 W budget | Difficult Docker deployment, difficult OTA updates, limited Linux ecosystem, limited MLOps support |
| Overall Assessment | Best balance across all Constraint Triangle dimensions | Over-engineered for a lightweight MLP classifier | Technically feasible but difficult to maintain and scale |

---

# Final Recommendation

| Recommended Hardware | Justification |
|----------------------|---------------|
| **Raspberry Pi 5 + Hailo AI HAT** | • Meets the 90-second latency requirement<br>• Meets the 10 W power budget<br>• Supports offline operation<br>• Supports MQTT-based communication<br>• Supports Docker containerization<br>• Supports Ansible-based deployment and orchestration<br>• Supports future MLOps expansion and fleet-scale management |

---

## Rejection Summary

| Hardware | Reason for Rejection |
|-----------|---------------------|
| Jetson Orin Nano | Rejected due to excessive cost, excessive power consumption (15 W), and over-specification for the lightweight MLP workload. |
| STM32H7 MCU | Rejected due to deployment complexity, limited lifecycle management capabilities, lack of Docker support, limited OTA flexibility, and constrained MLOps scalability. |

---

# Task B2 – Arithmetic Intensity and Roofline Analysis

## Computation Summary

| Parameter | Formula / Calculation | Result |
|------------|----------------------|--------|
| Model FLOPs | Given | 45 MFLOPs (45,000,000 FLOPs) |
| Memory Access | Given | 18 MB (18,000,000 Bytes) |
| Arithmetic Intensity | 45,000,000 ÷ 18,000,000 | 2.5 FLOPs/Byte |
| Peak Compute Performance | Given | 16 GFLOPs/s |
| Memory Bandwidth | Given | 12 GB/s |
| Ridge Point | 16 ÷ 12 | 1.33 FLOPs/Byte |
| Roofline Comparison | Actual AI (2.5) > Ridge Point (1.33) | Compute-Bound Workload |
| Roofline Classification (Compute-Bound) | Based on Arithmetic Intensity > Ridge Point | **Primary Bottleneck:** Compute throughput rather than memory bandwidth |

# Recommended Optimizations for a Compute-Bound Model

| Optimization Technique | Purpose |
|------------------------|---------|
| INT8 Quantization | Reduces computational workload and model size |
| Structured Pruning | Removes less important parameters to reduce FLOPs |
| Hailo NPU Acceleration | Offloads inference from CPU to dedicated AI accelerator |
| TensorFlow Lite Optimization | Enables efficient edge-device execution |

# Expected Benefits

| Performance Metric | Expected Improvement |
|-------------------|----------------------|
| Inference Latency | 2× to 4× lower latency |
| Model Size | Approximately 4× smaller |
| Energy Consumption | Lower energy per inference |
| Edge Deployment Efficiency | Improved throughput and resource utilization |

# Final Conclusion

| Item | Result |
|------|--------|
| Arithmetic Intensity (AI) | 2.5 FLOPs/Byte |
| Ridge Point | 1.33 FLOPs/Byte |
| Classification | Compute-Bound |
| Recommended Deployment Optimizations | INT8 Quantization, Structured Pruning, Hailo NPU Acceleration, TensorFlow Lite Optimization |
| Expected Outcome | Reduced latency, smaller model size, and lower power consumption while meeting LogiEdge real-time inference requirements |
