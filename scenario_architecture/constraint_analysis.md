# Task A1 – Constraint Analysis

## 1. Latency Constraint

The primary business requirement of FreightBridge is the detection of refrigeration anomalies within **90 seconds** of fault occurrence. A refrigeration failure can increase cargo temperature by approximately **1°C per minute**, meaning that after three minutes a pharmaceutical shipment may already enter the **Critical** state. The project specifies that alerting must occur within **90 seconds**.

According to the Edge AI Foundations module, cloud round-trip latency typically ranges from **100 ms to 500 ms** and becomes highly unreliable in poor-connectivity environments. Edge inference, however, generally operates within **1–15 ms**, while cloud latency ranges between **50–500 ms**.

The major challenge is not computational latency but connectivity latency. FreightBridge reports connectivity gaps of **35–90 minutes** on several highway segments. During these outages, a cloud-only architecture would be incapable of receiving sensor data, performing inference, or generating alerts. Consequently, cloud-only AI cannot satisfy the 90-second SLA.

An edge-based approach allows local inference every few seconds and immediate local alarm generation regardless of network availability.

### Therefore, Edge AI is mandatory because:

- Local inference latency < 100 ms
- Alert generation independent of WAN connectivity
- Compliance with the 90-second fault detection requirement

---

## 2. Bandwidth Constraint

Each truck generates the following data volume:

| Data Stream | Frequency / Event Rate | Calculations | Daily Volume |
|------------|------------------------|--------------|-------------|
| Temperature Sensor | 1 Hz | 86,400 samples/day × 8 bytes/sample = 691,200 bytes | 0.66 MB/day |
| Vibration Sensor (3-axis Accelerometer) | 500 Hz | Sample size = 3 axes × 4 bytes (Float32) = 12 bytes/sample<br>Daily samples = 500 × 86,400 = 43,200,000 samples/day<br>Daily volume = 43,200,000 × 12 = 518,400,000 bytes | 494 MB/day |
| Door Events | Assume 100 events/day | 100 × 32 bytes/event = 3,200 bytes | 0.003 MB/day |

### Total Raw Data Per Truck

- Total = 0.66 + 494 + 0.003
- **Total Raw Data per Truck = 494.66 MB/day**

### Total Raw Data for Fleet (85 Trucks)

- 494.66 × 85
- **Total Fleet Data = 42.0461 GB/day**

---

## 3. Transmission Cost Analysis

| Metric | Calculation | Value |
|----------|------------|---------|
| Transmission Cost Rate | Given | ₹0.10/MB |
| Daily Cost per Truck | 494.66 × ₹0.10 | ₹49.46/day |
| Daily Cost for 85 Trucks | ₹49.46 × 85 | ₹4,204/day |
| Annual Fleet Cost | ₹4,204 × 365 | ₹15.34 lakh/year |

Using edge processing, only alerts and summaries are transmitted, reducing bandwidth by 
