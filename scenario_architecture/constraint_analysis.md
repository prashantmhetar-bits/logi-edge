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

## 3. Transmission Cost

| Metric | Calculation | Value |
|----------|------------|---------|
| Transmission Cost Rate | Given | ₹0.10/MB |
| Daily Cost per Truck | 494.66 × ₹0.10 | ₹49.46/day |
| Daily Cost for 85 Trucks | ₹49.46 × 85 | ₹4,204/day |
| Annual Fleet Cost | ₹4,204 × 365 | ₹15.34 lakh/year |


### Analysis : Bandwidth & Cost

Assuming **200 alerts/day** and an average alert size of **200 bytes**:

- Data transmitted per day = 200 × 200 bytes
- = 40,000 bytes/day
- ≈ 39.06 KB/day
- ≈ 0.04 MB/day

### Comparison of Deployment Strategies

| Deployment Strategy | Data Transmitted | Cost Impact |
|--------------------|------------------|-------------|
| Cloud-Only Raw Streaming | 494.66 MB/day/truck | ₹15.34 lakh/year for 85 trucks |
| Edge AI (Alert-Based Transmission) | ~40 KB/day/truck (0.04 MB/day) | Negligible transmission cost (~₹0.004/day) |
| Bandwidth Reduction | 494.66 MB/day → 0.04 MB/day | **99.99% reduction** |

### Key Observation

The Edge AI architecture reduces network traffic from **494.66 MB/day** to **0.04 MB/day** per truck, resulting in a **99.99% reduction in bandwidth consumption**. This significantly lowers communication costs while ensuring only meaningful events are transmitted to the cloud.

---

## 4. Connectivity Constraint

FreightBridge experiences **35–90 minute cellular outages** on multiple transportation routes. The project requirements explicitly state that a cloud-only operation is unacceptable under these conditions.

### Limitations of a Cloud-Only Architecture

A cloud-only solution would:

1. Stop receiving sensor data during network outages.
2. Miss developing refrigeration faults.
3. Fail chain-of-custody logging requirements.
4. Delay critical alerts until connectivity is restored.

### Proposed Edge AI Solution

The proposed Edge AI architecture addresses these challenges by:

1. Running a local MQTT broker on the edge device.
2. Performing machine learning inference locally.
3. Storing alerts and events in SQLite.
4. Forwarding queued alerts when connectivity resumes.

This design follows the **Store-and-Forward architecture**, where alerts are buffered locally during connectivity interruptions and synchronized with the cloud once network access is restored. As a result, continuous monitoring and fault detection remain operational despite intermittent cellular coverage.

---

## 5. Privacy Constraint

Cold-chain pharmaceutical shipments contain temperature compliance records that form part of regulated chain-of-custody documentation.

### Privacy Benefits of Edge AI

The proposed Edge AI solution improves privacy and data governance by ensuring that:

1. Raw vibration streams remain on the device.
2. Temperature history remains local.
3. Only classified states and alerts are transmitted.
4. Reduced data exposure lowers interception and leakage risks.

Edge processing supports data sovereignty by retaining sensitive operational data on the vehicle while transmitting only alerts and summarized metadata to cloud services. Consequently, the organization maintains greater control over operational information and reduces cybersecurity and compliance risks.

This architecture therefore supports FreightBridge's contractual and regulatory obligations by ensuring that sensitive pharmaceutical logistics data remains under organizational control while still enabling centralized monitoring and reporting.



