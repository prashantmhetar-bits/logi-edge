Task A1 – Constraint Analysis
-----------------------------
1. Latency Constraint
   
The primary business requirement of FreightBridge is detection of refrigeration anomalies within 90 seconds of fault occurrence. A refrigeration failure can increase cargo temperature by approximately 1°C per minute, meaning that after three minutes a pharmaceutical shipment may already enter the Critical state.  Project specifies that alerting must occur within 90 seconds. 
According to the Edge AI foundations module, cloud round-trip latency typically ranges from 100 ms to 500 ms and becomes highly unreliable in poor-connectivity environments. Edge inference, however, generally operates within 1-15 ms while cloud latency ranges between 50-500 ms.
The major challenge is not computational latency but connectivity latency. FreightBridge reports connectivity gaps of 35-90 minutes on several highway segments. During these outages, a cloud-only architecture would be incapable of receiving sensor data, performing inference, or generating alerts. Consequently, cloud-only AI cannot satisfy the 90-second SLA. An edge-based approach allows local inference every few seconds and immediate local alarm generation regardless of network availability.
Therefore, Edge AI is mandatory because:
	•	Local inference latency < 100 ms
	•	Alert generation independent of WAN connectivity
	•	Compliance with 90-second fault detection requirement

2. Bandwidth Constraint: Each truck generates:

Data Stream						Frequency / Event Rate		Calculations																													Daily Volume
-----------------------------------------------------------------------------------------------------------------------------------
Temperature Sensor		1 Hz											86,400 samples/day × 8 bytes/sample = 691,200 bytes										0.66 MB/day

Vibration Sensor 			500 Hz										Sample size = 3 axes × 4 bytes (Float32) = 12 bytes/sample
(3-axis Accelerometer)													Daily samples = 500 × 86,400 = 43,200,000 samples/day
																								Daily volume = 43,200,000 × 12 = 518,400,000 bytes										494 MB/day

Door Events																			Assume 100 events/day	100 × 32 bytes/event = 3,200 bytes							0.003 MB/day

------------------------------------------------------------------------------------------------------------------------------------
Total Raw Data per Truck																																				0.66 + 494 + 0.003	=	494.66 MB/day
Total Raw data 																																												 496.66 × 85	=	42.0461GB/day
------------------------------------------------------------------------------------------------------------------------------------

3. Transmission Cost Analysis:

Metric																						Calculation																														Value
---------------------------------------------------------------------------------------------------------------------------------
Transmission Cost Rate														Given																																	₹0.10/MB
Daily Cost per Truck															494.66 × ₹0.10																												₹49.46/day
Daily Cost for 85 Trucks													₹49.46 × 85																														₹4,204/day
Annual Fleet Cost																	₹4,204 × 365																													₹15.34 lakh/year

-----------------------------------------------------------------------------------------------------------------------------------------

Using edge processing we shall transmit only alerts and summaries, reducing bandwidth by more than 99%.
Assume: 200 alarts/day and 200 bytes/alert i.e 40KB/day. 

Deployment Strategy																	Data Transmitted																Cost Impact
---------------------------------------------------------------------------------------------------------------------------------------
Cloud-Only Raw Streaming														494.66 MB/day/truck															₹15.34 lakh/year for 85 trucks
Edge AI (Alert-Based Transmission)									~40 KB/day/truck																Negligible transmission cost ₹ 0.004/day
Bandwidth Reduction																	494.66 MB/day → 0.04 MB/day											>99.99% reduction

-----------------------------------------------------------------------------------------------------------------------------------------

4. Connectivity Constraint

FreightBridge experiences 35-90 minute cellular outages on multiple routes. Project explicitly states that cloud-only operation is unacceptable.
A cloud-only architecture would:
	•	Stop receiving sensor data
	•	Miss developing refrigeration faults
	•	Fail chain-of-custody logging
	•	Delay alerts until connectivity restoration
The proposed edge architecture solves this by:
	•	Running local MQTT broker
	•	Performing inference locally
	•	Storing alerts in SQLite
	•	Forwarding queued alerts when connectivity resumes
This approach follows the Store-and-Forward architecture explains local queueing and delayed synchronization under intermittent connectivity

5. Privacy Constraint
   
Cold-chain pharmaceutical shipments contain temperature compliance records that form part of regulated chain-of-custody documentation.
Edge AI improves privacy because:
	•	Raw vibration streams remain on device
	•	Temperature history remains local
	•	Only classified state and alerts are transmitted
	•	Reduced exposure surface lowers interception risk
Module 1 specifies that edge processing supports data sovereignty by ensuring sensitive data remains local and only metadata is transmitted upstream.
This architecture therefore supports FreightBridge's contractual commitment to pharmaceutical customers by ensuring operational data remains under organizational control.
