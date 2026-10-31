# Task F3 – Deployment Recommendation

> **Operations director's question:** *"Which model should we deploy to the 85 trucks?"*
>
> **Answer: deploy M3, the structurally pruned + full-INT8 model (`cold_chain_mlp_pruned_int8.tflite`), running on the Cortex-A76 CPU of the Raspberry Pi 5 + Hailo AI HAT edge node, behind the temperature rule in the inference service. Keep M2 (`cold_chain_mlp_ptq_int8.tflite`) in the same image as the rollback.**
>
> M3 passes all three deployment gates. Its estimated p95 latency on the Pi 5 (about 0.4 ms) is 0.4 % of the 100 ms per-inference target, it fits the device's storage and on-chip memory with more than 99 % headroom, and it detects 17/17 Critical validation samples (100 % recall, gate > 95 %). It has the smallest file and memory footprint of the three, ties M2 on latency, and loses no accuracy. On its own, though, no model flags a temperature breach without a vibration fault (§4), so the deployed service adds a temperature rule after the model.

---

## 1. Evidence – Task F2 benchmark results

Measured with `optimisation/benchmark.py` (Lab 2 methodology): batch size 1, 10 warm-up runs excluded, 200 timed runs. Accuracy is measured on the shared held-out validation split (59 samples: 24 Normal / 18 Warning / 17 Critical). Energy follows Lab 2 Step 3: E = P × t, with P = the laptop's 15 W TDP for every variant, a conservative upper bound. psutil CPU % is recorded alongside, as in Lab 2, but not multiplied in.

**Machine:** HP ProBook 440 G9, Intel Core i7-1255U (15 W TDP, 12 logical CPUs), on AC power, Windows 11, Python 3.13.5, TensorFlow 2.21.0. Raw output: `optimisation/results/benchmark_results.json`; the same table as CSV with Pareto flags is `optimisation/results/benchmark_results.csv`, and the annotated Pareto chart is `optimisation/results/pareto_chart.png`.

| Variant | Mean latency (ms) | p95 latency (ms) | File size (KB) | Accuracy (%) | Energy / inference (mJ) | Critical recall |
|---|---:|---:|---:|---:|---:|---:|
| M1 – FP32 baseline | 2.2606 | 3.4425 | 34.46 ¹ | 100.00 | 33.90826 | 100 % (17/17) |
| M2 – PTQ INT8 | 0.0145 | 0.0149 | 4.58 | 100.00 | 0.21740 | 100 % (17/17) |
| **M3 – Pruned 35 % + PTQ INT8** | **0.0200** | **0.0359** | **3.75** | **100.00** | **0.30054** | **100 % (17/17)** |

¹ M1's size is the `.keras` file that `inference_service.py` loads. The same model exported as a TF SavedModel, the format Task F1 names, is **51.43 KB**. M1 is the largest variant either way.

Supporting measurements:

| Variant | Parameters | Model RAM – weights + activations (KB) | Notes |
|---|---:|---:|---|
| M1 | 803 | 3.36 | Needs the full TensorFlow runtime. Through `model.predict()`, the call `inference_service.py` actually makes, latency is **63.9 ms mean / 73.6 ms p95** |
| M2 | 803 | 0.99 | INT8 weights and activations, INT32 biases |
| M3 | 400 | 0.53 | Hidden layers 32→21 and 16→10 neurons (35.4 % of neurons removed, 50.2 % of parameters) |

**Why M3's structured pruning is custom code.** `tfmot`'s `prune_low_magnitude` prunes individual weights: it sets them to zero but keeps them, so the dense layers stay 32 and 16 wide and the INT8 file does not get smaller. Its only structured option is 2:4 sparsity (2 zeros in every block of 4 weights), which also keeps every neuron. Neither removes whole neurons, which is what "structured filter pruning" means for an MLP. `training/prune_quantise.py` therefore takes the PolynomialDecay schedule (0 % → 35 %) from `tfmot` and applies it to whole neurons, removing those with the smallest incoming-weight L2 norm. It then rebuilds a physically smaller network, checks it gives the same outputs as the masked one (max difference < 1e-5), and quantises it to full INT8. That is why M3 has half of M1's parameters and the smallest file.

What the numbers say:
- **Accuracy doesn't separate the variants.** Quantisation and pruning cost no accuracy here: INT8 and FP32 predictions agree on 100 % of validation samples for both M2 and M3. The validation classes are easy to separate, though, and §4 shows a case outside them where M3 behaves differently.
- **M1 is the slow, large one.** The INT8 forward passes are more than 100× faster than M1's (0.015–0.020 ms against 2.26 ms), and M3's file is 9.2× smaller than M1's.
- **Between M2 and M3, size decides.** M3 is 18 % smaller and uses 46 % less model RAM. Their latencies are within run-to-run noise: M2 was faster in this run, M3 in the previous one (§6).
- **Energy follows latency.** Every variant is charged the same 15 W, so energy ranks exactly like latency, and M2 and M3 are tied for the same reason.

---

## 2. Gate 1 – The 90-second SLA as a per-inference latency budget

The SLA requires a refrigeration fault to raise an alert within **90 s** of onset. Most of that time is consumed by how the edge service (`inference/inference_service.py`) builds its input, not by the model:

| Stage (worst case) | Source | Time |
|---|---|---:|
| Smoothing filter flushes pre-fault values (5-sample moving average on the 0.5 Hz vibration stream) | `vib_buffer = deque(maxlen=5)` | 10 s |
| Feature window fills with post-fault data | `WINDOW_SIZE_SEC = 30.0` | 30 s |
| Wait for the next evaluation step (fault lands just after a step) | `STEP_SIZE_SEC = 10.0` | 10 s |
| **Total before the model sees a fully post-fault window** | | **50 s** |
| **Remaining for inference + local MQTT publish + alarm** | 90 − 50 | **40 s** |

That gives the per-inference budget at three levels:

| Level | Budget | Why |
|---|---:|---|
| SLA residual | 40 s | Time left in the 90 s SLA after windowing |
| Hard ceiling | **10 s** | The service evaluates every 10 s on one thread; a slower inference would queue and push later alerts past the SLA |
| Engineering target (Task A1) | **100 ms** | "Local inference latency < 100 ms" |

| Variant | p95 on laptop | p95 on Pi 5, assuming 10× slower (to be verified on device) | Share of the 100 ms target used (Pi estimate) | Meets SLA? |
|---|---:|---:|---:|---|
| M1 – forward pass | 3.44 ms | ~34 ms | 34 % | ✅ |
| M1 – via `model.predict()` (as deployed) | 73.6 ms | **~736 ms** | **~736 % – misses the target** | ⚠️ meets the 10 s ceiling, misses the 100 ms target |
| M2 | 0.0149 ms | ~0.15 ms | 0.15 % | ✅ |
| **M3** | **0.0359 ms** | **~0.36 ms** | **0.36 %** | ✅ |

The **10× slowdown** is a deliberately pessimistic assumption for the Pi 5's Cortex-A76 against the laptop's i7-1255U. The pilot should re-run `optimisation/benchmark.py` on one Pi to replace it with measured figures. Conclusion: inference is less than 0.01 % of the 90 s SLA for the INT8 models. M1, as the service actually runs it, is the only variant that risks missing the 100 ms target on the slower Pi CPU.

---

## 3. Gate 2 – Hardware storage ("Flash") and on-chip memory ("SRAM") constraints

The device chosen in Task B1 is the **Raspberry Pi 5 + Hailo AI HAT**. It is a Linux SBC (single-board computer), not an MCU, so the Flash/SRAM constraints map as follows:

| MCU term | Pi 5 equivalent | Capacity |
|---|---|---|
| Flash (stores the model) | microSD card. The Pi 5's single PCIe lane is taken by the Hailo HAT, so there is no NVMe drive | GBs |
| SRAM (holds weights + activations while running) | On-chip cache of the BCM2712 SoC: **512 KB L2 per Cortex-A76 core**, 2 MB shared L3. Backed by 4–8 GB LPDDR4X | 512 KB L2 |

| Variant | File size vs microSD | Model RAM vs 512 KB L2 | Runtime the model needs in the container |
|---|---|---|---|
| M1 | 34.46 KB – fits | 3.36 KB (0.66 %) – fits | **Full TensorFlow (hundreds of MB)** |
| M2 | 4.58 KB – fits | 0.99 KB (0.19 %) – fits | TFLite interpreter only |
| **M3** | **3.75 KB – fits** | **0.53 KB (0.10 %) – fits** | **TFLite interpreter only** |

**M3's entire working set fits in one core's L2 cache with more than 99.8 % headroom.** On the Pi, the real footprint cost is the runtime, not the model. Only M1 needs full TensorFlow; M2 and M3 can be served by the much smaller TFLite/LiteRT interpreter. Today's image ships full TensorFlow so that the CI test can switch all three variants through `MODEL_PATH`. Moving to a LiteRT-only image is a follow-up optimisation once M1 is retired.

**Cross-check against the MCU rejected in Task B1 (STM32H743: 2 MB Flash, 1 MB SRAM):** M3 would use 0.18 % of Flash and 0.05 % of SRAM. The INT8 models would even fit on a microcontroller. The Pi was chosen for Docker, OTA and Ansible support, not because of model size.

**About the Hailo NPU:** the Hailo-8/8L runs models compiled to `.hef` with Hailo's Dataflow Compiler; it does not run `.tflite` files. M3 has 400 parameters and runs in about 0.02 ms on CPU, so offloading it would gain nothing. **Run M3 on the A76 CPU** and keep the NPU for future heavier models, such as camera-based cargo inspection.

This refines the Task B2 recommendation in `hardware/hardware_justification.md`. B2's "Hailo NPU Acceleration" was worked out for the brief's given model of 45 MFLOPs and 18 MB. The trained MLP is far smaller: about 1.5 kFLOPs per inference for M1 and 0.7 kFLOPs for M3, with M3's weights and activations taking 0.53 KB. B2's other two optimisations, INT8 quantisation and structured pruning, are applied as M2 and M3.

**Power, for completeness:** at one inference per 10 s (8,640 per day), M3 uses about 2.6 J/day and M1 about 293 J/day, even at the full 15 W upper bound. Both are negligible against the node's ~648 kJ/day draw at 7.5 W. Power does not separate the variants.

---

## 4. Gate 3 – Class 2 (Critical) recall on the recommended variant

| Variant | Critical samples detected | Critical recall | Gate (> 95 %) |
|---|---:|---:|---|
| **M3** | **17 / 17** | **100 %** | ✅ **PASS** |

The gate is enforced, not just reported. `training/prune_quantise.py` exits with an error, and CI fails, if the INT8 M3 model's Critical recall is ≤ 95 % or its accuracy is ≤ 88 %.

**Limitation the director should know about:** the validation set holds only **17 Critical samples**. A single miss would drop recall to 94.1 % and fail the gate, so in practice the gate demands zero misses. On so few samples, 17/17 supports a 95 %-confidence lower bound on true recall of only **83.8 %** (Clopper–Pearson). Showing that true recall exceeds 95 % with 95 % confidence needs **at least 59 consecutive Critical events detected with zero misses**. That should be an explicit exit criterion for the pilot (§5).

**What the 17/17 does and does not show.** The dataset follows the brief's recipe: Warning windows come from `--anomaly temp_drift`, and Critical windows from `--anomaly combined`, which is the same drift plus the bearing-wear vibration step. At +0.08 °C per reading the drift passes +3 °C after about 40 s, so almost every Warning window already has a Critical-level temperature (median window mean 38.6 °C). The temperature inputs are also clipped (next paragraph). Warning and Critical windows therefore look the same to the model on temperature, and only the vibration features tell them apart (median RMS 0.45 g against 1.21 g). The 17/17 shows that M3 reliably recognises the refrigeration-unit failure signature. It does not show that M3 flags a temperature breach on its own.

A check with a steady (no longer drifting) cargo temperature and normal vibration confirms the gap:

| Steady cargo temperature, normal vibration | Brief's class | M1 | M2 | M3 |
|---|---|---|---|---|
| 4.0 °C (setpoint) | Normal | Normal | Normal | Normal |
| 5.5 °C | Warning | Warning | Warning | **Normal** |
| 8 – 30 °C | Critical | Warning | Warning | **Normal** |
| 0 °C (freezing) | Critical | Normal | Normal | Normal |

M1 and M2 move to Warning by 4.6 °C but never reach Critical. M3 stays Normal at every steady temperature tested: after pruning it responds to the temperature *rising*, not to the temperature *being high*. No variant detects a cold excursion.

**Temperature rule in the inference service.** Because no model enforces the brief's temperature bands on its own, the service applies a rule after the model (`inference/temperature_rule.py`, tested in `tests/test_temperature_rule.py`). It takes the window's mean temperature and raises the model's class to at least Warning when that is more than 1 °C from setpoint, and to Critical when it is more than 3 °C from setpoint, on either side. The rule never lowers the model's class, so the vibration fault is still caught by the model. The setpoint is the `TEMP_SETPOINT_C` environment variable (default 4.0 °C). Each MQTT payload carries `status` (model + rule) and `model_status` (model alone); `confidence` stays the model's, so the E1 PSI monitor still watches the model.

| Deployed system: M3 + rule | Result |
|---|---|
| Steady 5.5 °C / 8–30 °C / 0 °C, normal vibration | Warning / Critical / Critical, as the brief requires |
| Validation Critical windows | 17/17 Critical (unchanged) |
| Validation Normal windows | 24/24 Normal: no false alarms |
| Validation Warning windows | 2 stay Warning, 16 are raised to Critical |

The 16 raised windows have a mean temperature more than 3 °C above setpoint, so the brief's own class table calls them Critical; the dataset labels them Warning only because they came from the `temp_drift` run. M2 + rule gives the same results. A longer-term fix is to label training windows by the brief's temperature bands instead of by simulator mode (Task D1).

**Quantisation input range.** The INT8 input quantiser covers normalised values from −2.685 to +9.996, one scale shared by all six features. For temperature mean (training mean 4.01 °C, scale floored at 0.1 °C) that is only **3.74 °C to 5.01 °C**:
- **Above setpoint**, the limit equals the ±10 clip the service applies before any model runs, so INT8 loses nothing extra: all three variants see 5 °C and 70 °C as the same value. 99 % of Warning and 98 % of Critical windows sit at this limit, and the temperature rate-of-change is clipped the same way (98–100 %).
- **Below setpoint**, INT8 saturates at 3.74 °C, while FP32 M1 sees down to 3.01 °C. A cold excursion such as freezing, which the brief's "> 3 °C outside setpoint" also counts as Critical, is invisible to M2 and M3. The training data contains no cold excursions either.
- **Vibration** stays inside the range: the Critical step to about 1.2 g RMS normalises to about +7.6.

---

## 5. Decision and rollout

| Criterion | M1 FP32 | M2 PTQ INT8 | **M3 Pruned + INT8** |
|---|---|---|---|
| 90 s SLA / 100 ms per-inference target | ⚠️ ~74 ms p95 on laptop via `predict()`; likely over target on Pi | ✅ | ✅ (tied with M2) |
| Storage (microSD) | ✅ 34.46 KB | ✅ 4.58 KB | ✅ **3.75 KB** (smallest) |
| On-chip memory (512 KB L2) | ✅ 3.36 KB | ✅ 0.99 KB | ✅ **0.53 KB** (smallest) |
| Runtime footprint | ❌ full TensorFlow | ✅ TFLite only | ✅ TFLite only |
| Validation accuracy > 88 % | ✅ 100 % | ✅ 100 % | ✅ 100 % |
| Critical recall > 95 % | ✅ 100 % | ✅ 100 % | ✅ **100 % (17/17)** |
| Temperature-only breach (§4) | ⚠️ model: Warning at most; ✅ with rule | ⚠️ model: Warning at most; ✅ with rule | ⚠️ model: Normal; ✅ with rule |
| **Decision** | Reject | **Rollback model** | **Deploy, with the temperature rule** |

**Why M3 over M2:** they are tied on accuracy, Critical recall and latency (which of the two is faster changes from run to run, §6). M3 is smaller (−18 % file, −46 % RAM, −50 % parameters). On a steady temperature breach M3 alone is weaker than M2 (Normal instead of Warning, §4), but the temperature rule covers that case for every model, so it does not decide between them. **Why keep M2:** pruning removed 35 % of the hidden neurons, which leaves the model less spare capacity if field data drifts. Rolling back needs no image rebuild. The E2 playbook copies whichever file `model_src` names to `/opt/logibridge/model.tflite` and restarts the container on it, so re-running the playbook with `-e model_src=<path to cold_chain_mlp_ptq_int8.tflite>` swaps in M2. M2 also ships inside the image, so a plain `docker run` can switch with `-e MODEL_PATH=cold_chain_mlp_ptq_int8.tflite`.

**Rollout plan for the 85-truck pilot:**
1. Deploy with the Ansible playbook (Task E2). Its `model_src` defaults to M3 (`cold_chain_mlp_pruned_int8.tflite`), and the image's own `MODEL_PATH` default is M3 too. The temperature rule runs in the same container; its setpoint is `TEMP_SETPOINT_C` (default 4.0 °C).
2. On one truck, re-run `optimisation/benchmark.py` on the Pi 5 to replace the 10× latency assumption with measured figures.
3. Run the PSI drift monitor (Task E1) on every truck, with `reference_dist.json` built from M3's outputs. If drift alerts persist, roll back to M2 by re-running the playbook with `-e model_src=<path to cold_chain_mlp_ptq_int8.tflite>` and an M2 reference distribution.
4. **Pilot exit criterion:** at least 59 field-confirmed Critical events detected with zero misses, which demonstrates Critical recall > 95 % at 95 % confidence. The events must include temperature-only breaches (normal vibration) and at least one cold excursion, which only the temperature rule catches (§4). Only then extend to the full 265-truck fleet.

---

## 6. Method notes and limitations

- **Laptop, not target hardware.** All latencies were measured on an x86 i7-1255U. Pi 5 figures in §2 are estimates that assume a 10× slowdown.
- **Run-to-run variance.** Across runs on AC power, M1's mean latency ranged from 2.10 to 3.80 ms (eager TensorFlow overhead varies) and the INT8 models from 0.010 to 0.022 ms. M1 is always far slower, but which of M2 and M3 is faster changes between runs: M3 in the previous run (0.0096 against 0.0110 ms), M2 in this one (0.0145 against 0.0200 ms). Treat their latency, and therefore their energy, as tied.
- **Energy is a conservative upper bound.** Following Lab 2 Step 3, every variant is charged the full 15 W TDP. psutil puts this process at only 5–8 % of one logical CPU during back-to-back inference (and under-reads even a pure busy loop on this laptop), so real energy per inference is lower. Because P is the same for every variant, energy ranks exactly like latency.
- **The benchmark measures the models alone.** The temperature rule (§4) adds two comparisons per window and does not change latency measurably.
- **Small, synthetic validation set** (59 samples) – see the confidence bound in §4.

**Reproduce:**

```
.\.venv\Scripts\python.exe training\train_model.py
.\.venv\Scripts\python.exe training\convert_ptq.py
.\.venv\Scripts\python.exe training\prune_quantise.py
.\.venv\Scripts\python.exe optimisation\benchmark.py
.\.venv\Scripts\python.exe optimisation\pareto.py
```
