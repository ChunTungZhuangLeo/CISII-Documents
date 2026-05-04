# Test Plan
## Eye Snake Localization Using Vision

**Project:** Vision-Based Localization for Micro-Continuum Surgical Robots
**Student:** Chuntung Zhuang (czhuang3@jh.edu)
**Course:** CIS II - Spring 2026, Johns Hopkins University

---

## 1. Test Plan Overview

### 1.1 Purpose
This document defines the testing strategy to verify that the Eye Snake Localization System meets all functional requirements and achieves the target accuracy for vitreoretinal surgery applications.

### 1.2 Scope
Testing covers:
- Unit tests for individual components
- Integration tests for data pipelines
- Model accuracy tests on simulation and real data
- Performance benchmarks for real-time inference
- Sim-to-real transfer validation

### 1.3 Test Environment

| Environment | Description |
|-------------|-------------|
| **Development** | Ubuntu 20.04, Python 3.8, PyTorch 2.0, CUDA 11.8 |
| **Simulation** | AMBF simulator with I²RIS eye snake model |
| **Hardware** | NVIDIA GPU (RTX 3090 or equivalent), 32GB RAM |
| **Real Robot** | I²RIS on SHER base, Leica microscope, ROS Noetic |

---

## 2. Test Categories

### 2.1 Unit Tests

| Test ID | Component | Test Description | Pass Criteria |
|---------|-----------|------------------|---------------|
| UT-01 | Camera Projection | Verify 3D-to-2D projection math | Reprojection error < 0.1 px |
| UT-02 | Data Loader | Load simulation sample correctly | All fields present and valid |
| UT-03 | Data Loader | Load real sample correctly | Image + labels match |
| UT-04 | Augmentation | Apply brightness augmentation | Values in valid range [0, 255] |
| UT-05 | Augmentation | Apply Gaussian blur | Kernel applied correctly |
| UT-06 | Soft-Argmax | Compute soft-argmax from heatmap | Correct coordinate extraction |
| UT-07 | Loss Function | Compute MSE loss for regression | Gradient flows correctly |
| UT-08 | Loss Function | Compute heatmap loss | Valid probability distribution |
| UT-09 | Model Forward | PointNet forward pass | Output shape [B, 3, 3] |
| UT-10 | Model Forward | HRNet forward pass | Output shape [B, 3, H, W] |
| UT-11 | Model Forward | Integral Regression forward | Output shape [B, 3, 2] |
| UT-12 | Checkpoint | Save and load model weights | Weights identical after reload |

### 2.2 Integration Tests

| Test ID | Components | Test Description | Pass Criteria |
|---------|------------|------------------|---------------|
| IT-01 | Data Pipeline → Model | End-to-end training step | Loss decreases |
| IT-02 | Model → Evaluation | Inference + metric computation | Metrics computed correctly |
| IT-03 | AMBF → Data Collection | Generate sim sample with labels | Valid RGB, depth, labels |
| IT-04 | ROS → Inference | Real-time image processing | < 100ms latency |
| IT-05 | Multi-model Ensemble | Combine predictions | Averaged output valid |
| IT-06 | Config → Training | Load YAML config and train | Training runs without error |

### 2.3 Model Accuracy Tests

| Test ID | Model | Dataset | Test Description | Pass Criteria |
|---------|-------|---------|------------------|---------------|
| MA-01 | PointNet | Simulation | Evaluate on 15 test samples | Error < 15 mm |
| MA-02 | PointNet++ | Simulation | Evaluate on 15 test samples | Error < 12 mm |
| MA-03 | DGCNN | Simulation | Evaluate on 15 test samples | Error < 8 mm |
| MA-04 | HRNet-Heatmap | Simulation | Evaluate on 15 test samples | Error < 5 mm |
| MA-05 | Hybrid Fusion | Simulation | Evaluate on 15 test samples | Error < 5 mm |
| MA-06 | Transformer Fusion | Simulation | Evaluate on 15 test samples | Error < 6 mm |
| MA-07 | Integral Regression | Real (Real-Only) | 50 test frames | Error < 50 px |
| MA-08 | Integral Regression | Real (Sim→Real FT) | 50 test frames | Error < 25 px |
| MA-09 | Multi-Resolution Ensemble | Real | 50 test frames | Error < 20 px |
| MA-10 | All 13 Methods | Real | Benchmark comparison | Rankings consistent |

### 2.4 Performance Tests

| Test ID | Test Description | Pass Criteria |
|---------|------------------|---------------|
| PT-01 | Inference latency (224 px) | < 20 ms per frame |
| PT-02 | Inference latency (512 px) | < 50 ms per frame |
| PT-03 | Inference latency (1024 px) | < 100 ms per frame |
| PT-04 | GPU memory usage (batch=1) | < 4 GB |
| PT-05 | GPU memory usage (batch=8) | < 12 GB |
| PT-06 | Training throughput | > 10 samples/second |
| PT-07 | Data loading speed | < 50 ms per batch |
| PT-08 | Multi-scale inference | < 150 ms for 5 scales |

### 2.5 Sim-to-Real Transfer Tests

| Test ID | Training Regime | Test Description | Pass Criteria |
|---------|-----------------|------------------|---------------|
| ST-01 | Sim-Only | Zero-shot transfer to real | Evaluate degradation |
| ST-02 | Real-Only (N=30) | Minimal real data | Establish baseline |
| ST-03 | Real-Only (N=150) | Full real data | Best real-only result |
| ST-04 | Sim→Real FT (N=30) | Finetune with 30 real | Improvement over ST-02 |
| ST-05 | Sim→Real FT (N=150) | Finetune with 150 real | Best overall result |
| ST-06 | Resolution Comparison | 224 vs 512 vs 1024 px | Higher resolution better |
| ST-07 | Per-Sample Analysis | Hard vs easy samples | FT helps hard cases |

---

## 3. Test Procedures

### 3.1 Unit Test Procedure (UT-06: Soft-Argmax)

**Objective:** Verify soft-argmax correctly extracts coordinates from heatmap

**Setup:**
1. Create synthetic heatmap with known peak location
2. Initialize soft-argmax function

**Steps:**
1. Generate Gaussian heatmap centered at (0.3, 0.7)
2. Apply soft-argmax function
3. Compare output to expected (0.3, 0.7)

**Expected Result:**
- Output coordinates within 0.01 of expected values

**Test Code:**
```python
def test_soft_argmax():
    # Create heatmap with peak at (0.3, 0.7)
    heatmap = create_gaussian_heatmap(64, 64, center=(0.3, 0.7))

    # Apply soft-argmax
    coords = soft_argmax(heatmap.unsqueeze(0))

    # Verify
    assert abs(coords[0, 0] - 0.3) < 0.01
    assert abs(coords[0, 1] - 0.7) < 0.01
```

### 3.2 Model Accuracy Test Procedure (MA-08)

**Objective:** Validate Integral Regression achieves target accuracy with sim-to-real finetuning

**Setup:**
1. Pretrained model on 300 simulation images
2. 150 real training images, 50 real test images
3. Evaluation metrics: mean pixel error, success rate

**Steps:**
1. Load simulation-pretrained checkpoint
2. Finetune on 150 real images (200 epochs, lr=1e-4)
3. Evaluate on 50 held-out real test frames
4. Compute mean pixel error and per-keypoint breakdown

**Expected Result:**
- Mean pixel error < 25 px (approx. 0.38 mm)
- Success rate (<5mm equivalent) > 80%

**Evaluation Script:**
```bash
python evaluate.py \
    --checkpoint checkpoints/integral_reg_simreal.pth \
    --test_data data/real/test_50.json \
    --resolution 512 \
    --output results/MA-08.json
```

### 3.3 Performance Test Procedure (PT-02)

**Objective:** Verify inference meets real-time requirements at 512 px

**Setup:**
1. Trained model checkpoint
2. GPU: NVIDIA RTX 3090
3. 100 test images

**Steps:**
1. Load model to GPU
2. Warm up with 10 inference passes
3. Time 100 inference passes
4. Compute mean and std latency

**Expected Result:**
- Mean latency < 50 ms
- 99th percentile < 75 ms

**Benchmark Script:**
```bash
python benchmark.py \
    --model checkpoints/integral_reg.pth \
    --resolution 512 \
    --num_samples 100 \
    --warmup 10
```

### 3.4 Sim-to-Real Transfer Test Procedure (ST-07)

**Objective:** Analyze finetuning benefit across sample difficulty

**Setup:**
1. Real-only and Sim→Real FT models
2. 50 test samples with ground truth
3. Difficulty metric: real-only error per sample

**Steps:**
1. Evaluate real-only model, record per-sample errors
2. Evaluate finetuned model, record per-sample errors
3. Compute improvement = real_only_error - ft_error
4. Bin samples into quartiles by difficulty
5. Analyze improvement per quartile

**Expected Result:**
- Q4 (hardest) shows largest improvement
- 46+ of 50 samples improve with finetuning

---

## 4. Test Data

### 4.1 Simulation Test Set
- **Size:** 15 samples (10% of 150 train samples)
- **Visual Conditions:** Mix of 4 textures × 3 lightings
- **Pose Distribution:** Uniform coverage of workspace
- **Labels:** Automatic 3D keypoint projection

### 4.2 Real Test Set
- **Size:** 50 frames (held out from 217 total)
- **Videos:** Stratified across V1-V6 sessions
- **Pose Distribution:** Diverse bending configurations
- **Labels:** Manual 2D keypoint annotation

### 4.3 Data Split Verification
```python
def verify_no_overlap():
    """Ensure train and test sets are disjoint."""
    train_frames = load_train_set()
    test_frames = load_test_set()

    train_ids = set(f['video_id'] + '_' + str(f['frame_id']) for f in train_frames)
    test_ids = set(f['video_id'] + '_' + str(f['frame_id']) for f in test_frames)

    assert len(train_ids & test_ids) == 0, "Data leakage detected!"
```

---

## 5. Test Metrics

### 5.1 Primary Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| Mean Pixel Error | `sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)` | pixels |
| Mean 3D Error | `sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2 + (z_pred - z_gt)^2)` | mm |
| Success Rate @5mm | `count(error < 5mm) / total` | % |
| Success Rate @10mm | `count(error < 10mm) / total` | % |

### 5.2 Per-Keypoint Metrics

| Keypoint | Expected Error Range |
|----------|---------------------|
| Gripper Tip | 15-30 px (smallest, hardest) |
| First Disk | 10-25 px |
| Last Disk | 10-25 px |

### 5.3 Conversion Reference
At typical working distance of CMSR microscope:
- **1 pixel ≈ 0.015 mm**
- **21.5 px ≈ 0.32 mm**
- **16.3 px ≈ 0.24 mm**

---

## 6. Test Schedule

| Phase | Tests | Duration | Dependencies |
|-------|-------|----------|--------------|
| Week 1-2 | UT-01 to UT-12 | 2 weeks | Code complete |
| Week 3-4 | IT-01 to IT-06 | 2 weeks | Unit tests pass |
| Week 5-8 | MA-01 to MA-10 | 4 weeks | Integration tests pass |
| Week 9-10 | PT-01 to PT-08 | 2 weeks | Models trained |
| Week 11-12 | ST-01 to ST-07 | 2 weeks | Performance validated |
| Week 13-15 | Regression + Final | 3 weeks | All tests pass |

---

## 7. Acceptance Criteria

### 8.1 Minimum Viable Product
- [ ] All unit tests pass (UT-01 to UT-12)
- [ ] All integration tests pass (IT-01 to IT-06)
- [ ] At least one model achieves <5mm error on simulation (MA-04)
- [ ] Inference latency < 100ms at 512px (PT-02)

### 8.2 Target Goals
- [ ] Sim-to-real finetuning reduces error by >40% (ST-05)
- [ ] Multi-resolution ensemble achieves <20px on real (MA-09)
- [ ] 80%+ samples improve with finetuning (ST-07)

### 8.3 Stretch Goals
- [ ] Sub-millimeter accuracy (<16.3px) on real data
- [ ] Real-time ROS integration demonstrated
- [ ] All 13 benchmark methods evaluated

---

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sim-to-real gap too large | High | Domain randomization, hybrid initialization |
| Limited real data | Medium | Data augmentation, transfer learning |
| GPU memory constraints | Medium | Gradient checkpointing, mixed precision |
| Annotation errors | Medium | Cross-validation, outlier detection |
| Hardware access delays | High | Front-load simulation work |

