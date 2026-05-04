# Functional Specifications
## Eye Snake Localization Using Vision

**Project:** Vision-Based Localization for Micro-Continuum Surgical Robots
**Student:** Chuntung Zhuang (czhuang3@jh.edu)
**Mentors:** Adnan Munawar, Iulian Iordachita, Mojtaba Esfandiari
**Course:** CIS II - Spring 2026, Johns Hopkins University

---

## 1. System Overview

The Eye Snake Localization System is a vision-based software pipeline that predicts the 3D position of the I²RIS (Integrated Intraocular Robotic System) snake robot's gripper tip and key segments from camera imagery during vitreoretinal surgery.

### 1.1 Purpose
Provide real-time, accurate localization of the snake robot tip to enable:
- Safe teleoperation with visual feedback
- Future autonomous or semi-autonomous surgical assistance
- Independent position estimates that supplement encoder-based tracking

### 1.2 Clinical Context
Vitreoretinal surgery requires sub-millimeter precision. Current encoder-based approaches suffer from:
- Cable slack and hysteresis in tendon-driven mechanisms
- Calibration drift over time
- Accumulated kinematic errors

Vision-based localization bypasses these issues by directly observing the robot's position.

---

## 2. Functional Requirements

### 2.1 Core Localization Functions

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Predict 3D gripper tip position from RGB images | Must Have | Mean error <5mm on simulation test set |
| FR-2 | Predict positions of first disk (segment 11) and last disk (segment 1) | Must Have | 3 keypoints tracked simultaneously |
| FR-3 | Support multiple input modalities (RGB, point cloud, multimodal) | Must Have | At least 3 model architectures implemented |
| FR-4 | Achieve real-time inference (>10 Hz) | Must Have | Inference time <100ms per frame |
| FR-5 | Generalize across different camera intrinsics | Should Have | Work with both simulation and real cameras |
| FR-6 | Achieve sub-millimeter accuracy on real robot | Nice to Have | Mean error <1mm on real test set |

### 2.2 Data Collection Functions

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-7 | Generate synthetic training data from AMBF simulation | Must Have | 300+ annotated samples with ground truth |
| FR-8 | Support diverse visual conditions (textures, lighting) | Must Have | 4 retinal textures × 3 lighting = 12 configurations |
| FR-9 | Collect real robot data via ROS integration | Must Have | Synchronized RGB capture from microscope |
| FR-10 | Provide automatic 2D keypoint annotation from 3D projection | Must Have | Labels: (u,v) coordinates for 3 keypoints |

### 2.3 Training and Evaluation Functions

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-11 | Train models on simulation data | Must Have | Convergent training on 300 samples |
| FR-12 | Fine-tune models on real data | Must Have | Sim-to-real transfer with 150 real images |
| FR-13 | Evaluate localization accuracy on held-out test set | Must Have | Report mean pixel error and mm error |
| FR-14 | Compare multiple model architectures | Must Have | Benchmark 6+ approaches |
| FR-15 | Support multi-resolution training (224, 512, 768, 1024 px) | Should Have | Resolution-dependent performance analysis |

### 2.4 Integration Functions

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-16 | ROS node for real-time inference | Must Have | Publish predicted positions as ROS messages |
| FR-17 | Integration with surgical control pipeline | Should Have | Compatible with existing I²RIS control stack |
| FR-18 | Confidence estimation for predictions | Nice to Have | Output uncertainty alongside predictions |

---

## 3. Input Specifications

### 3.1 Image Input
- **Format:** RGB images (PNG/JPEG)
- **Resolution:** 1024×1024 (simulation), 1920×1080 (real microscope)
- **Source:** AMBF virtual camera or Leica surgical microscope

### 3.2 Point Cloud Input
- **Format:** NumPy array (.npy)
- **Size:** 4096 3D points per sample
- **Coordinate System:** Camera frame (meters)

### 3.3 Ground Truth Labels
- **Gripper tip position:** (x, y, z) in mm
- **First disk position:** (x, y, z) in mm
- **Last disk position:** (x, y, z) in mm
- **2D projections:** (u, v) pixel coordinates

### 3.4 Camera Parameters
- **Intrinsic matrix K:** 3×3 focal length and principal point
- **Extrinsic matrix T:** 4×4 camera-to-world transform
- **Distortion coefficients:** Radial and tangential (k1, k2, p1, p2)

---

## 4. Output Specifications

### 4.1 Primary Output
- **Predicted 3D position:** (x, y, z) in mm for each keypoint
- **Predicted 2D position:** (u, v) in pixels for each keypoint

### 4.2 Secondary Output
- **Inference time:** Milliseconds per frame
- **Confidence score:** Optional uncertainty estimate (0-1)

### 4.3 Evaluation Metrics
- **Mean pixel error:** Euclidean distance in image space
- **Mean 3D error:** Euclidean distance in mm
- **Success rate <5mm:** Percentage of predictions within 5mm
- **Success rate <10mm:** Percentage of predictions within 10mm

---

## 5. Model Architectures

The system implements and benchmarks the following architectures:

### 5.1 Point Cloud Models
| Model | Method | Input |
|-------|--------|-------|
| PointNet | Shared MLP + Max Pooling | 4096×3 points |
| PointNet++ | Hierarchical Set Abstraction | 4096×3 points |
| DGCNN | Dynamic Graph CNN (EdgeConv) | 4096×3 points |

### 5.2 RGB Models
| Model | Method | Input |
|-------|--------|-------|
| HRNet-Heatmap | High-Resolution Net + Soft-Argmax | 1024×1024 RGB |
| Integral Regression | ResNet-50 + Soft-Argmax | 512-1024 RGB |
| SimpleBaseline | ResNet-50 + Deconvolution | 224-512 RGB |

### 5.3 Multimodal Fusion Models
| Model | Method | Input |
|-------|--------|-------|
| Hybrid Fusion | ResNet-18 + PointNet + MLP | RGB + Point Cloud + Joints |
| Transformer Fusion | ViT + PointNet + Cross-Attention | RGB + Point Cloud |

---

## 6. Performance Requirements

### 6.1 Accuracy Targets
| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Mean error (simulation) | <10mm | <5mm | <2mm |
| Mean error (real) | <20mm | <10mm | <1mm |
| Success rate <5mm | 50% | 80% | 95% |

### 6.2 Speed Requirements
| Metric | Requirement |
|--------|-------------|
| Inference latency | <100ms per frame |
| Training throughput | >10 samples/second |
| Data loading | <50ms per batch |

### 6.3 Resource Requirements
| Resource | Specification |
|----------|---------------|
| GPU Memory | <8GB for inference |
| CPU Memory | <16GB |
| Storage | <10GB for models and data |

---

## 7. Constraints and Assumptions

### 7.1 Assumptions
- Snake robot is visible in the camera field of view
- At least partial visibility of gripper and disk segments
- Adequate lighting for feature extraction
- Camera is calibrated with known intrinsics

### 7.2 Constraints
- No depth sensor available for real data (microscope optics limitation)
- Limited real training data (150 annotated frames maximum)
- Sim-to-real domain gap in appearance and lighting
- Real-time requirements for surgical integration

### 7.3 Out of Scope
- Full 6-DoF pose estimation (rotation)
- Multi-robot tracking
- Deformable tissue interaction modeling
- Closed-loop control implementation

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-04 | Chuntung Zhuang | Initial specification |
