# Design and Interface Specifications
## Eye Snake Localization Using Vision

**Project:** Vision-Based Localization for Micro-Continuum Surgical Robots
**Student:** Chuntung Zhuang (czhuang3@jh.edu)
**Course:** CIS II - Spring 2026, Johns Hopkins University

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Eye Snake Localization System                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Data       │    │   Training   │    │   Inference          │  │
│  │   Pipeline   │───▶│   Pipeline   │───▶│   Pipeline           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         ▼                   ▼                      ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ AMBF Sim     │    │ PyTorch      │    │ ROS Integration      │  │
│  │ Real Camera  │    │ Models       │    │ Real-time Inference  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Data Sources                               │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│   AMBF Sim      │  Leica Microscope│  Joint Encoders                │
│   - RGB         │  - RGB Video     │  - Position                    │
│   - Depth       │  - ROS Bags      │  - Velocity                    │
│   - Point Cloud │                  │                                │
└────────┬────────┴────────┬────────┴────────────┬────────────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Processing Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│  - Image preprocessing (resize, normalize, augment)                  │
│  - Point cloud processing (sampling, centering)                      │
│  - Keypoint projection (3D to 2D)                                   │
│  - Dataset creation (train/val/test splits)                         │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Layer                                  │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│  Point Cloud │    RGB       │  Multimodal  │   Classical           │
│  Models      │    Models    │  Fusion      │   Baselines           │
├──────────────┼──────────────┼──────────────┼───────────────────────┤
│  PointNet    │ HRNet        │ Hybrid       │ Template Match        │
│  PointNet++  │ IntegralReg  │ Transformer  │ Color Segmentation    │
│  DGCNN       │ SimpleBase   │ Fusion       │                       │
│              │ ViTPose      │              │                       │
└──────────────┴──────────────┴──────────────┴───────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Output Layer                                   │
├─────────────────────────────────────────────────────────────────────┤
│  - 3D keypoint positions (gripper, first disk, last disk)           │
│  - 2D projected coordinates                                          │
│  - Confidence scores                                                 │
│  - ROS messages for integration                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Architecture

### 2.1 Robot System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Surgical Robot System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   SHER (Steady Hand Eye Robot)                                       │
│   └── Stabilizes overall surgical apparatus                          │
│       │                                                              │
│       ▼                                                              │
│   I²RIS (Integrated Intraocular Robotic System)                      │
│   └── Micro-continuum platform                                       │
│       │                                                              │
│       ▼                                                              │
│   Eye Snake                                                          │
│   └── 11 segmented elements                                          │
│   └── Alternating revolute joints                                    │
│   └── Two-finger gripper (fixed + movable)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Eye Snake Structure

```
                    Gripper Tip (Keypoint 1)
                         ▼
                    ┌─────────┐
                    │ Gripper │ ◄── Movable finger
                    │  (2)    │ ◄── Fixed finger
                    └────┬────┘
                         │
              Segment 11 (First Disk, Keypoint 2)
                    ┌────┴────┐
                    │ Disk 11 │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ Disk 10 │
                    └────┬────┘
                         │
                        ...
                         │
                    ┌────┴────┐
                    │ Disk 2  │
                    └────┬────┘
                         │
               Segment 1 (Last Disk, Keypoint 3)
                    ┌────┴────┐
                    │ Disk 1  │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  Base   │
                    └─────────┘
```

### 2.3 Camera Setup

| Component | Specification |
|-----------|---------------|
| **Simulation** | AMBF virtual camera, 1024×1024 resolution |
| **Real** | Leica surgical microscope, 1920×1080 resolution |
| **Calibration** | Checkerboard calibration for intrinsics |
| **Distortion** | Radial + tangential correction applied |

---

## 3. Software Architecture

### 3.1 Directory Structure

```
eye_snake_localization/
├── ambf_simulation/           # AMBF simulation environment
│   ├── models/                # Robot and eye ADF files
│   ├── textures/              # Retinal background textures (4)
│   └── scripts/               # Data collection scripts
│
├── data/                      # Dataset storage
│   ├── simulation/            # 300 sim samples × 12 configs
│   │   ├── rgb/               # 1024×1024 RGB images
│   │   ├── depth/             # Depth maps
│   │   ├── point_cloud/       # 4096×3 point clouds
│   │   └── labels/            # Ground truth positions
│   └── real/                  # Real microscope data
│       ├── videos/            # 6 video sequences (V1-V6)
│       ├── frames/            # 217 annotated frames
│       └── annotations/       # Manual keypoint labels
│
├── models/                    # Model implementations
│   ├── pointnet.py            # PointNet architecture
│   ├── pointnet2.py           # PointNet++ architecture
│   ├── dgcnn.py               # DGCNN architecture
│   ├── hrnet.py               # HRNet-Heatmap
│   ├── integral_regression.py # Integral Regression
│   ├── simple_baseline.py     # SimpleBaseline
│   ├── hybrid_fusion.py       # RGB + PC fusion
│   └── transformer_fusion.py  # ViT + PointNet fusion
│
├── training/                  # Training scripts
│   ├── train.py               # Main training loop
│   ├── losses.py              # Loss functions
│   ├── augmentations.py       # Data augmentation
│   └── config/                # Training configurations
│
├── inference/                 # Inference pipeline
│   ├── predict.py             # Single-image prediction
│   ├── ros_node.py            # ROS integration
│   └── benchmark.py           # Speed benchmarking
│
├── evaluation/                # Evaluation scripts
│   ├── evaluate.py            # Compute metrics
│   ├── visualize.py           # Visualization tools
│   └── compare_models.py      # Model comparison
│
└── utils/                     # Utility functions
    ├── camera.py              # Camera projection
    ├── transforms.py          # Coordinate transforms
    └── data_loader.py         # Dataset classes
```

### 3.2 Key Classes and Interfaces

#### 3.2.1 Dataset Interface

```python
class EyeSnakeDataset:
    """Base dataset class for eye snake localization."""

    def __init__(self, root_dir: str, split: str, resolution: int):
        """
        Args:
            root_dir: Path to data directory
            split: 'train', 'val', or 'test'
            resolution: Input image resolution (224, 512, 768, 1024)
        """

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'image': Tensor [3, H, W],
                'point_cloud': Tensor [N, 3],
                'keypoints_3d': Tensor [3, 3],  # 3 keypoints × xyz
                'keypoints_2d': Tensor [3, 2],  # 3 keypoints × uv
                'camera_intrinsics': Tensor [3, 3]
            }
        """
```

#### 3.2.2 Model Interface

```python
class LocalizationModel(nn.Module):
    """Base class for all localization models."""

    def forward(self, batch: Dict) -> Dict:
        """
        Args:
            batch: Dictionary containing input data

        Returns:
            {
                'pred_3d': Tensor [B, 3, 3],      # Predicted 3D positions
                'pred_2d': Tensor [B, 3, 2],      # Predicted 2D positions
                'heatmaps': Tensor [B, 3, H, W],  # Optional heatmaps
                'confidence': Tensor [B, 3]       # Optional confidence
            }
        """
```

#### 3.2.3 Trainer Interface

```python
class Trainer:
    """Training and evaluation manager."""

    def __init__(self, model, config, device):
        """Initialize trainer with model and config."""

    def train_epoch(self, dataloader) -> Dict:
        """Train for one epoch, return metrics."""

    def evaluate(self, dataloader) -> Dict:
        """Evaluate on dataloader, return metrics."""

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
```

---

## 4. Data Flow

### 4.1 Simulation Data Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ AMBF Engine │────▶│ Render RGB  │────▶│ Save Image  │
│             │     │ + Depth     │     │ (.png)      │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       ▼                                       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Robot State │────▶│ Project 3D  │────▶│ Save Labels │
│ (xyz)       │     │ to 2D (uv)  │     │ (.json)     │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 4.2 Real Data Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Microscope  │────▶│ ROS Bag     │────▶│ Extract     │
│ Video       │     │ Recording   │     │ Frames      │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Manual      │
                                        │ Annotation  │
                                        └─────────────┘
```

### 4.3 Training Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      Two-Stage Training                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Stage 1: Simulation Pretraining                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ 300 Sim     │────▶│ Train on    │────▶│ Pretrained  │        │
│  │ Images      │     │ Sim Data    │     │ Weights     │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                 │                 │
│  Stage 2: Real Finetuning                       ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ 150 Real    │────▶│ Finetune    │────▶│ Final       │        │
│  │ Images      │     │ on Real     │     │ Model       │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Interface Specifications

### 5.1 ROS Interface

#### Published Topics
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/eye_snake/gripper_position` | `geometry_msgs/Point` | Predicted gripper 3D position |
| `/eye_snake/keypoints_2d` | `sensor_msgs/PointCloud` | 2D keypoint positions |
| `/eye_snake/confidence` | `std_msgs/Float32MultiArray` | Prediction confidence |

#### Subscribed Topics
| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/microscope/image_raw` | `sensor_msgs/Image` | RGB image from microscope |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |

### 5.2 Configuration File Format

```yaml
# config.yaml
model:
  name: "integral_regression"
  backbone: "resnet50"
  pretrained: true

training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 200
  optimizer: "adamw"
  scheduler: "cosine"

data:
  resolution: 512
  num_keypoints: 3
  augmentation:
    brightness: 0.2
    blur: true
    noise: 0.1

inference:
  multi_scale: true
  scales: [480, 512, 544]
  ensemble: false
```

### 5.3 Command Line Interface

```bash
# Training
python train.py --config config.yaml --gpus 1

# Evaluation
python evaluate.py --checkpoint model.pth --data_dir ./data/real

# Real-time inference
python ros_node.py --model model.pth --rate 30

# Benchmark
python benchmark.py --model model.pth --num_samples 100
```

---

## 6. Algorithm Design

### 6.1 Keypoint Projection

The 2D keypoint coordinates are obtained by projecting 3D positions:

```
(u, v)^T = K · P · (T_camera^world)^-1 · p_target^world
```

Where:
- `K`: Camera intrinsic matrix (3×3)
- `P`: Projection matrix
- `T_camera^world`: Camera pose in world frame (4×4)
- `p_target^world`: 3D keypoint position in world frame

### 6.2 Soft-Argmax Decoding

For heatmap-based methods, coordinates are extracted via differentiable soft-argmax:

```python
def soft_argmax(heatmap, temperature=1.0):
    """
    Args:
        heatmap: [B, H, W] predicted probability map
        temperature: Softmax temperature
    Returns:
        coords: [B, 2] predicted (x, y) coordinates
    """
    B, H, W = heatmap.shape

    # Create coordinate grids
    x_coords = torch.linspace(0, 1, W).to(heatmap.device)
    y_coords = torch.linspace(0, 1, H).to(heatmap.device)

    # Apply softmax
    heatmap = F.softmax(heatmap.view(B, -1) / temperature, dim=-1)
    heatmap = heatmap.view(B, H, W)

    # Compute expected coordinates
    x = (heatmap.sum(dim=1) * x_coords).sum(dim=-1)
    y = (heatmap.sum(dim=2) * y_coords).sum(dim=-1)

    return torch.stack([x, y], dim=-1)
```

### 6.3 Multi-Resolution Ensemble

```python
def ensemble_predict(models, image, scales):
    """
    Ensemble predictions from multiple resolutions.

    Args:
        models: List of models trained at different resolutions
        image: Input image [C, H, W]
        scales: List of scales per model (e.g., [±32, ±64 px])
    Returns:
        pred: Averaged prediction [3, 2] for 3 keypoints
    """
    predictions = []

    for model, model_scales in zip(models, scales):
        for scale in model_scales:
            # Resize image
            resized = F.interpolate(image, size=scale)

            # Get prediction in normalized coordinates
            pred = model(resized)
            predictions.append(pred)

    # Average all predictions
    return torch.stack(predictions).mean(dim=0)
```

---

## 7. Error Handling

### 7.1 Input Validation
- Check image dimensions and channels
- Validate point cloud size (must have N points)
- Verify camera intrinsics are valid (positive focal length)

### 7.2 Runtime Errors
- GPU out of memory: Reduce batch size or resolution
- Model loading failure: Check checkpoint compatibility
- ROS connection timeout: Retry with exponential backoff

### 7.3 Prediction Failures
- Keypoint outside image bounds: Clamp to valid range
- Low confidence prediction: Return last valid prediction
- Occlusion detection: Flag as uncertain

