# User Documentation
## Eye Snake Localization Using Vision

**Project:** Vision-Based Localization for Micro-Continuum Surgical Robots
**Student:** Chuntung Zhuang (czhuang3@jh.edu)
**Course:** CIS II - Spring 2026, Johns Hopkins University

---

## 1. Introduction

### 1.1 Overview
The Eye Snake Localization System provides real-time 3D position estimation of the I²RIS snake robot's gripper and key segments during vitreoretinal surgery. Using deep learning on microscope imagery, it enables accurate localization without relying solely on encoder feedback.

### 1.2 Key Features
- **Vision-based localization** from RGB microscope images
- **Multiple model architectures** (13+ methods benchmarked)
- **Sim-to-real transfer** training for data-efficient deployment
- **Real-time inference** (>10 Hz) for surgical integration
- **ROS integration** for seamless control pipeline connection

### 1.3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 18.04 | Ubuntu 20.04 |
| Python | 3.7 | 3.8+ |
| GPU | NVIDIA GTX 1080 | NVIDIA RTX 3090 |
| VRAM | 8 GB | 24 GB |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |

---

## 2. Installation

### 2.1 Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Install CUDA (if not already installed)
# Follow NVIDIA's official guide for your GPU
```

### 2.2 Clone Repository

```bash
git clone https://github.com/ChunTungZhuangLeo/eye-snake-localization.git
cd eye-snake-localization
```

### 2.3 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2.4 Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Project dependencies
pip install -r requirements.txt
```

### 2.5 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from models import IntegralRegression; print('Models loaded successfully')"
```

---

## 3. Quick Start

### 3.1 Download Pretrained Models

```bash
# Download pretrained checkpoints
./scripts/download_checkpoints.sh

# Verify checkpoints
ls checkpoints/
# Expected: integral_reg_simreal_512.pth, hrnet_heatmap.pth, ...
```

### 3.2 Run Demo on Sample Image

```bash
# Single image inference
python demo.py --image samples/test_image.png --model integral_regression

# Output:
# Gripper tip: (0.45, 0.62) -> (864, 670) px
# First disk:  (0.42, 0.58) -> (806, 626) px
# Last disk:   (0.38, 0.51) -> (730, 550) px
# Inference time: 32.5 ms
```

### 3.3 Visualize Predictions

```bash
python visualize.py \
    --image samples/test_image.png \
    --model integral_regression \
    --output results/visualization.png
```

---

## 4. Data Preparation

### 4.1 Dataset Structure

```
data/
├── simulation/
│   ├── images/           # 1024×1024 RGB images
│   │   ├── sample_000.png
│   │   ├── sample_001.png
│   │   └── ...
│   ├── point_clouds/     # 4096×3 point clouds
│   │   ├── sample_000.npy
│   │   └── ...
│   └── labels.json       # Ground truth annotations
│
└── real/
    ├── frames/           # 1920×1080 RGB images
    │   ├── v1_frame_001.png
    │   └── ...
    └── annotations.json  # Manual keypoint labels
```

### 4.2 Label Format

```json
{
  "sample_000": {
    "image_path": "images/sample_000.png",
    "keypoints_2d": {
      "gripper": [512, 634],
      "first_disk": [489, 601],
      "last_disk": [445, 532]
    },
    "keypoints_3d": {
      "gripper": [2.45, -1.23, 8.91],
      "first_disk": [2.12, -1.15, 8.45],
      "last_disk": [1.56, -0.98, 7.21]
    }
  }
}
```

### 4.3 Generating Simulation Data

```bash
# Start AMBF simulator
cd ambf_simulation
./ambf_simulator --launch_file eye_snake.yaml &

# Run data collection script
python collect_data.py \
    --num_samples 300 \
    --output_dir ../data/simulation \
    --textures 4 \
    --lightings 3
```

### 4.4 Annotating Real Data

```bash
# Launch annotation tool
python annotate.py --video data/real/videos/v1.mp4

# Controls:
# - Click to place keypoint
# - 1/2/3 to select keypoint type
# - N for next frame
# - S to save
# - Q to quit
```

---

## 5. Training

### 5.1 Training Configuration

Create a configuration file `configs/my_config.yaml`:

```yaml
model:
  name: "integral_regression"
  backbone: "resnet50"
  pretrained_backbone: true
  num_keypoints: 3

data:
  train_dir: "data/simulation"
  val_dir: "data/simulation"
  resolution: 512
  augmentation:
    brightness: 0.2
    contrast: 0.2
    blur_prob: 0.3
    noise_std: 0.05

training:
  batch_size: 4
  epochs: 200
  learning_rate: 0.0001
  optimizer: "adamw"
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_epochs: 5

checkpoint:
  save_dir: "checkpoints"
  save_freq: 10
```

### 5.2 Training Commands

```bash
# Stage 1: Simulation pretraining
python train.py \
    --config configs/sim_pretrain.yaml \
    --data_dir data/simulation \
    --output_dir checkpoints/sim_pretrain

# Stage 2: Real finetuning
python train.py \
    --config configs/real_finetune.yaml \
    --data_dir data/real \
    --pretrained checkpoints/sim_pretrain/best.pth \
    --output_dir checkpoints/sim_real_ft
```

### 5.3 Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir checkpoints/sim_real_ft/logs

# View at http://localhost:6006
```

### 5.4 Training Tips

| Scenario | Recommendation |
|----------|----------------|
| Limited GPU memory | Reduce batch size, use gradient checkpointing |
| Overfitting | Add augmentation, reduce learning rate |
| Slow convergence | Increase learning rate, add warmup |
| Unstable training | Use gradient clipping, reduce lr |

---

## 6. Evaluation

### 6.1 Evaluate on Test Set

```bash
python evaluate.py \
    --checkpoint checkpoints/sim_real_ft/best.pth \
    --test_data data/real/test_50.json \
    --resolution 512 \
    --output results/evaluation.json
```

### 6.2 Understanding Results

```json
{
  "overall": {
    "mean_pixel_error": 21.5,
    "std_pixel_error": 8.3,
    "success_rate_5mm": 0.86,
    "success_rate_10mm": 1.00
  },
  "per_keypoint": {
    "gripper": {"mean_error": 24.2, "std": 9.1},
    "first_disk": {"mean_error": 20.1, "std": 7.8},
    "last_disk": {"mean_error": 20.2, "std": 8.0}
  }
}
```

### 6.3 Benchmark All Models

```bash
python benchmark_all.py \
    --data_dir data/real \
    --output results/benchmark.csv
```

---

## 7. Inference

### 7.1 Single Image Inference

```python
from inference import Predictor

# Load model
predictor = Predictor(
    checkpoint="checkpoints/sim_real_ft/best.pth",
    model_name="integral_regression",
    resolution=512,
    device="cuda"
)

# Run inference
image = load_image("test.png")
result = predictor.predict(image)

print(f"Gripper: {result['gripper_2d']} px")
print(f"Inference time: {result['time_ms']:.1f} ms")
```

### 7.2 Batch Inference

```python
# Process multiple images
images = [load_image(f) for f in image_files]
results = predictor.predict_batch(images, batch_size=8)
```

### 7.3 Multi-Scale Ensemble

```python
# Enable multi-scale inference for best accuracy
predictor = Predictor(
    checkpoint="checkpoints/best.pth",
    model_name="integral_regression",
    resolution=512,
    multi_scale=True,
    scales=[480, 512, 544]
)
```

---

## 8. ROS Integration

### 8.1 Launch ROS Node

```bash
# Source ROS and workspace
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash

# Launch localization node
roslaunch eye_snake_localization localization.launch \
    model:=checkpoints/best.pth \
    resolution:=512
```

### 8.2 ROS Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/microscope/image_raw` | `sensor_msgs/Image` | Input image (subscribed) |
| `/eye_snake/keypoints` | `geometry_msgs/PoseArray` | Predicted 3D positions |
| `/eye_snake/keypoints_2d` | `sensor_msgs/PointCloud` | 2D pixel coordinates |
| `/eye_snake/visualization` | `sensor_msgs/Image` | Annotated image |

### 8.3 ROS Parameters

```yaml
# localization.yaml
eye_snake_localization:
  model_path: "checkpoints/best.pth"
  model_name: "integral_regression"
  resolution: 512
  rate: 30  # Hz
  multi_scale: false
  publish_visualization: true
```

### 8.4 Example Usage

```python
#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseArray

def callback(msg):
    gripper = msg.poses[0].position
    print(f"Gripper at: ({gripper.x:.2f}, {gripper.y:.2f}, {gripper.z:.2f}) mm")

rospy.init_node('localization_listener')
rospy.Subscriber('/eye_snake/keypoints', PoseArray, callback)
rospy.spin()
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Batch size too large | Reduce batch size or resolution |
| Model not loading | Version mismatch | Check PyTorch version compatibility |
| Poor accuracy | Domain shift | Fine-tune on real data |
| Slow inference | CPU fallback | Ensure CUDA is available |
| ROS not connecting | Topic mismatch | Verify topic names |

### 9.2 Debug Mode

```bash
# Enable verbose logging
python inference.py --debug --log-level DEBUG

# Profile inference time
python inference.py --profile
```

### 9.3 Getting Help

- **GitHub Issues:** Report bugs and request features
- **Email:** czhuang3@jh.edu
- **Documentation:** See `/docs` folder for detailed guides

---

## 10. API Reference

### 10.1 Predictor Class

```python
class Predictor:
    """Main inference class for eye snake localization."""

    def __init__(
        self,
        checkpoint: str,
        model_name: str = "integral_regression",
        resolution: int = 512,
        device: str = "cuda",
        multi_scale: bool = False,
        scales: List[int] = None
    ):
        """
        Initialize predictor.

        Args:
            checkpoint: Path to model checkpoint
            model_name: Model architecture name
            resolution: Input image resolution
            device: 'cuda' or 'cpu'
            multi_scale: Enable multi-scale inference
            scales: List of scales for multi-scale mode
        """

    def predict(self, image: np.ndarray) -> Dict:
        """
        Run inference on single image.

        Args:
            image: RGB image as numpy array [H, W, 3]

        Returns:
            {
                'gripper_2d': (u, v),
                'first_disk_2d': (u, v),
                'last_disk_2d': (u, v),
                'gripper_3d': (x, y, z),  # if depth available
                'confidence': [c1, c2, c3],
                'time_ms': float
            }
        """

    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8
    ) -> List[Dict]:
        """Run inference on batch of images."""
```

### 10.2 Model Factory

```python
from models import create_model

# Create model by name
model = create_model(
    name="integral_regression",
    backbone="resnet50",
    num_keypoints=3,
    pretrained=True
)

# Available models:
# - "pointnet", "pointnet2", "dgcnn"
# - "hrnet", "integral_regression", "simple_baseline"
# - "hybrid_fusion", "transformer_fusion"
# - "vitpose", "dinov2_regression", "detr"
```

---

## 11. Appendix

### A. Keypoint Definitions

| ID | Name | Location | Color (viz) |
|----|------|----------|-------------|
| 0 | Gripper Tip | Center of gripper fingers | Red |
| 1 | First Disk | Segment 11 (nearest gripper) | Green |
| 2 | Last Disk | Segment 1 (nearest base) | Blue |

### B. Coordinate Systems

```
World Frame (AMBF):
  X: Right
  Y: Up
  Z: Out of screen (toward camera)

Camera Frame:
  X: Right
  Y: Down
  Z: Into scene (depth)

Image Frame:
  u: Columns (0 = left)
  v: Rows (0 = top)
```

### C. Model Performance Summary

| Model | Resolution | Pixel Error | Speed |
|-------|------------|-------------|-------|
| Integral Regression | 512 | 21.5 px | 32 ms |
| DINOv2 Regression | 512 | 41.8 px | 45 ms |
| Hourglass | 512 | 60.0 px | 28 ms |
| Multi-Res Ensemble | 512/768/1024 | 16.3 px | 120 ms |

---

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-04 | Chuntung Zhuang | Initial documentation |
