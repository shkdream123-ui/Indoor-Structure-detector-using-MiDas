# **Indoor Structure Detector using MiDaS**


## 📌 **Introduction**

This project aims to explore a lightweight approach for indoor structural understanding using monocular depth estimation.

**Short-term goal**
To detect simple indoor structural elements such as walls and corners by combining:
Edge detection
Gradient distribution from depth maps

**Long-term goal**
This work serves as a foundation for future indoor scene understanding, where higher-level systems (e.g., mapping or reconstruction) can leverage structurally meaningful cues extracted from raw visual input.


## **⚙️ Core Ideas & Methodology**
1. Edge-based Structural Hypothesis
Perform edge detection on input frames
Extract dominant directions from detected edges
Analyze the distribution of orientations

**Heuristic scoring:**

**Wall**
Dominant single direction
Strong directional consistency
**Corner**
Multiple dominant directions
Clear intersection between directions
2. Depth-based Structural Refinement
(1) Object Mask Generation
Use depth map to identify:
Near objects
Sharp depth discontinuities
Generate an object mask
Exclude object regions from further structural analysis

👉 목적: 구조(벽, 모서리)와 객체를 분리

(2) Structure Mask Gradient Analysis
Focus only on structure mask (depth-filtered region)
Analyze gradient flow within depth map

**Key observations:**

Wall-like structure
Consistent gradient flow in a single direction
Corner-like structure
Two distinct gradient flows intersecting

<img width="1809" height="576" alt="Gemini_Generated_Image_qa6ioaqa6ioaqa6i" src="https://github.com/user-attachments/assets/7f445c6b-82fb-4b7f-a8ac-c1581f8e4dcd" />

## **🧠 Design Philosophy**

Rather than making a hard final decision, this detector is designed to:

Generate as much structural information as possible, leaving final interpretation to higher-level systems.

## **📊 Results**

![structure-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/19579d6d-950e-415d-9a67-5babf8275c28)

Real-time edge + depth fusion
Structural hypothesis visualization (wall / corner likelihood)
⚠️ Limitations
Depth maps from monocular estimation are inherently noisy
Edge detection is sensitive to lighting and texture
Structural inference may fluctuate frame-to-frame

👉 On current stages of development:

Fine-tuning is needed
Unable to decide using a single frame
🔄 System Perspective

This detector is not intended to be a final decision-maker, but rather:

A feature generator for:
Mapping systems
Scene reconstruction pipelines
Providing:
Structural cues
Directional consistency
Candidate regions of interest
## **🚀 Future Work**
Temporal consistency enhancement
Integration with SLAM / mapping pipelines
Structural tracking across frames
Long-term Vision
Improve robustness using deep learning-based refinement
Learn:
Stable structural priors
Noise-resistant feature representations
