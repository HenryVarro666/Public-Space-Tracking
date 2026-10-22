<div align="center">

# Public Space Tracking

**Detect, track and count people entering and leaving a public space.**

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Object%20Detection-FF6F00?logo=tensorflow&logoColor=white)
![Detectron2](https://img.shields.io/badge/Detectron2-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
[![homepage](https://img.shields.io/badge/homepage-ccao2679.com-555555)](https://www.ccao2679.com/projects/#public-space-tracking)

<img src="docs/figures/overview.png" alt="Overview poster: compress video, detect people, track centroids, count crossings of an entry line" width="100%">

</div>

<!-- problem-framing:start -->

## Problem and evidence

> **Can the detection-and-tracking pipeline count entrances and exits reliably after frame subsampling, compared with manual event annotations on held-out video?**

**Why it matters:** Counting errors can accumulate even when individual frames look convincing. A useful occupancy prototype needs event-level validation and explicit failure cases.

**Central tension:** Subsampling lowers processing cost but makes motion association harder; a simple centroid rule can confuse crossing people, occlusion and missed detections.

[Full problem and verification plan](PROBLEM.md)

<!-- problem-framing:end -->

## Highlights

- **Detection:** people in each frame with the TensorFlow Object Detection API (Faster R-CNN Inception-ResNet v2), compared with Detectron2.
- **Tracking and counting:** box centroids are linked across frames by nearest-centroid continuity; a track that crosses the entry line counts as in or out.
- **Outputs:** annotated detection, counter and optical-flow videos, plus an occupancy heatmap.

<p align="center"><img src="demos/testHeatmap.png" alt="Detected person positions accumulated over a street scene" width="640"></p>

## Pipeline

| Step | Script | What it does |
|---|---|---|
| Compress | `compression.py` | Keep the first 5,000 frames and every 5th frame, resolution unchanged |
| Ingest | `ingestion.py` | Save entrance and exit lines for a camera view to JSON |
| Detect | `detection.py` | Run a pre-trained detector and save boxes to CSV |
| Visualize | `detectionVis.py` | Draw boxes on frames and rebuild the video |
| Count | `counter-builder.ipynb` | Link centroids into tracks and count line crossings |
| Heatmap | `detectionVis-Heatmap.ipynb` | Accumulate positions into a heatmap |

Detector weights go in `models/` (see the [TF2 detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)).

<details>
<summary>Original notes</summary>

### Test

1. Object_detection_T1.ipynb
TensorFlow 

2. Detectron2_T2.ipynb
PyTorch

## Steps
### Compression
compression.py

compress + shorten videos (without changing the resloution)

For example, the raw video has 7000+ frames. For better testing, we choose first 5000 frames, and then we divide it by each 5 frames. 

So we can get a new video, in which people can still move smothly.

### Ingestion
ingestion.py

The project is about public space tracking, so we need the location of entrance and exit. This script can save information to json file.

### Detection
detection.py

With pre-trained model, we can detect people in processed videos, and save the bounding box data to csv file.

[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
(download models and extract to models/ folder)

### Visilization
DetectionVis.py

Using csv file, we can draw the bounding boxs on each frame, and then combine them together, we can get a new video.

### Counter
counter-builder.ipynb

Continuity

By calculating the distance of nearest centeriods of each bounding box, we can know whether it is faster or slower than the mean distance. If it is slower, we can assume the centeriod is continued. If it can be contined by 5 frames, it will be someone's plotline. 
    I know this is not perfect regarding mean distance and continuity. They may be improved later :)

By calculating the location of plotline and boundingbox, we can know whether a person is entering or leaving the public area.

### Plot
detectionVis-Heatmap.ipynb

Draw the heatmaps based on bounding boxes. It needs linear transformation in the future to convert it to a top view picture.

</details>
