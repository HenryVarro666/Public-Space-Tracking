# Problem and acceptance

> **Can the detection-and-tracking pipeline count entrances and exits reliably after frame subsampling, compared with manual event annotations on held-out video?**

**Why it matters:** Counting errors can accumulate even when individual frames look convincing. A useful occupancy prototype needs event-level validation and explicit failure cases.

**Central tension:** Subsampling lowers processing cost but makes motion association harder; a simple centroid rule can confuse crossing people, occlusion and missed detections.

**Current evidence and gap:** [README.md](README.md) describes video compression, pretrained person detection, linking centroids between frames five apart, and a centroid-dot heatmap. The counter notebook marks links that enter or leave hand-drawn rectangles but computes no in/out totals, and the original notes acknowledge the limits of the linking rule. These components do not establish counting accuracy or a geometrically correct top-view map.

**Acceptance test:** Annotate entrance/exit events in a permitted video sample, then hold complete clips out of tuning. Compare the current centroid tracker with detection-only counting and a clearly specified association baseline at matched detections. Vary frame stride while recording event precision/recall, cumulative count error, identity switches and runtime. Separate occluded and crossing-person cases. If speed gains cause unacceptable count drift, reduce stride or improve association before extending deployment.

This is a prospective engineering test; no identification or surveillance capability is claimed.

中文版：[PROBLEM.zh.md](PROBLEM.zh.md)
