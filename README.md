# Project: RL-based GPU Resource Management in an Edge-like Kubernetes Environment

This project aims to design, implement, and evaluate a Reinforcement Learning (RL) agent for dynamic scheduling and resource management of GPU-accelerated workloads in a local edge-like Kubernetes environment.

## Experimental Setup Status (As of Current Progress)

This section details the configuration of the experimental environment established so far.

### 1. Host System
*   **Operating System:** Ubuntu 24.04 LTS
*   **Primary GPU:** NVIDIA GeForce RTX 3080 (10GB VRAM)
*   **NVIDIA Drivers:** Installed directly on the host OS.
*   **Container Runtime:** Docker with NVIDIA Container Toolkit configured.

### 2. Kubernetes Environment (Simulating an Edge-like Node)
*   **Platform:** Kind (Kubernetes in Docker)
*   **Cluster Name:** `gpu-cluster`
*   **Cluster Configuration (`kind-1w-config.yaml`):**
    *   1 Control-Plane Node
    *   1 Worker Node
*   **GPU Enablement in Kind:**
    *   The host system has the necessary NVIDIA drivers and NVIDIA Container Toolkit.
    *   **NVIDIA GPU Operator:** Installed via Helm into the `gpu-operator` namespace within the `kind` cluster.
        *   The installation was performed with `driver.enabled` set to `false` since host drivers are utilized.
        *   This operator is responsible for making the host's GPU available as a schedulable resource within the `kind` cluster (specifically on the worker node).
*   **Purpose:** This `kind` cluster is configured to simulate a single, GPU-enabled edge-like node. The RL agent will be developed and tested for its ability to manage GPU resources effectively in this environment.

### 3. Next Steps (Planned)
*   Deployment of monitoring stack (`kube-prometheus-stack`, `dcgm-exporter`).
*   Deployment of the MobileNetV4 inference service (requesting GPU).
*   Setup of a load generator.
*   Development and evaluation of the RL agent and baseline policies.

---
*This README will be updated as the experiment progresses.*
