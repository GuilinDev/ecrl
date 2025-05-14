# Experiment Progress Log

This document tracks the setup, experiments, and observations for the RL-based GPU resource management project.

## Phase 1: Environment Setup - MicroK8s

The primary goal of this phase was to establish a stable and functional local Kubernetes environment using MicroK8s, suitable for GPU-accelerated workloads.

### Key Decisions & Rationale:

*   **Initial KubeEdge Attempt & Pivot:**
    *   Initially, KubeEdge was considered for a more "realistic" edge simulation as per `experiments/plan.md` (v1).
    *   Encountered persistent conflicts between EdgeCore and the `kubelet` from a `kind` cluster (used to host CloudCore). EdgeCore requires no other Kubelet on the node.
    *   Troubleshooting included:
        *   Identifying `kind`'s `kubelet` as the source of conflict.
        *   Attempting to make CloudCore (in `kind`) accessible to EdgeCore on the host via `kubectl port-forward`.
        *   Realizing the fundamental incompatibility of running EdgeCore and a `kind` control plane (with its `kubelet`) on the same machine without more complex setups (e.g., VM for EdgeCore).
    *   **Decision:** Abandoned KubeEdge in favor of a direct Kubernetes (initially `kind`, then MicroK8s) setup to simplify the environment and focus on RL agent interaction with the Kubernetes API, which offers similar abstractions for resource management. `experiments/plan.md` was updated accordingly.

*   **Transition from `kind` to MicroK8s:**
    *   **`kind` Setup for GPU:**
        *   Attempted to set up a `kind` cluster (1 control-plane, 1 worker) with GPU support.
        *   Installed NVIDIA GPU Operator via Helm.
        *   Encountered issues with GPU Operator pods (`nvidia-dcgm-exporter`, `gpu-feature-discovery`, `nvidia-container-toolkit-daemonset`) stuck in `Init`.
        *   Diagnosed `FailedCreatePodSandBox` ("no runtime for nvidia") and `driver-validation` failures in `nvidia-container-toolkit-daemonset`.
        *   Suspected incompatibility between a very new beta NVIDIA driver (570.133.07) and GPU Operator v25.3.0.
        *   After a driver change, `kind` cluster creation failed with `exec: "nvidia-container-runtime": executable file not found in $PATH`.
        *   Though `docker run --gpus all ... nvidia-smi` worked, indicating Docker's NVIDIA runtime was functional, `kind` still had issues.
    *   **Decision:** Switched to MicroK8s due to its simpler GPU setup (`microk8s enable nvidia`) and potentially better integration with host NVIDIA drivers. `experiments/plan.md` was updated.

### MicroK8s Setup Steps & Observations:

1.  **Installation:**
    *   Installed MicroK8s via `sudo snap install microk8s --classic`.
    *   Added user to `microk8s` group: `sudo usermod -aG microk8s $USER && newgrp microk8s`.

2.  **Core Addon Enablement:**
    *   Enabled `helm3` and `dns`: `microk8s enable helm3 dns` (dns is usually enabled by default or as a dependency).
    *   Enabled `nvidia` for GPU support: `microk8s enable nvidia`. This was successful and simpler than the `kind` + GPU Operator Helm install. Node became schedulable with GPU resources.
        *   This step simplified the `install.sh` plan, removing the manual GPU Operator Helm installation.

3.  **Storage, Networking, and Monitoring Addon Enablement:**
    *   Command executed: `microk8s enable hostpath-storage ingress metallb:10.64.140.43-10.64.140.49 metrics-server prometheus`
    *   **`hostpath-storage`:** Enabled successfully. Note: Warned as not suitable for production.
    *   **`ingress`:** Enabled successfully.
    *   **`metallb`:**
        *   Enabled with the specified IP range `10.64.140.43-10.64.140.49`.
        *   During enablement, encountered several errors:
            ```
            Error from server (InternalError): error when creating "STDIN": Internal error occurred: failed calling webhook "ipaddresspoolvalidationwebhook.metallb.io": failed to call webhook: Post "https://webhook-service.metallb-system.svc:443/validate-metallb-io-v1beta1-ipaddresspool?timeout=10s": dial tcp 10.152.183.231:443: connect: connection refused
            ```
            (and similar for `l2advertisementvalidationwebhook.metallb.io`)
        *   Despite these initial webhook errors, MetalLB reported as successfully enabled, and `ipaddresspool.metallb.io/default-addresspool` along with `l2advertisement.metallb.io/default-advertise-all-pools` were created. This suggests the webhooks became available shortly after the initial attempts.
    *   **`metrics-server`:** Enabled successfully.
    *   **`prometheus`:**
        *   Enabled successfully but issued a deprecation warning: `'prometheus' is deprecated and will soon be removed. Please use 'observability' instead.`
        *   The command proceeded to enable the `observability` addon, which includes `kube-prometheus-stack`, `loki`, and `tempo`.
        *   Observability stack reported as enabled (user/pass: admin/prom-operator).

### Current Status (as of 2025-05-13):

*   MicroK8s is installed and running.
*   Essential addons for GPU workloads, storage, networking, and observability are enabled:
    *   `helm3`
    *   `dns`
    *   `nvidia`
    *   `hostpath-storage`
    *   `ingress`
    *   `metallb` (IP range `10.64.140.43-10.64.140.49`)
    *   `metrics-server`
    *   `observability` (replacing the deprecated `prometheus` addon)
*   The system seems ready for deploying the MobileNetV4 workload and starting baseline experiments as outlined in `experiments/plan.md`.

### Decisions for Workload Deployment (as of 2025-05-13):

*   **Project Directory Structure for Experiments:**
    *   `experiments/scripts/`: For MicroK8s deployment YAMLs and related scripts.
    *   `experiments/baseline/`: For baseline experiment code, configurations, and results.
    *   `experiments/rl/`: For RL agent code, training scripts, models, and results.
    *   `experiments/data/`: For image datasets (e.g., Tiny ImageNet validation set).
*   **Image Dataset:** Tentatively **Tiny ImageNet (validation set)**, to be placed in `experiments/data/tiny-imagenet/val/`.
*   **Load Generator:** **Locust** (Python-based, flexible).
*   **MobileNetV4 Serving Method:** **Triton Inference Server** (performance, GPU support, flexibility).
*   **MobileNetV4 Kubernetes Manifest:** To be created at `experiments/scripts/mobilenetv4-triton-deployment.yaml`.

### Next Steps (from `experiments/plan.md`):

1.  **Verify all addon pods are running correctly, especially in `metallb-system` and `observability` namespaces.**
    *   **Status: COMPLETED (2025-05-13)**
    *   `metallb-system` pods (`controller`, `speaker`) are `Running` and `READY 1/1`.
    *   `observability` pods (Alertmanager, Grafana, Prometheus Operator, Kube State Metrics, Node Exporter, Loki, Promtail, Prometheus, Tempo) are all `Running` and `READY`. (Alertmanager had 1 restart but recovered).
2.  Proceed with deploying the MobileNetV4 service.
3.  Deploy the load generator (Locust/k6).
4.  Begin Phase 2: Baseline Evaluation.

---
*This log will be updated as the experiment progresses.*

## Environment Setup ✅
- [x] Install MicroK8s
- [x] Enable required addons (nvidia, metallb, observability)
- [x] Verify GPU support
- [x] Configure storage

## Baseline Experiment Setup ✅
- [x] Create Triton deployment for MobileNetV4
- [x] Set up PVC for model storage
- [x] **Switched to downloading pre-converted ONNX model (MobileNetV4 Conv Small) from Hugging Face Hub**
- [x] Update PVC population script to use downloaded ONNX model
- [x] Configure Locust for load testing
- [x] Create automation scripts (`Makefile` with `download-hf-model`, `baseline` targets)
- [x] Set up result collection

## Baseline Experiment Execution 🔄
- [ ] Run `make baseline` to download model, prepare PVC, deploy, and run tests:
  - [ ] Low load (e.g., 10 users)
  - [ ] Medium load (e.g., 50 users)
  - [ ] High load (e.g., 100 users)
- [ ] Collect and analyze results (stored in `results/baseline/`)
- [ ] Document baseline performance metrics

## RL Agent Development ⏳
- [ ] Design state space
- [ ] Define action space
- [ ] Implement reward function
- [ ] Develop PPO agent
- [ ] Create training pipeline

## RL Training ⏳
- [ ] Train agent with baseline data
- [ ] Validate agent performance
- [ ] Fine-tune hyperparameters
- [ ] Save best model

## RL Evaluation ⏳
- [ ] Deploy trained agent
- [ ] Run comparison experiments
- [ ] Collect performance metrics
- [ ] Compare with baseline

## Analysis and Documentation ⏳
- [ ] Statistical analysis
- [ ] Performance comparison
- [ ] Write final report
- [ ] Prepare presentation

## Notes
- Baseline experiment automation is now complete with `make baseline` command (includes ONNX download).
- **Model download script: `experiments/scripts/download_hf_onnx_model.py` (downloads `onnx-community/mobilenetv4_conv_small.e2400_r224_in1k`).**
- **Model location after download: `experiments/models/mobilenetv4/1/model.onnx`.**
- Results will be stored in `results/baseline/` directory with timestamps
- Each experiment run includes pod status, logs, and performance metrics
- Use `make clean-baseline` to clean up experiment resources, `make clean` to also remove downloaded model and results.
