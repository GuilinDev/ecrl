# Experimental Plan: RL-based GPU Resource Management in a MicroK8s-based Kubernetes Environment

## 1. Overall Objective

To design, implement, and evaluate a Reinforcement Learning (RL) agent for dynamic scheduling and resource management of GPU-accelerated workloads (specifically MobileNetV4 inference) in a local edge‑like Kubernetes environment powered by MicroK8s. The RL agent's performance will be compared against a baseline using standard Kubernetes scheduling and Horizontal Pod Autoscaling (HPA). The goal is to improve resource utilization while maintaining or enhancing application performance (latency, throughput, QoS).

## 2. Experimental Environment

* **Primary Hardware:** User's local machine with:

  * CPU: (User to specify)
  * RAM: (User to specify)
  * **GPU: NVIDIA GeForce RTX 3080 (10GB VRAM)**
* **Operating System:** **Dual-boot Ubuntu 24.04 LTS** (running natively on the primary hardware).
* **Local Kubernetes Setup (MicroK8s):**

  * **MicroK8s** installed via `sudo snap install microk8s --classic`.
  * Enable GPU support and Helm3:

    ```bash
    sudo usermod -aG microk8s $USER && newgrp microk8s
    microk8s enable helm3 nvidia metrics-server dns hostpath-storage metallb:10.64.140.43-10.64.140.49
    ```
  * The built‑in `nvidia` addon will detect and use host NVIDIA drivers and Container Toolkit automatically.
  * **Monitoring Addons:** `microk8s enable prometheus` (or deploy `kube-prometheus-stack` via Helm) and enable `ingress` / `metalLB` for service exposure.
* **Automation Script:** Adapt the existing `install.sh` to:

  * Install Docker and NVIDIA Container Toolkit if missing on Ubuntu.
  * Install and configure MicroK8s with the above addons.
  * Deploy NVIDIA GPU Operator via:

    ```bash
    microk8s helm3 repo add nvidia https://nvidia.github.io/gpu-operator
    microk8s helm3 repo update
    microk8s helm3 install gpu-operator nvidia/gpu-operator \
      -n gpu-operator --create-namespace \
      --version v25.3.0 \
      --set driver.enabled=false \
      --set driver.usePrecompiled=true \
      --set driver.precompiled.driverVersion=570
    ```

## 3. Workload

* **Application:** **MobileNetV4** image classification model.
* **Deployment:** Kubernetes Deployment in MicroK8s requesting `nvidia.com/gpu: 1`:

  * **Serving Method:**

    * **Triton Inference Server** with GPU acceleration
    * Configuration: `instance_group [ { kind: KIND_GPU, count: 1 } ]` in model config
  * Deployment manifest: `scripts/mobilenetv4-triton-deployment.yaml`
  * GPU utilization: ~43% during inference with 1.5GB VRAM usage

## 4. Reinforcement Learning (RL) Agent

* **Algorithm:** **PPO (Proximal Policy Optimization)**.
* **State Space (S):** Metrics from Prometheus:

  * Node utilization (CPU, memory, GPU/VRAM)
  * Pod resource metrics
  * Inference queue length, latency
  * Replica count
* **Action Space (A):** Dynamic adjustments:

  * HPA target utilization
  * Replica count
  * CPU/GPU requests & limits
* **Reward Function (R):** Composite:

  ```text
  R = w1·(GPU_util_gain) + w2·(resource_efficiency) - w3·(latency_penalty) - w4·(qos_violation)
  ```

## 5. Baseline Policy

* **Scheduler:** Kubernetes default
* **Autoscaling:** HPA with static CPU thresholds

## 6. Monitoring & Load Generation

* **Monitoring:**

  * `kube-prometheus-stack` via Helm or `microk8s enable prometheus`
  * DCGM-Exporter
* **Load Generation:**

  * **Locust** or **k6** deployed in-cluster or on host
  * Defined profiles: Low, High, Mixed load

## 7. Key Performance Indicators (KPIs)

* Latency (avg, p95, p99)
* Throughput (req/s)
* CPU, memory, GPU utilization
* QoS violation rate
* Replica count over time
* Model responsiveness (success rate with synthetic data)

## 8. Experimental Procedure / Phases

**Phase 1: Environment Setup**

1. Prepare Ubuntu 24.04 with RTX 3080 drivers.
2. Run adapted `install.sh` to install Docker, NVIDIA toolkit, MicroK8s with addons.
3. Deploy GPU Operator, monitoring stack, and DCGM-Exporter.
4. Deploy MobileNetV4 service and load generator.

**Phase 2: Baseline Evaluation**

1. Configure default scheduler + HPA.
2. Run each load profile N times (e.g., N=5):
   - Low load: 10 users
   - Medium load: 50 users
   - High load: 100 users
3. Collect QoS metrics from Locust and resource metrics from Prometheus.
4. Evaluate model responsiveness using synthetic data.
5. Compute means & 95% CIs for all metrics.

**Phase 3: RL Training**

1. Implement PPO agent interacting via Kubernetes API.
2. Train under varied loads, monitor reward curves.
3. Apply convergence criteria, save model.

**Phase 4: RL Evaluation**

1. Load best policy in evaluation mode.
2. Repeat load profiles N times.
3. Collect KPIs and compare against baseline.

**Phase 5: Analysis & Comparison**

1. Statistical tests (e.g., t-test) on KPI differences.
2. Visualize with tables & charts.
3. Discuss strengths, weaknesses, limitations.
4. Outline future work.

## 9. Statistical Analysis

* N=5 repeats per condition
* Report mean ± 95% CI
* Use appropriate tests for significance

## 10. Considerations from Referenced Report

* Convergence definitions and metrics (reward curves, policy stability)
* Stopping criteria for training
* System‑level robustness evaluation

## 12. Challenges and Mitigations

### Model Accuracy Evaluation
* **Challenge**: Mapping between Tiny ImageNet classes (200) and ImageNet classes (1000) used by MobileNetV4 is difficult
* **Mitigation**: Use synthetic data to evaluate model responsiveness instead of true accuracy
* **Alternative**: If accuracy is critical, consider using a model trained specifically on Tiny ImageNet or creating a proper class mapping with expert knowledge

### Resource Metrics Collection
* **Challenge**: Detailed GPU metrics are needed for RL agent
* **Mitigation**: Use DCGM-Exporter for comprehensive GPU metrics
* **Alternative**: Implement custom metrics collection if standard exporters are insufficient

## 11. Final Output

Document results for thesis and potential submission to IPCCC, SEC.

---

# 实验计划：基于强化学习的 MicroK8s Kubernetes 环境 GPU 资源管理

## 1. 总体目标

设计、实现并评估一个强化学习（RL）代理，用于在本地基于 MicroK8s 的边缘类 Kubernetes 环境中对 GPU 加速工作负载（MobileNetV4 推理）进行动态调度和资源管理。RL 代理的性能将与使用标准 Kubernetes 调度和水平 Pod 自动缩放（HPA）的基线进行比较，目标是在保证或提升延迟、吞吐量和 QoS 的同时，提高资源利用率。

## 2. 实验环境

* **主要硬件：** 本地机器，

  * CPU：（用户指定）
  * 内存：（用户指定）
  * **GPU：NVIDIA GeForce RTX 3080（10GB 显存）**
* **操作系统：** Ubuntu 24.04 LTS 双系统。
* **本地 Kubernetes：MicroK8s**

  * 安装：`sudo snap install microk8s --classic`。
  * 加入 microk8s 组并启用插件：

    ```bash
    sudo usermod -aG microk8s $USER && newgrp microk8s
    microk8s enable helm3 nvidia metrics-server dns hostpath-storage prometheus ingress metallb:10.64.140.43-10.64.140.49
    ```
  * `nvidia` 插件自动使用宿主机驱动。
* **自动化脚本：** 调整 `install.sh` 安装 Docker、NVIDIA 工具箱、MicroK8s 及上述插件，并部署 GPU Operator 和监控组件。

## 3. 工作负载

* **应用：** MobileNetV4 图像分类。
* **部署：** Kubernetes Deployment，资源请求 `nvidia.com/gpu: 1`。

  * 服务框架：Triton 或自定义 FastAPI
  * 参考 `scripts/vllm-k8s.yaml` 结构，替换镜像和启动参数。

## 4. RL 代理

* **算法：** PPO。
* **状态空间：** Prometheus 指标（节点/Pod CPU、内存、GPU 利用率、队列长度、延迟、Replica 数）
* **动作空间：** 调整 HPA 目标、Replica 数、资源请求/限制
* **奖励：** 复合函数，平衡 GPU 利用、资源效率、延迟和 QoS 违规。

## 5. 基线策略

* **调度器：** Kubernetes 默认
* **自动缩放：** HPA 静态阈值

## 6. 监控与负载

* **监控：** `kube-prometheus-stack` 或 `microk8s enable prometheus` + DCGM-Exporter
* **负载：** Locust/k6，定义低/高/混合模式

## 7. KPI 指标

延迟（avg, p95, p99）、吞吐、CPU/内存/GPU 利用、QoS 违规率、Replica 数变化

## 8. 实验流程

1. 环境搭建；
2. 基线评估 N=5；
3. RL 训练；
4. RL 评估 N=5；
5. 统计分析 & 比较；

## 9. 统计分析

结果以平均 ± 95% CI 呈现，使用适当统计检验

## 10. 报告考量

参考“强化学习在开放与持续运行环境中的收敛性”报告的收敛与评估框架

## 11. 最终输出

形成论文材料，可投稿 IPCCC、SEC。
