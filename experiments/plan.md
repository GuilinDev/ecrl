Experimental Plan: RL-based GPU Resource Management in a MicroK8s-based Kubernetes Environment

1. Overall Objective

To design, implement, and evaluate a Reinforcement Learning (RL) agent for dynamic scheduling and resource management of GPU-accelerated workloads (specifically MobileNetV4 inference) in a local edge‑like Kubernetes environment powered by MicroK8s. The RL agent's performance will be compared against a baseline using standard Kubernetes scheduling and Horizontal Pod Autoscaling (HPA). The goal is to improve resource utilization while maintaining or enhancing application performance (latency, throughput, QoS).

2. Experimental Environment

Primary Hardware: User's local machine with:

CPU: (User to specify)

RAM: (User to specify)

GPU: NVIDIA GeForce RTX 3080 (10GB VRAM)

Operating System: Dual-boot Ubuntu 24.04 LTS (running natively on the primary hardware).

Local Kubernetes Setup (MicroK8s):

MicroK8s installed via `sudo snap install microk8s --classic`.

Enable core addons, GPU support, and monitoring:

```shell
sudo usermod -aG microk8s $USER && newgrp microk8s # Or re-login
microk8s status --wait-ready
microk8s enable helm3 nvidia metrics-server dns hostpath-storage prometheus ingress metallb:10.64.140.43-10.64.140.49
```

The `nvidia` addon will detect and use host NVIDIA drivers and the NVIDIA Container Toolkit automatically, deploying necessary components like the device plugin and DCGM-exporter. The `prometheus` addon provides the monitoring stack.

Automation Script (`install.sh` - Simplified):

Focus on: host prerequisites (NVIDIA drivers, NVIDIA Container Toolkit if not present), MicroK8s installation, user group modification, and enabling the core MicroK8s addons as listed above.

3. Workload

Application: MobileNetV4 image classification model.

Deployment: Kubernetes Deployment in MicroK8s requesting `nvidia.com/gpu: 1`:

Serving Method:

Triton Inference Server (recommended) or

Custom Python server (FastAPI/Flask)

Adapt manifest from existing `scripts/vllm-k8s.yaml` (change image/command/args).

4. Reinforcement Learning (RL) Agent

Algorithm: PPO (Proximal Policy Optimization).

State Space (S): Metrics from Prometheus (enabled via MicroK8s addon):

Node utilization (CPU, memory, GPU/VRAM from DCGM-exporter included with `nvidia` addon)

Pod resource metrics

Inference queue length, latency

Replica count

Action Space (A): Dynamic adjustments:

HPA target utilization

Replica count

CPU/GPU requests & limits

Reward Function (R): Composite:

```text
R = w1·(GPU_util_gain) + w2·(resource_efficiency) - w3·(latency_penalty) - w4·(qos_violation)
```

5. Baseline Policy

Scheduler: Kubernetes default

Autoscaling: HPA with static CPU thresholds

6. Monitoring & Load Generation

Monitoring:

Prometheus (enabled via `microk8s enable prometheus`).

DCGM-Exporter (deployed as part of `microk8s enable nvidia`).

Grafana can be optionally enabled via `microk8s enable grafana` if dashboards are needed.

Load Generation:

Locust or k6 deployed in-cluster or on host

Defined profiles: Low, High, Mixed load

7. Key Performance Indicators (KPIs)

Latency (avg, p95, p99)

Throughput (req/s)

CPU, memory, GPU utilization

QoS violation rate

Replica count over time

8. Experimental Procedure / Phases

Phase 1: Environment Setup

1.  Prepare Ubuntu 24.04 host with RTX 3080 NVIDIA drivers and NVIDIA Container Toolkit.
2.  Run simplified `install.sh` to install MicroK8s and enable core addons (DNS, storage, Helm, metrics-server, ingress, metallb, nvidia, prometheus).
3.  Verify GPU availability in MicroK8s and that Prometheus is scraping GPU metrics from DCGM-exporter.
4.  Deploy MobileNetV4 service and load generator.

Phase 2: Baseline Evaluation

Configure default scheduler + HPA.

Run each load profile N times (e.g., N=5).

Collect KPIs from Prometheus.

Compute means & 95% CIs.

Phase 3: RL Training

Implement PPO agent interacting via Kubernetes API.

Train under varied loads, monitor reward curves.

Apply convergence criteria, save model.

Phase 4: RL Evaluation

Load best policy in evaluation mode.

Repeat load profiles N times.

Collect KPIs and compare against baseline.

Phase 5: Analysis & Comparison

Statistical tests (e.g., t-test) on KPI differences.

Visualize with tables & charts.

Discuss strengths, weaknesses, limitations (e.g., single powerful edge node simulation).

Outline future work.

9. Statistical Analysis

N=5 repeats per condition

Report mean ± 95% CI

Use appropriate tests for significance

10. Considerations from Referenced Report

Convergence definitions and metrics (reward curves, policy stability)

Stopping criteria for training

System‑level robustness evaluation

11. Final Output

Document results for thesis and potential submission to IPCCC, SEC.

实验计划：基于强化学习的 MicroK8s Kubernetes 环境 GPU 资源管理

1. 总体目标

设计、实现并评估一个强化学习（RL）代理，用于在本地基于 MicroK8s 的边缘类 Kubernetes 环境中对 GPU 加速工作负载（MobileNetV4 推理）进行动态调度和资源管理。RL 代理的性能将与使用标准 Kubernetes 调度和水平 Pod 自动缩放（HPA）的基线进行比较，目标是在保证或提升延迟、吞吐量和 QoS 的同时，提高资源利用率。

2. 实验环境

主要硬件： 本地机器，

CPU：（用户指定）

内存：（用户指定）

GPU：NVIDIA GeForce RTX 3080（10GB 显存）

操作系统： Ubuntu 24.04 LTS 双系统。

本地 Kubernetes：MicroK8s

通过 `sudo snap install microk8s --classic` 安装 MicroK8s。

启用核心插件、GPU 支持和监控：

```shell
sudo usermod -aG microk8s $USER && newgrp microk8s # 或重新登录
microk8s status --wait-ready
microk8s enable helm3 nvidia metrics-server dns hostpath-storage prometheus ingress metallb:10.64.140.43-10.64.140.49
```

`nvidia` 插件将自动检测并使用宿主机 NVIDIA 驱动程序和 NVIDIA Container Toolkit，部署必要的组件如设备插件和 DCGM-exporter。`prometheus` 插件提供监控堆栈。

自动化脚本 (`install.sh` - 简化版):

主要关注：宿主机先决条件 (NVIDIA 驱动, NVIDIA Container Toolkit 如果不存在), MicroK8s 安装, 用户组修改, 以及启用上述核心 MicroK8s 插件。

3. 工作负载

应用： MobileNetV4 图像分类。

部署： Kubernetes Deployment，资源请求 `nvidia.com/gpu: 1`。

服务框架：Triton 或自定义 FastAPI

参考 `scripts/vllm-k8s.yaml` 结构，替换镜像和启动参数。

4. RL 代理

算法： PPO。

状态空间： Prometheus 指标 (通过 MicroK8s 插件启用):

节点利用率 (CPU、内存、GPU/显存 来自 `nvidia` 插件包含的 DCGM-exporter)

Pod 资源指标

推理队列长度、延迟

Replica 数

动作空间： 调整 HPA 目标、Replica 数、资源请求/限制

奖励： 复合函数，平衡 GPU 利用、资源效率、延迟和 QoS 违规。

5. 基线策略

调度器： Kubernetes 默认

自动缩放： HPA 静态阈值

6. 监控与负载

监控：

Prometheus (通过 `microk8s enable prometheus` 启用)。

DCGM-Exporter (作为 `microk8s enable nvidia` 的一部分部署)。

如果需要仪表盘，可选择通过 `microk8s enable grafana` 启用 Grafana。

负载： Locust/k6，定义低/高/混合模式

7. KPI 指标

延迟（avg, p95, p99）、吞吐、CPU/内存/GPU 利用、QoS 违规率、Replica 数变化

8. 实验流程

阶段 1：环境搭建

1.  准备好带有 RTX 3080 NVIDIA 驱动和 NVIDIA Container Toolkit 的宿主机 Ubuntu 24.04。
2.  运行简化的 `install.sh` 脚本安装 MicroK8s 并启用核心插件 (DNS, storage, Helm, metrics-server, ingress, metallb, nvidia, prometheus)。
3.  验证 MicroK8s 中 GPU 的可用性，以及 Prometheus 是否从 DCGM-exporter 抓取 GPU 指标。
4.  部署 MobileNetV4 服务和负载生成器。

阶段 2：基线评估 N=5；

阶段 3：RL 训练；

阶段 4：RL 评估 N=5；

阶段 5：统计分析 & 比较；讨论优势、劣势、局限性 (例如，单个强大边缘节点的模拟)。

9. 统计分析

结果以平均 ± 95% CI 呈现，使用适当统计检验

10. 报告考量

参考“强化学习在开放与持续运行环境中的收敛性”报告的收敛与评估框架

11. 最终输出

形成论文材料，目标投稿 IPCCC、SEC。

