# Experimental Plan: RL-based GPU Resource Management in an Edge-like Kubernetes Environment

## 1. Overall Objective
To design, implement, and evaluate a Reinforcement Learning (RL) agent for dynamic scheduling and resource management of GPU-accelerated workloads (specifically MobileNetV4 inference) in a local edge-like Kubernetes environment. The RL agent's performance will be compared against a baseline using standard Kubernetes scheduling and Horizontal Pod Autoscaling (HPA). The goal is to improve resource utilization while maintaining or enhancing application performance (latency, throughput, QoS).

## 2. Experimental Environment

* **Primary Hardware:** User's local machine with:
    * CPU: (User to specify)
    * RAM: (User to specify)
    * **GPU: NVIDIA GeForce RTX 3080 (10GB VRAM)**
* **Operating System:** **Dual-boot Ubuntu 24.04 LTS** (running natively on the primary hardware).
* **Local Kubernetes Setup (Kind):**
    * **Kind (Kubernetes in Docker):** To be installed on Ubuntu 24.04 to create a local Kubernetes cluster, simulating an edge-like environment.
    * **GPU Support for Kind:** Enabled by:
        * Host NVIDIA drivers (already on Ubuntu 24.04).
        * Docker with NVIDIA Container Toolkit configured within Ubuntu.
        * `nvkind` utility for creating GPU-enabled Kind clusters.
        * **NVIDIA GPU Operator:** To be installed via Helm into the Kind cluster (e.g., `helm install --wait gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace --set driver.enabled=false`) as host has drivers.
    * The `install.sh` script (provided by user, adapted to skip host driver installation and modify `snap` commands if needed) will be used to automate this setup.
    * **`cloud-provider-kind`:** Installation is included in `install.sh` to simulate LoadBalancer services for Kind.
* **Simplification Note:** KubeEdge (CloudCore + EdgeCore) was initially considered for a more realistic edge simulation but has been removed from the plan due to setup complexities and conflicts (kubelet vs. EdgeCore) encountered when attempting to run both CloudCore (in Kind) and EdgeCore on the same host machine. The core RL goal remains achievable within the Kind Kubernetes environment.

## 3. Workload
* **Application:** **MobileNetV4** image classification model.
* **Deployment:** Deployed as an inference service within a Kubernetes Deployment in the Kind cluster.
    * **Serving Method:** vLLM (as seen in the `scripts/vllm-k8s.yaml` example) is designed for LLMs and is **not suitable** for MobileNetV4 (a CNN). Instead, MobileNetV4 will be served using a dedicated inference server appropriate for CNNs, such as:
        * **Triton Inference Server (Recommended):** Supports multiple frameworks and offers good performance on GPUs.
        * **Custom Python-based server (FastAPI/Flask):** Provides flexibility if specific custom logic is needed.
        * (TensorFlow Serving or TorchServe are also viable alternatives).
    * The Kubernetes manifest structure (Deployment, Service) from `scripts/vllm-k8s.yaml` can be adapted, but the container `image`, `command`, and `args` must be changed to use the chosen MobileNetV4 serving solution.
* **GPU Requirement:** The service will request `nvidia.com/gpu: 1` resource.

## 4. Reinforcement Learning (RL) Agent
* **Algorithm:** **PPO (Proximal Policy Optimization)**.
* **State Space (S):** Metrics collected via Prometheus from the Kind cluster, potentially including:
    * Node resource utilization (CPU, Memory, GPU Utilization, GPU Memory Usage).
    * Pod resource utilization (CPU, Memory, GPU Utilization).
    * Inference request queue length.
    * Average inference latency.
    * Current number of service replicas.
    * (Metrics will be normalized and carefully selected).
* **Action Space (A):** Dynamically adjust configurations of the MobileNetV4 service within Kind, for example:
    * Modifying HPA target utilization values.
    * Directly setting the number of replicas for the MobileNetV4 Deployment.
    * (Potentially) Adjusting CPU/GPU resource requests/limits for the inference pods.
* **Reward Function (R):** A composite function designed to balance multiple objectives:
    * Example: `Reward = w1 * (GPU_Util_Benefit) + w2 * (Other_Resource_Efficiency) - w3 * (Latency_Penalty) - w4 * (QoS_Violation_Penalty)`.
    * Weights (`w1, w2, w3, w4`) and specific benefit/penalty functions will require careful tuning.

## 5. Baseline Policy
* **Scheduler:** Kubernetes default scheduler within the Kind cluster.
* **Autoscaling:** Standard Horizontal Pod Autoscaler (HPA) configured for the MobileNetV4 Deployment, based on static thresholds for CPU utilization or a similar metric.

## 6. Monitoring & Load Generation
* **Monitoring:**
    * **`kube-prometheus-stack`:** Will be deployed via Helm chart (e.g., from `prometheus-community`) for a comprehensive monitoring solution including Prometheus and Grafana.
    * **DCGM-Exporter:** Will be deployed within Kind to provide detailed NVIDIA GPU metrics to Prometheus.
* **Load Generation:**
    * A dedicated load generation tool/script.
    * **Recommended Tools:**
        * **Locust or k6:** Preferred for their scriptability, ability to generate diverse load patterns, and ease of deployment within Kubernetes.
    * Capable of sending inference requests to the MobileNetV4 service.
    * **Load Profiles:** Experiments will be run under different, clearly defined load profiles:
        * Low Load
        * High Load
        * Mixed/Variable Load

## 7. Key Performance Indicators (KPIs) for Evaluation
The following KPIs will be collected for both the baseline and the RL agent:
* Average Inference Response Time (Latency).
* Percentile Latencies (e.g., p95, p99).
* System Throughput (inferences per second).
* Node & Pod Resource Utilization:
    * CPU Utilization (%)
    * Memory Usage (%)
    * GPU Utilization (%)
    * GPU Memory Usage (%)
* QoS Violation Rate (e.g., percentage of requests exceeding a predefined latency threshold).
* Number of active service replicas over time.

## 8. Experimental Procedure / Phases

**Phase 1: Environment Setup**
1.  Prepare Ubuntu 24.04 host with NVIDIA drivers for RTX 3080.
2.  Run the adapted `install.sh` script to set up Docker, Kind with GPU support, NVIDIA Container Toolkit, `nvkind`. Confirm `cloud-provider-kind` is active.
3.  Install NVIDIA GPU Operator into the Kind cluster via Helm (`--set driver.enabled=false`).
4.  Deploy Monitoring stack: `kube-prometheus-stack` (Prometheus, Grafana) and `dcgm-exporter` into Kind.
5.  Deploy MobileNetV4 inference service into Kind using the chosen serving method (e.g., Triton or custom FastAPI app, requesting GPU).
6.  Set up the load generator (e.g., Locust or k6, deployed in Kind or on the host).

**Phase 2: Baseline Policy Evaluation**
1.  Configure Kubernetes default scheduler and static HPA for the MobileNetV4 service in Kind.
2.  Execute each defined load profile (Low, High, Mixed) against the service.
3.  Repeat each run N times (e.g., N=5) for statistical validity.
4.  Collect all defined KPIs from Prometheus for each run.
5.  Calculate average KPIs and confidence intervals.

**Phase 3: RL Agent Training**
1.  Implement the PPO agent, including state representation, action execution (interacting with Kind's Kubernetes API), and reward calculation.
2.  Run the training loop: the agent interacts with the MobileNetV4 service running in Kind under various (or specific training) load conditions.
3.  Monitor training progress (reward curves, losses, evaluation metrics on validation scenarios if applicable).
4.  Apply stopping criteria based on convergence (e.g., reward plateau, performance saturation).
5.  Save the trained RL agent model(s).

**Phase 4: RL Agent Evaluation**
1.  Load the best trained RL agent policy (in evaluation mode).
2.  Execute the *exact same* load profiles (Low, High, Mixed) used for baseline evaluation.
3.  Repeat each run N times (e.g., N=5).
4.  Collect all defined KPIs from Prometheus.
5.  Calculate average KPIs and confidence intervals.

**Phase 5: Results Analysis and Comparison**
1.  Compare the averaged KPIs (and CIs) of the RL agent against the baseline policy for each load profile.
2.  Use appropriate statistical tests (e.g., t-tests) to determine the significance of observed differences.
3.  Visualize results using tables and graphs.
4.  Discuss the findings, analyze trade-offs made by the RL agent, and identify its strengths/weaknesses.
5.  Acknowledge limitations (e.g., local setup represents a high-capability edge node, not resource-constrained devices; **KubeEdge-specific features and architecture were not explored due to the simplification of removing KubeEdge from the setup**).
6.  Outline future work.

## 9. Statistical Analysis
* Each experimental condition (load profile for baseline, load profile for RL agent) will be repeated N=5 times (or a suitable number) to allow for statistical analysis.
* Results will be presented as mean ± confidence interval (e.g., 95% CI).
* Statistical tests will be used to compare the performance of the RL agent and the baseline.

## 10. Considerations from Referenced Report
The initial research report "强化学习在开放与持续运行环境中的收敛性：定义、度量与评估" will guide aspects of:
* Defining and monitoring training convergence (e.g., reward curves, policy stability).
* Criteria for stopping the RL training phase.
* Framework for evaluating the final "goodness" of the learned policy beyond just training metrics, focusing on system-level KPIs and robustness under varied loads.

## 11. Final Output
The results of this experimental plan will be documented for a thesis and/or publication in relevant conferences (e.g., IPCCC, SEC).

# 实验计划：基于强化学习的边缘类 Kubernetes 环境 GPU 资源管理

## 1. 总体目标
设计、实现并评估一个强化学习（RL）代理，用于在本地边缘类 Kubernetes 环境中对 GPU 加速工作负载（特别是 MobileNetV4 推理）进行动态调度和资源管理。RL 代理的性能将与使用标准 Kubernetes 调度和水平 Pod 自动缩放器（HPA）的基线进行比较。目标是在保证或提升应用性能（延迟、吞吐量、QoS）的同时，提高资源利用率。

## 2. 实验环境

* **主要硬件：** 用户本地计算机，配置如下：
    * CPU：（用户指定）
    * RAM：（用户指定）
    * **GPU：NVIDIA GeForce RTX 3080（10GB显存）**
* **操作系统：** **双系统 Ubuntu 24.04 LTS**（在主要硬件上本地运行）。
* **本地 Kubernetes 设置 (Kind)：**
    * **Kind (Kubernetes in Docker)：** 安装在 Ubuntu 24.04 上，用于创建本地 Kubernetes 集群，模拟一个边缘类环境。
    * **Kind 的 GPU 支持：** 通过以下方式启用：
        * 宿主机 NVIDIA 驱动程序（已安装在 Ubuntu 24.04 上）。
        * 在 Ubuntu 内配置了 NVIDIA Container Toolkit 的 Docker。
        * 使用 `nvkind` 实用程序创建支持 GPU 的 Kind 集群。
        * **NVIDIA GPU Operator：** 将通过 Helm 安装到 Kind 集群中（例如，`helm install --wait gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace --set driver.enabled=false`），因为宿主机已有驱动。
    * 用户提供的 `install.sh` 脚本（经过调整，如果 Ubuntu 上已有驱动则跳过宿主机驱动安装步骤，并根据需要修改 `snap` 命令）将用于自动化此设置。
    * **`cloud-provider-kind`：** 已包含在 `install.sh` 脚本中，用于模拟 Kind 的 LoadBalancer 服务。
* **简化说明：** 最初考虑使用 KubeEdge (CloudCore + EdgeCore) 以进行更真实的边缘模拟，但由于在尝试在同一台宿主机上同时运行 CloudCore (在 Kind 中) 和 EdgeCore 时遇到了设置复杂性和冲突 (kubelet vs. EdgeCore)，因此已将其从计划中移除。核心的 RL 目标在 Kind Kubernetes 环境中仍然可以实现。

## 3. 工作负载
* **应用：** **MobileNetV4** 图像分类模型。
* **部署：** 作为推理服务部署在 Kind 集群内的 Kubernetes Deployment 中。
    * **服务方法：** vLLM (如 `scripts/vllm-k8s.yaml` 示例所示) 主要为大语言模型设计，**不适用于** MobileNetV4 (卷积神经网络)。MobileNetV4 将使用适合 CNN 的专用推理服务器提供服务，例如：
        * **Triton Inference Server (推荐):** 支持多种框架，在 GPU 上表现良好。
        * **自定义 Python 服务 (FastAPI/Flask):** 如果需要特定的自定义逻辑，则提供灵活性。
        * (TensorFlow Serving 或 TorchServe 也是可行的备选方案)。
    * 可以借鉴 `scripts/vllm-k8s.yaml` 中的 Kubernetes 清单结构 (Deployment, Service)，但容器的 `image`、`command` 和 `args` 必须更改以适配所选的 MobileNetV4 服务方案。
* **GPU 需求：** 该服务将请求 `nvidia.com/gpu: 1` 资源。

## 4. 强化学习 (RL) 代理
* **算法：** **PPO (Proximal Policy Optimization)**。
* **状态空间 (S)：** 从 Kind 集群通过 Prometheus 收集的指标，可能包括：
    * 节点资源利用率（CPU、内存、GPU 利用率、GPU 显存使用率）。
    * Pod 资源利用率（CPU、内存、GPU 利用率）。
    * 推理请求队列长度。
    * 平均推理延迟。
    * 当前服务副本数。
    * （指标将进行规范化并仔细选择）。
* **动作空间 (A)：** 动态调整 Kind 内 MobileNetV4 服务的资源配置，例如：
    * 修改 HPA 目标利用率值。
    * 直接设置 MobileNetV4 Deployment 的副本数。
    * （可能）调整推理 Pod 的 CPU/GPU 资源请求/限制。
* **奖励函数 (R)：** 一个旨在平衡多个目标的复合函数：
    * 示例：`奖励 = w1 * (GPU利用率收益) + w2 * (其他资源效率) - w3 * (延迟惩罚) - w4 * (QoS违规惩罚)`。
    * 权重（`w1, w2, w3, w4`）和具体的收益/惩罚函数需要仔细调整。

## 5. 基线策略
* **调度器：** Kind 集群内的 Kubernetes 默认调度器。
* **自动缩放：** 为 MobileNetV4 Deployment 配置的标准水平 Pod 自动缩放器 (HPA)，基于 CPU 利用率或类似指标的静态阈值。

## 6. 监控与负载生成
* **监控：**
    * **`kube-prometheus-stack`：** 将通过 Helm chart (例如，来自 `prometheus-community` 的) 部署，以获得包括 Prometheus 和 Grafana 在内的全面监控解决方案。
    * **DCGM-Exporter：** 将部署在 Kind 内部，为 Prometheus 提供详细的 NVIDIA GPU 指标。
* **负载生成：**
    * 专用的负载生成工具/脚本。
    * **推荐工具：**
        * **Locust 或 k6：** 因其可脚本化、能生成多样化负载模式且易于在 Kubernetes 内部署而被优先选择。
    * 能够向 MobileNetV4 服务发送推理请求。
    * **负载模式：** 实验将在不同且明确定义的负载模式下运行：
        * 低负载
        * 高负载
        * 混合/可变负载

## 7. 关键性能指标 (KPI) 用于评估
将为基线和 RL 代理收集以下 KPI：
* 平均推理响应时间（延迟）。
* 百分位延迟（例如，p95, p99）。
* 系统吞吐量（每秒推理次数）。
* 节点和 Pod 资源利用率：
    * CPU 利用率 (%)
    * 内存使用率 (%)
    * GPU 利用率 (%)
    * GPU 显存使用率 (%)
* QoS 违规率（例如，超过预定义延迟阈值的请求百分比）。
* 随时间变化的活动服务副本数。

## 8. 实验流程/阶段

**阶段 1：环境搭建**
1.  准备好带有 RTX 3080 NVIDIA 驱动的宿主机 Ubuntu 24.04。
2.  运行调整后的 `install.sh` 脚本，安装 Docker、支持 GPU 的 Kind、NVIDIA Container Toolkit、`nvkind`。确认 `cloud-provider-kind` 已激活。
3.  通过 Helm 将 NVIDIA GPU Operator 安装到 Kind 集群中 (`--set driver.enabled=false`)。
4.  将监控栈：`kube-prometheus-stack` (Prometheus、Grafana) 和 `dcgm-exporter` 部署到 Kind 中。
5.  使用选定的服务方法（例如 Triton 或自定义 FastAPI 应用，并请求 GPU）将 MobileNetV4 推理服务部署到 Kind 中。
6.  设置负载生成器（例如 Locust 或 k6，部署在 Kind 中或宿主机上）。

**阶段 2：基线策略评估**
1.  在 Kind 中为 MobileNetV4 服务配置 Kubernetes 默认调度器和静态 HPA。
2.  针对该服务执行每种定义的负载模式（低、高、混合）。
3.  为保证统计有效性，每次运行重复 N 次（例如，N=5）。
4.  从 Prometheus 收集每次运行的所有已定义 KPI。
5.  计算平均 KPI 和置信区间。

**阶段 3：RL 代理训练**
1.  实现 PPO 代理，包括状态表示、动作执行（与 Kind 的 Kubernetes API 交互）和奖励计算。
2.  运行训练循环：代理与在 Kind 中运行的 MobileNetV4 服务在各种（或特定的训练）负载条件下交互。
3.  监控训练进度（奖励曲线、损失函数、验证场景下的评估指标（如果适用））。
4.  基于收敛性应用停止标准（例如，奖励平稳、性能饱和）。
5.  保存训练好的 RL 代理模型。

**阶段 4：RL 代理评估**
1.  加载训练好的最佳 RL 代理策略（评估模式）。
2.  执行与基线评估*完全相同*的负载模式（低、高、混合）。
3.  每次运行重复 N 次（例如，N=5）。
4.  从 Prometheus 收集所有已定义的 KPI。
5.  计算平均 KPI 和置信区间。

**阶段 5：结果分析与比较**
1.  比较 RL 代理与基线策略在每种负载模式下的平均 KPI（和置信区间）。
2.  使用适当的统计检验（例如，t 检验）来确定观察到的差异是否显著。
3.  使用表格和图表可视化结果。
4.  讨论研究结果，分析 RL 代理所做的权衡，并确定其优缺点。
5.  承认局限性（例如，本地设置代表一个高能力的边缘节点，而非资源受限的设备；**由于从设置中移除了 KubeEdge，因此未探索 KubeEdge 特定的功能和架构**）。
6.  概述未来工作。

## 9. 统计分析
* 每个实验条件（基线的负载模式、RL 代理的负载模式）将重复 N=5 次（或合适的次数）以进行统计分析。
* 结果将以平均值 ± 置信区间（例如，95% CI）的形式呈现。
* 将使用统计检验来比较 RL 代理和基线的性能。

## 10. 参考报告的考量
最初的研究报告"强化学习在开放与持续运行环境中的收敛性：定义、度量与评估"将指导以下方面：
* 定义和监控训练收敛性（例如，奖励曲线、策略稳定性）。
* RL 训练阶段的停止标准。
* 评估学习到的策略最终"好坏程度"的框架，不仅仅是训练指标，重点关注系统级 KPI 和在不同负载下的鲁棒性。

## 11. 最终产出
此实验计划的结果将记录下来，用于撰写论文和/或在相关会议（例如，IPCCC, SEC）上发表。
