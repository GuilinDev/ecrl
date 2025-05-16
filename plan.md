# Experiment Plan for RL-based Resource Management in Kubernetes

## Overview

This project aims to develop a Reinforcement Learning (RL) agent to optimize resource allocation for inference workloads in Kubernetes. The agent will dynamically adjust resource allocations to maximize GPU utilization while maintaining Quality of Service (QoS) requirements.

## Experimental Setup

### Infrastructure
- **Kubernetes Cluster**: MicroK8s on local machine
- **Inference Server**: Triton Inference Server
- **Model**: MobileNetV4
- **Load Generator**: Locust

### Metrics
- **QoS Metrics**: Latency (average, P95, P99), throughput, success rate
- **Resource Metrics**: GPU utilization, memory usage, CPU usage
- **Model Metrics**: Inference accuracy, model responsiveness

## Experiment Phases

### Phase 1: Baseline Experiments (Current)
- Deploy MobileNetV4 model on Triton Inference Server with static resource allocation
- Generate synthetic load using Locust
- Collect QoS metrics and resource utilization data
- Evaluate model responsiveness using synthetic data
- Establish baseline performance for comparison with RL agent

### Phase 2: RL Agent Development
- Design state space (resource metrics, QoS metrics)
- Design action space (resource allocation adjustments)
- Design reward function (balancing resource efficiency and QoS)
- Implement RL agent using PPO or SAC algorithm
- Train agent in simulation environment

### Phase 3: RL Agent Evaluation
- Deploy RL agent to control resource allocation in Kubernetes
- Compare performance against baseline
- Analyze improvements in resource utilization and QoS
- Fine-tune agent parameters

### Phase 4: Advanced Scenarios (Optional)
- Multi-model deployment
- Varying load patterns
- Resource contention scenarios
- Fault tolerance testing

## Detailed Experiment Specifications

### Baseline Experiment
- **Model Configuration**: MobileNetV4 deployed on Triton Inference Server
- **Resource Allocation**: 
  - CPU: 2 cores
  - Memory: 4GB
  - GPU: 1 (shared)
- **Load Pattern**: 
  - Constant load: 10 users
  - Step load: 10, 20, 50 users
  - Ramp load: 1-50 users over 5 minutes
- **Metrics Collection**:
  - QoS metrics from Locust
  - Resource metrics from Kubernetes metrics server
  - Model responsiveness using synthetic data

### RL Agent Experiment
- **State Space**:
  - Current resource allocation (CPU, memory, GPU)
  - Current QoS metrics (latency, throughput)
  - Current load (requests per second)
- **Action Space**:
  - Adjust CPU allocation (±0.5 cores)
  - Adjust memory allocation (±512MB)
  - Adjust batch size
  - Adjust number of model instances
- **Reward Function**:
  - R = w1·(GPU_util_gain) + w2·(resource_efficiency) - w3·(latency_penalty) - w4·(qos_violation)
  - QoS violation defined as latency > target_latency or success_rate < target_success_rate

## Timeline

1. **Week 1**: Setup infrastructure and baseline experiments
2. **Week 2**: Analyze baseline results and design RL agent
3. **Week 3**: Implement and train RL agent
4. **Week 4**: Evaluate RL agent and compare with baseline
5. **Week 5**: Fine-tune agent and document results

## Next Steps

1. Complete baseline experiments with different load patterns
2. Design and implement RL agent
3. Create simulation environment for initial training
4. Deploy and evaluate RL agent in Kubernetes
