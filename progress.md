# Experiment Progress Log

## Phase 1: Baseline Experiments

### 2025-05-15: Initial Setup and Baseline Experiments

#### Infrastructure Setup
- ✅ Set up MicroK8s Kubernetes cluster
- ✅ Enabled required add-ons (dns, storage, metrics-server)
- ✅ Created Kubernetes manifests for Triton Inference Server
- ✅ Created Kubernetes manifests for Locust load generator

#### Model Preparation
- ✅ Downloaded MobileNetV4 model
- ✅ Converted model to ONNX format
- ✅ Created Triton model repository structure
- ✅ Configured model for Triton Inference Server

#### Baseline Experiments
- ✅ Deployed MobileNetV4 on Triton Inference Server
- ✅ Deployed Locust load generator
- ✅ Ran baseline experiments with constant load (10 users)
- ✅ Collected QoS metrics (latency, throughput, success rate)
- ✅ Evaluated model responsiveness using synthetic data

#### Data Collection
- ✅ Collected QoS metrics from Locust
- ✅ Collected resource utilization data from Kubernetes
- ✅ Evaluated model responsiveness with synthetic data

#### Findings
1. **QoS Metrics**:
   - Average latency: ~72 ms
   - P95 latency: ~74 ms
   - P99 latency: ~81 ms
   - Throughput: ~13 requests/second
   - Success rate: 100%

2. **Model Responsiveness**:
   - Model successfully processes all synthetic inputs
   - Consistent output distribution (all predictions favor class 644)
   - Stable latency across requests

3. **Challenges**:
   - Unable to evaluate true model accuracy with Tiny ImageNet due to class mapping issues
   - Used synthetic data to evaluate model responsiveness instead

### Next Steps

1. **Complete Baseline Experiments**:
   - Run experiments with different load patterns (step load, ramp load)
   - Collect and analyze results for different load scenarios

2. **RL Agent Design**:
   - Define state space based on collected metrics
   - Define action space for resource adjustments
   - Design reward function balancing resource efficiency and QoS

3. **Simulation Environment**:
   - Create simulation environment for initial RL agent training
   - Implement environment interface compatible with RL libraries

## Challenges and Solutions

### Challenge 1: Model Accuracy Evaluation
- **Issue**: Unable to properly map Tiny ImageNet classes to ImageNet classes used by MobileNetV4
- **Attempted Solutions**:
  - Created class mapping scripts
  - Tried Top-5 accuracy evaluation
  - Experimented with direct class comparison
- **Current Solution**: Used synthetic data to evaluate model responsiveness instead of true accuracy
- **Future Work**: If accuracy evaluation is critical, consider:
  - Using a model trained specifically on Tiny ImageNet
  - Creating a proper class mapping with expert knowledge
  - Using a simpler dataset with clear class mapping

### Challenge 2: Kubernetes Resource Metrics
- **Issue**: Need to collect detailed GPU metrics for RL agent
- **Solution**: Plan to use DCGM exporter for detailed GPU metrics in next phase

## Resources and References

- **Tiny ImageNet Dataset**: Located at `/home/guilin/allProjects/ecrl/data/tiny-imagenet/tiny-imagenet-200`
- **Model Repository**: Located at `/home/guilin/allProjects/ecrl/models`
- **Experiment Results**: Located at `/home/guilin/allProjects/ecrl/experiments/results/baseline`
- **Evaluation Scripts**: Located at `/home/guilin/allProjects/ecrl/experiments/scripts`
