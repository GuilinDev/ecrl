#!/bin/bash

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Create namespace if it doesn't exist
$KUBECTL create namespace workloads --dry-run=client -o yaml | $KUBECTL apply -f -

# Apply PVC and Triton deployment
echo "Deploying Triton server..."
$KUBECTL apply -f "$(dirname "$0")/mobilenetv4-triton-deployment.yaml"

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
$KUBECTL wait --for=condition=available deployment/mobilenetv4-triton-deployment -n workloads --timeout=300s

# Delete existing Locust configmap if it exists, then create from file
echo "Updating Locust configuration..."
$KUBECTL delete configmap locustfile-config -n workloads --ignore-not-found=true
$KUBECTL create configmap locustfile-config --from-file=locustfile.py="$(dirname "$0")/locustfile.py" -n workloads

# Deploy Locust
echo "Deploying Locust..."
$KUBECTL apply -f "$(dirname "$0")/locust-deployment.yaml"

# Wait for Locust to be ready
echo "Waiting for Locust to be ready..."
$KUBECTL wait --for=condition=available deployment/locust-master -n workloads --timeout=300s
$KUBECTL wait --for=condition=available deployment/locust-worker -n workloads --timeout=300s

# Port forward Locust web interface
echo "Starting port forward for Locust web interface..."
$KUBECTL port-forward -n workloads svc/locust-master 8089:8089 &

# Print instructions
echo "Baseline experiment setup complete!"
echo "Access Locust web interface at http://localhost:8089"
echo "Configure your test with:"
echo "- Number of users: 10"
echo "- Spawn rate: 1"
echo "- Host: http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000"
echo ""
echo "Press Ctrl+C to stop the port forward when done" 