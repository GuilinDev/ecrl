#!/bin/bash

# Create namespace if it doesn't exist
kubectl create namespace workloads --dry-run=client -o yaml | kubectl apply -f -

# Apply PVC and Triton deployment
echo "Deploying Triton server..."
kubectl apply -f mobilenetv4-triton-deployment.yaml

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
kubectl wait --for=condition=available deployment/mobilenetv4-triton-deployment -n workloads --timeout=300s

# Update Locust configmap with the locustfile
echo "Updating Locust configuration..."
kubectl create configmap locustfile-config --from-file=locustfile.py -n workloads --dry-run=client -o yaml | kubectl apply -f -

# Deploy Locust
echo "Deploying Locust..."
kubectl apply -f locust-deployment.yaml

# Wait for Locust to be ready
echo "Waiting for Locust to be ready..."
kubectl wait --for=condition=available deployment/locust-master -n workloads --timeout=300s
kubectl wait --for=condition=available deployment/locust-worker -n workloads --timeout=300s

# Port forward Locust web interface
echo "Starting port forward for Locust web interface..."
kubectl port-forward -n workloads svc/locust-master 8089:8089 &

# Print instructions
echo "Baseline experiment setup complete!"
echo "Access Locust web interface at http://localhost:8089"
echo "Configure your test with:"
echo "- Number of users: 10"
echo "- Spawn rate: 1"
echo "- Host: http://mobilenetv4-triton-svc.workloads.svc.cluster.local:8000"
echo ""
echo "Press Ctrl+C to stop the port forward when done" 