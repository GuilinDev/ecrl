#!/bin/bash

# Use microk8s kubectl
KUBECTL="microk8s kubectl"

# Create a temporary directory for model preparation
TEMP_DIR=$(mktemp -d)
MODEL_DIR="$TEMP_DIR/mobilenetv4/1"

# Create model directory structure
mkdir -p "$MODEL_DIR"

# Download MobileNetV4 model (you'll need to replace this with actual model download)
# This is a placeholder - you'll need to implement actual model download
echo "Downloading MobileNetV4 model..."
# Example: wget https://example.com/mobilenetv4.onnx -O "$MODEL_DIR/model.onnx"

# Create a temporary pod to copy files to PVC
cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-copy-pod
  namespace: workloads
spec:
  containers:
  - name: copy-container
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: model-storage
      mountPath: /mnt/models
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: mobilenetv4-model-pvc
EOF

# Wait for pod to be ready
$KUBECTL wait --for=condition=Ready pod/model-copy-pod -n workloads

# Copy model files to PVC
$KUBECTL cp "$TEMP_DIR/mobilenetv4" workloads/model-copy-pod:/mnt/models/

# Clean up
$KUBECTL delete pod model-copy-pod -n workloads
rm -rf "$TEMP_DIR"

echo "Model files have been copied to PVC" 