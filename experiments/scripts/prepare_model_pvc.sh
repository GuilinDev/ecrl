#!/bin/bash

# Use sudo with microk8s kubectl
KUBECTL="sudo microk8s kubectl"

# Relative path to the ONNX model from this script's location
# This assumes the script is in experiments/scripts and model is in experiments/models
LOCAL_MODEL_PATH="../../models/mobilenetv4/1/model.onnx"

# Create a temporary directory for model preparation
TEMP_DIR=$(mktemp -d)
MODEL_DIR_IN_TEMP="$TEMP_DIR/mobilenetv4/1"

# Create model directory structure in temp
mkdir -p "$MODEL_DIR_IN_TEMP"

echo "Looking for local ONNX model at: $LOCAL_MODEL_PATH"
if [ -f "$LOCAL_MODEL_PATH" ]; then
    echo "Local ONNX model found. Copying to temporary location: $MODEL_DIR_IN_TEMP/model.onnx"
    cp "$LOCAL_MODEL_PATH" "$MODEL_DIR_IN_TEMP/model.onnx"
else
    echo "ERROR: Local ONNX model not found at $LOCAL_MODEL_PATH!"
    echo "Please run the export script first (e.g., make export-model)."
    # Create a placeholder if model not found, so PVC copy doesn't fail, but Triton will likely fail.
    echo "This is a placeholder because the real model.onnx was not found." > "$MODEL_DIR_IN_TEMP/model.onnx"
    echo "Triton will likely fail to load this placeholder model."
fi

# Create namespace if it doesn't exist (moved here to ensure it exists before pod creation)
$KUBECTL get ns workloads > /dev/null 2>&1 || $KUBECTL create namespace workloads

# Create a temporary pod to copy files to PVC
# The PVC must exist or be dynamically provisioned in the 'workloads' namespace.
echo "Applying PVC definition (expecting it to exist or be created by mobilenetv4-triton-deployment.yaml)..."
# Ensure mobilenetv4-model-pvc is defined, typically in your Triton deployment YAML.

cat <<EOF | $KUBECTL apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-copy-pod
  namespace: workloads
spec:
  restartPolicy: Never # Ensure it doesn't keep retrying if there's an issue
  containers:
  - name: copy-container
    image: busybox
    command: ["/bin/sh", "-c", "echo 'Copying files...'; mkdir -p /mnt/models/mobilenetv4/1 && cp /temp_model_files/mobilenetv4/1/model.onnx /mnt/models/mobilenetv4/1/model.onnx && echo 'Copy complete. Verifying...' && ls -lR /mnt/models && echo 'Sleeping for a bit to allow volume to sync...' && sleep 5"]
    volumeMounts:
    - name: model-storage # This is the PVC
      mountPath: /mnt/models
    - name: temp-model-storage # This is the hostPath temporary model files
      mountPath: /temp_model_files
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: mobilenetv4-model-pvc
  - name: temp-model-storage # Mount the temporary directory from host
    hostPath:
      path: $TEMP_DIR # This is the mktemp directory
      type: Directory
EOF

echo "Waiting for model-copy-pod to complete..."
# Wait for the pod to complete, with a timeout
if $KUBECTL wait --for=condition=Succeeded pod/model-copy-pod -n workloads --timeout=120s; then
    echo "model-copy-pod completed successfully."
    echo "Logs from model-copy-pod:"
    $KUBECTL logs model-copy-pod -n workloads
else
    echo "ERROR: model-copy-pod did not succeed within timeout."
    echo "Current status of model-copy-pod:"
    $KUBECTL describe pod model-copy-pod -n workloads
    echo "Logs from model-copy-pod (if any):"
    $KUBECTL logs model-copy-pod -n workloads
fi

# Clean up the pod
echo "Deleting model-copy-pod..."
$KUBECTL delete pod model-copy-pod -n workloads --ignore-not-found=true

# Clean up the temporary directory from host
echo "Cleaning up temporary directory: $TEMP_DIR"
rm -rf "$TEMP_DIR"

echo "Model files preparation process complete." 