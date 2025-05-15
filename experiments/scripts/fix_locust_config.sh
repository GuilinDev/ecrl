#!/bin/bash

# 使用sudo与microk8s kubectl
KUBECTL="sudo microk8s kubectl"

echo "检查locustfile-config ConfigMap的内容..."
CM_CONTENT=$($KUBECTL get configmap locustfile-config -n workloads -o jsonpath='{.data.locustfile\.py}')

if [[ "$CM_CONTENT" == *"Content will be mounted"* ]]; then
  echo "检测到占位符内容，正在修复ConfigMap..."
  
  # 删除旧的ConfigMap
  $KUBECTL delete configmap locustfile-config -n workloads
  
  # 从文件创建新的ConfigMap
  $KUBECTL create configmap locustfile-config --from-file=locustfile.py="$(dirname "$0")/locustfile.py" -n workloads
  
  # 重启Locust pod以应用新的ConfigMap
  echo "重启Locust pod以应用变更..."
  $KUBECTL rollout restart deployment/locust-master deployment/locust-worker -n workloads
  
  echo "等待Locust pod重启..."
  sleep 10
  
  # 检查pod状态
  echo "Locust pod状态："
  $KUBECTL get pods -n workloads -l app=locust-master
  $KUBECTL get pods -n workloads -l app=locust-worker
  
  echo "ConfigMap已修复。请等待pod完全启动后检查它们的状态。"
else
  echo "ConfigMap内容看起来没有问题，不需要修复。"
fi 