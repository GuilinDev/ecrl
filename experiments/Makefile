.PHONY: baseline clean-baseline download-hf-model prepare-model deploy-baseline run-baseline collect-results clean

# Directory for storing experiment results
RESULTS_DIR := results
BASELINE_DIR := $(RESULTS_DIR)/baseline
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
BASELINE_RESULT := $(BASELINE_DIR)/baseline_$(TIMESTAMP)

# Use sudo with microk8s kubectl
KUBECTL := sudo microk8s kubectl

# Python interpreter (ensure it has torch and torchvision)
PYTHON := python3 # Ensure this python has huggingface_hub installed

# Create necessary directories
$(shell mkdir -p $(BASELINE_DIR) experiments/models/mobilenetv4/1)

download-hf-model:
	@echo "Downloading ONNX model from Hugging Face Hub..."
	@$(PYTHON) ./scripts/download_hf_onnx_model.py
	@echo "ONNX model download complete. Check experiments/models/mobilenetv4/1/model.onnx"

baseline: download-hf-model prepare-model deploy-baseline run-baseline collect-results

prepare-model:
	@echo "Preparing model files for PVC..."
	@./scripts/prepare_model_pvc.sh

deploy-baseline:
	@echo "Deploying baseline components..."
	@./scripts/run_baseline_experiment.sh

run-baseline:
	@echo "Running baseline experiment..."
	@echo "Please configure and start the test in Locust web interface (http://localhost:8089)"
	@echo "Press Enter when the test is complete..."
	@bash -c 'read -p "Press Enter to continue..."'

collect-results:
	@echo "Collecting experiment results..."
	@mkdir -p $(BASELINE_RESULT)
	@$(KUBECTL) get pods -n workloads -o wide > $(BASELINE_RESULT)/pod_status.txt
	@$(KUBECTL) logs -n workloads deployment/mobilenetv4-triton-deployment --all-containers=true > $(BASELINE_RESULT)/triton_logs.txt 2>/dev/null || echo "No triton logs found or error collecting."
	@$(KUBECTL) logs -n workloads deployment/locust-master > $(BASELINE_RESULT)/locust_master_logs.txt 2>/dev/null || echo "No locust-master logs found or error collecting."
	# Try to get logs from one of the locust workers if the deployment exists
	@WORKER_POD=$($(KUBECTL) get pods -n workloads -l app=locust-worker -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && \
	    if [ ! -z "$$WORKER_POD" ]; then \
	        $(KUBECTL) logs -n workloads $$WORKER_POD > $(BASELINE_RESULT)/locust_worker_logs.txt; \
	    else \
	        echo "No locust-worker pods found or error collecting logs."; \
	    fi
	@echo "Results collected in $(BASELINE_RESULT)"

clean-baseline:
	@echo "Cleaning up baseline experiment..."
	@$(KUBECTL) delete namespace workloads --ignore-not-found=true
	@echo "Cleanup complete"

clean:
	@echo "Cleaning all experiment artifacts..."
	@rm -rf $(RESULTS_DIR)
	@rm -f experiments/models/mobilenetv4/1/model.onnx
	@make clean-baseline
