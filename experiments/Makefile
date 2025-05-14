.PHONY: baseline clean-baseline

# Directory for storing experiment results
RESULTS_DIR := results
BASELINE_DIR := $(RESULTS_DIR)/baseline
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
BASELINE_RESULT := $(BASELINE_DIR)/baseline_$(TIMESTAMP)

# Use sudo with microk8s kubectl
KUBECTL := sudo microk8s kubectl

# Create necessary directories
$(shell mkdir -p $(BASELINE_DIR))

baseline: prepare-model deploy-baseline run-baseline collect-results

prepare-model:
	@echo "Preparing model files..."
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
	@$(KUBECTL) logs -n workloads deployment/mobilenetv4-triton-deployment > $(BASELINE_RESULT)/triton_logs.txt
	@$(KUBECTL) logs -n workloads deployment/locust-master > $(BASELINE_RESULT)/locust_master_logs.txt
	@$(KUBECTL) logs -n workloads deployment/locust-worker > $(BASELINE_RESULT)/locust_worker_logs.txt
	@echo "Results collected in $(BASELINE_RESULT)"

clean-baseline:
	@echo "Cleaning up baseline experiment..."
	@$(KUBECTL) delete namespace workloads --ignore-not-found=true
	@echo "Cleanup complete"

clean:
	@echo "Cleaning all experiment artifacts..."
	@rm -rf $(RESULTS_DIR)
	@make clean-baseline
