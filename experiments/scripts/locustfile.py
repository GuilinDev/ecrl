import time
import numpy as np
from locust import HttpUser, task, between
import tritonclient.http as httpclient
import tritonclient.utils as utils

class TritonUser(HttpUser):
    wait_time = between(1, 2)  # Wait 1-2 seconds between tasks
    triton_client = None

    def on_start(self):
        """Initialize Triton client when the user starts"""
        self.triton_client = httpclient.InferenceServerClient(
            url=f"{self.host}:8000",
            verbose=False
        )

    @task
    def inference_request(self):
        """Send inference request to Triton server"""
        try:
            # Prepare input data (random image data for testing)
            input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

            # Create input tensor
            input_tensor = httpclient.InferInput(
                "pixel_values",  # Input tensor name
                input_data.shape,
                "FP32"
            )
            input_tensor.set_data_from_numpy(input_data, binary_data=True)

            # Send inference request
            start_time = time.time()
            response = self.triton_client.infer(
                model_name="mobilenetv4",
                inputs=[input_tensor]
            )

            # Record response time
            response_time = time.time() - start_time
            self.environment.events.request.fire(
                request_type="inference",
                name="mobilenetv4",
                response_time=response_time * 1000,  # Convert to milliseconds
                response_length=0,
                exception=None
            )

        except Exception as e:
            self.environment.events.request.fire(
                request_type="inference",
                name="mobilenetv4",
                response_time=0,
                response_length=0,
                exception=e
            )