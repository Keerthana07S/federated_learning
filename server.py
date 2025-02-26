import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
import os
from joblib import load
import numpy as np

model_files = [f for f in os.listdir() if f.startswith("finalized_model")]
models = [load(f) for f in model_files]
weights = [model.get_weights() for model in models]
avg_weights = np.mean(weights, axis=0)
aggregated_model = models[0]  
aggregated_model.set_weights(avg_weights)
dump(aggregated_model, "aggregated_model_v_final.joblib")
