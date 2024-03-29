import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# mlflow packages
import mlflow
from mlflow import pyfunc
import mlflow.tensorflow

# Load in the data
iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# train_test_split
X, y = df.drop(columns = 'target'), df['target']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(x_train.shape, x_test.shape)

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

def mlflow_run(params, run_name="Tracking Experiment: iris logisitic regression"):
  with mlflow.start_run(run_name=r_name) as run:
    # get current run and experiment id
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
        
	# train model
    model = LogisticRegression(penalty = "l2", C=params["regularizer"])
    model.fit(x_train, y_train)

  return (experimentID, runID)

# Use the model
if __name__ == '__main__':
   # suppress any deprecated warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)

   regularizer = int(sys.argv[1]) if len(sys.argv) > 1 else 1.0
   params = {'regularizer': regularizer}
   
   (exp_id, run_id) = mlflow_run(params)

   print(f"Finished Experiment id={exp_id} and run id = {run_id}")