import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.trainer import Trainer

EXPERIMENT_NAME = "[PT] [Lisbon] [anaiscasilva] TaxiFareModel + V0.0.1"  

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("https://mlflow.lewagon.co/")

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

yourname = 'anaisacsilva'

if yourname is None:
    print("please define your name, il will be used as a parameter to log")

N = 100_000
df = get_data(nrows=N)
df = clean_data(df)
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
trainer = Trainer(X_train, y_train)
trainer.run()
rmse =trainer.evaluate(X_test, y_test)

#for model in ["linear", "Randomforest"]:
run = client.create_run(experiment_id)
client.log_metric(run.info.run_id, "rmse", rmse)
client.log_param(run.info.run_id, "model", 'LinearRegression')
client.log_param(run.info.run_id, "student_name", yourname)

