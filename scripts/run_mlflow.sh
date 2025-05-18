# scripts/run_mlflow.sh
mkdir -p mlruns
mlflow server \
   --backend-store-uri sqlite:///mlflow.db \
   --default-artifact-root ./mlruns \
   --host 0.0.0.0 \
   --port 5001
