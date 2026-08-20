# python -m src.churn_prediction.run_all_training
from src.churn_prediction.train_ml import main as train_ml
from src.churn_prediction.train_ann import main as train_ann

def main() -> None:
    train_ml()
    train_ann()

if __name__ == "__main__":
    main()