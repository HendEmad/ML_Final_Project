from src.image_classification.train_ml import main as train_ml
from src.image_classification.train_ann import main as train_ann
from src.image_classification.train_cnn import main as train_cnn
from src.image_classification.train_vgg import main as train_vgg

def main() -> None:
    train_ml()
    train_ann()
    train_cnn()
    train_vgg()

if __name__ == "__main__":
    main()