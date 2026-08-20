import os
import random
import cv2
import numpy as np
import tensorflow as tf

def load_data(
    data_path,
    class_names,
    image_size=64,
    max_images_per_class=None
):
    images = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"Folder not found: {class_path}")
            continue

        image_names = os.listdir(class_path)
        random.shuffle(image_names)
        if max_images_per_class is not None:
            image_names = image_names[:max_images_per_class]

        for image_name in image_names:
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            labels.append(label)

        print(
            f"{class_name}: "
            f"{len(image_names)} images loaded"
        )

    return (np.array(images), np.array(labels))

def data_loader(
    train_path,
    test_path,
    image_size,
    batch_size,
    validation_size=0.2,
    random_state=42
):
    train_data = (
        tf.keras.utils
        .image_dataset_from_directory(
            train_path,
            validation_split=validation_size,
            subset="training",
            seed=random_state,
            image_size=(
                image_size,
                image_size
            ),
            batch_size=batch_size
        )
    )

    validation_data = (
        tf.keras.utils
        .image_dataset_from_directory(
            train_path,
            validation_split=validation_size,
            subset="validation",
            seed=random_state,
            image_size=(
                image_size,
                image_size
            ),
            batch_size=batch_size
        )
    )

    test_data = (
        tf.keras.utils
        .image_dataset_from_directory(
            test_path,
            image_size=(
                image_size,
                image_size
            ),
            batch_size=batch_size,
            shuffle=False
        )
    )

    return (train_data, validation_data, test_data)

def tensorflow_dataset_to_numpy(
    dataset
):
    img_batches = []
    label_batches = []
    for imgs, labels in dataset:
        img_batches.append(imgs.numpy().astype(np.uint8))
        label_batches.append(labels.numpy().astype(np.uint8))

    all_imgs = np.concatenate(img_batches, axis=0)
    all_labels = np.concatenate(label_batches, axis=0)
    return (all_imgs, all_labels)