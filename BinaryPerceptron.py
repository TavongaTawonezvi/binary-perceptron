import numpy as np
import os, sys
from PIL import Image
import random

class BinaryPerceptron:
    def __init__(self, data,  learning_rate=0.02, max_epochs=10):
        self.data = data
        self.training_set_size = None
        self.validation_set_size = None
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = {'loss': [], 'accuracy': []}

    def transform_greyscale_images_to_vectors(self, image_files, label, normalize=True):
        images = []
        labels = []
        print("Loading image files... ")
        for file in image_files:
            try:

                img = Image.open(file)
                img_array = np.array(img)
                if normalize:
                    img_array = img_array / 255.0  # Normalize pixel values
                    img_array = img_array.flatten()
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
                continue
        print("Transformed " + str(len(images)) + " images to vectors with label " + str(label))
        return images, labels


    def get_image_files_from_folder(self, folder_path, alias):

        files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        return [os.path.join(folder_path, f) for f in files], alias

    def preprocess_data(self, train_split=0.8, shuffle=True, normalize=True):

        folder0_files, folder0_alias = self.get_image_files_from_folder(folder_path="./dataset/grayscale/train/bart_simpson",
                                                                        alias="bart")
        folder1_files, folder1_alias = self.get_image_files_from_folder(folder_path="./dataset/grayscale/train/lisa_simpson",
                                                                        alias="lisa")

        print("found " + str(len(folder0_files)) + " files in " + folder0_alias + " folder")
        print("found " + str(len(folder1_files)) + " files in " + folder1_alias + " folder")

        if shuffle:
            random.shuffle(folder0_files)
            random.shuffle(folder1_files)

        split0 = int(len(folder0_files) * train_split)
        split1 = int(len(folder1_files) * train_split)

        train_folder0_files = folder0_files[:split0]
        train_folder1_files = folder1_files[:split1]
        val_folder0_files = folder0_files[split0:]
        val_folder1_files = folder1_files[split1:]

        print("train_folder0_files: ", len(train_folder0_files))
        print("train_folder1_files: ", len(train_folder1_files))
        print("val_folder0_files: ", len(val_folder0_files))
        print("val_folder1_files: ", len(val_folder1_files))

        train_images0, train_labels0 = self.transform_greyscale_images_to_vectors(train_folder0_files, label=0,
                                                                                  normalize=normalize)
        train_images1, train_labels1 = self.transform_greyscale_images_to_vectors(train_folder1_files, label=1,
                                                                                  normalize=normalize)

        val_images0, val_labels0 = self.transform_greyscale_images_to_vectors(val_folder0_files, label=0,
                                                                              normalize=normalize)
        val_images1, val_labels1 = self.transform_greyscale_images_to_vectors(val_folder1_files, label=1,
                                                                              normalize=normalize)

        train_data = np.array(train_images0 + train_images1)
        train_labels = np.array(train_labels0 + train_labels1)
        val_data = np.array(val_images0 + val_images1)
        val_labels = np.array(val_labels0 + val_labels1)

        if shuffle:
            train_perm = np.random.permutation(len(train_data))
            val_perm = np.random.permutation(len(val_data))
            train_data = train_data[train_perm]
            train_labels = train_labels[train_perm]
            val_data = val_data[val_perm]
            val_labels = val_labels[val_perm]

        data = {
            'train': {
                'images': train_data,
                'labels': train_labels
            },
            'val': {
                'images': val_data,
                'labels': val_labels
            }
        }

        self.training_set_size = len(train_data)
        self.validation_set_size = len(val_data)
        print("train_data shape: ", train_data.shape)
        print("val_data shape: ", val_data.shape)
        return data

    def initialize_weights(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
    def activation_function(self, out):
        return 1 if out >= 0 else 0

    def apply_learning_rule(self, x, y):
        linear_output = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)

        update = self.learning_rate * (y - y_predicted)
        self.weights += update * x
        self.bias += update

    def fit(self):
        for epoch in range(self.max_epochs):

            for i in range(self.training_set_size):
                x_sample = self.data['train']['images'][i]  # Single image vector (784)
                y_sample = self.data['train']['labels'][i]  # Single label (0 or 1)
                self.apply_learning_rule(x_sample, y_sample)
            
            # evaluate on validation set
            for i in range(self.validation_set_size):
                x_val = self.data['val']['images'][i]
                y_val = self.data['val']['labels'][i]
                y_pred = self.predict(x_val)


    def predict(self, x_val):
        linear_output = np.dot(x_val, self.weights) + self.bias
        return self.activation_function(linear_output)


if __name__ == "__main__":


