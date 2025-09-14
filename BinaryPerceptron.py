import numpy as np
import os, sys
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class BinaryPerceptron:
    def __init__(self,  learning_rate, max_epochs):
        self.data = None
        self.training_set_size = None
        self.validation_set_size = None
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = {'train_accuracy': [], 'val_accuracy': []}

    def show_random_samples(self, n):
        images = self.data['train']['images']
        labels = self.data['train']['labels']
        idxs = np.random.choice(len(images), n, replace=False)
        for i, idx in enumerate(idxs):
            plt.subplot(1, n, i + 1)
            if images.shape[1]==784:
                plt.imshow(images[idx].reshape(28, 28), cmap='gray')
            else:
                plt.imshow(images[idx].reshape(28, 28, 3))
            plt.title("Label: " + str(labels[idx]))
            plt.axis('off')
        plt.show()

    def plot_pca(self):
        x = self.data['train']['images']
        y = self.data['train']['labels']
        pca = PCA(n_components=2)
        xp = pca.fit_transform(x)
        plt.figure(figsize=(6, 5))
        plt.scatter(xp[:, 0], xp[:, 1], c=y, alpha=0.6)
        plt.title("PCA of train set")
        plt.show()

        # t-SNE (slower) - sample if large
        from sklearn.manifold import TSNE
        sample_idx = np.random.choice(len(x), min(1000, len(x)), replace=False)
        xs = x[sample_idx]
        ys = y[sample_idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        xts = tsne.fit_transform(xs)
        plt.figure(figsize=(6, 5))
        plt.scatter(xts[:, 0], xts[:, 1], c=ys, alpha=0.6)
        plt.title("t-SNE of train sample")
        plt.show()

    def transform_images_to_vectors(self, image_files, label, normalize):
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

    def preprocess_data(self, train_split, shuffle, normalize):

        folder0_files, folder0_alias = self.get_image_files_from_folder(folder_path="./dataset/rgb/train/bart_simpson",
                                                                        alias="bart")
        folder1_files, folder1_alias = self.get_image_files_from_folder(folder_path="./dataset/rgb/train/milhouse_van_houten",
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

        train_files = set(train_folder0_files + train_folder1_files)
        val_files = set(val_folder0_files + val_folder1_files)
        print(f"Overlap: {len(train_files.intersection(val_files))}")

        train_images0, train_labels0 = self.transform_images_to_vectors(train_folder0_files, label=0,
                                                                                  normalize=normalize)
        train_images1, train_labels1 = self.transform_images_to_vectors(train_folder1_files, label=1,
                                                                                  normalize=normalize)

        val_images0, val_labels0 = self.transform_images_to_vectors(val_folder0_files, label=0,
                                                                              normalize=normalize)
        val_images1, val_labels1 = self.transform_images_to_vectors(val_folder1_files, label=1,
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


        self.data = data
        self.training_set_size = len(train_data)
        self.validation_set_size = len(val_data)
        print("train_data shape: ", train_data.shape)
        print("val_data shape: ", val_data.shape)
        return data

    def shuffle_training_data(self):
        # get random permutation indices
        train_indices = np.random.permutation(self.training_set_size)

        # apply same shuffle to images and labels
        self.data['train']['images'] = self.data['train']['images'][train_indices]
        self.data['train']['labels'] = self.data['train']['labels'][train_indices]

    def plot_train_val_accuracy(self):
        train_acc_history = self.training_history['train_accuracy']
        val_acc_history = self.training_history['val_accuracy']
        epochs = range(1, len(train_acc_history) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_acc_history, label="Train Accuracy")
        plt.plot(epochs, val_acc_history, label="Validation Accuracy")

        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")

        # Handle if values are 0–1 instead of 0–100
        if max(max(train_acc_history), max(val_acc_history)) <= 1.0:
            plt.ylim(0, 1.05)
        else:
            plt.ylim(0, 100)

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

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
        return y_predicted

    def fit(self):
        for epoch in range(self.max_epochs):
            # training phase
            correct_predictions_train = 0
            print("Starting epoch: " + str(epoch))
            for i in range(self.training_set_size):
                x_sample = self.data['train']['images'][i]  # Single image vector (784)
                y_sample = self.data['train']['labels'][i]  # Single label (0 or 1)
                y_pred_on_train = self.apply_learning_rule(x_sample, y_sample)
                if y_pred_on_train == y_sample:
                    correct_predictions_train += 1
            accuracy_train = (correct_predictions_train / self.training_set_size ) * 100
            self.training_history['train_accuracy'].append(accuracy_train)
            print("Training accuracy after epoch " + str(epoch) + ":  "+ str( accuracy_train) + "%")
            print("Finished epoch: " + str(epoch))
            # shuffle training set
            self.shuffle_training_data()


            # evaluate on validation set
            correct_predictions_val = 0
            for i in range(self.validation_set_size):
                x_val = self.data['val']['images'][i]
                y_val = self.data['val']['labels'][i]
                y_pred = self.predict(x_val)
                if y_pred == y_val:
                    correct_predictions_val += 1
            accuracy_val = (correct_predictions_val / self.validation_set_size) * 100
            self.training_history['val_accuracy'].append(accuracy_val)
            print("Validation accuracy after epoch " + str(epoch) + ":  "+ str( accuracy_val ) + "%")
            print("\n")
            print("-----------------------------------------------------")
            print("\n")


    def predict(self, x_val):
        linear_output = np.dot(x_val, self.weights) + self.bias
        return self.activation_function(linear_output)


if __name__ == "__main__":
    perc = BinaryPerceptron( learning_rate=0.01, max_epochs=25)
    perc.preprocess_data(train_split=0.8, shuffle=True, normalize=True)

    perc.show_random_samples(n=6)
    perc.plot_pca()
    perc.initialize_weights(input_size=perc.data['train']['images'].shape[1])

    perc.fit()
    perc.plot_train_val_accuracy()


