import numpy as np
import os, sys
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BinaryPerceptron:
    def __init__(self,  learning_rate, max_epochs, weight_init):
        self.data = None
        self.training_set_size = None
        self.validation_set_size = None
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = {'train_accuracy': [], 'val_accuracy': []}
        self.weight_init = weight_init

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
        if self.weight_init == "constant":
            self.weights = np.full(input_size, 0.5)
        elif self.weight_init == "gaussian":
            self.weights = np.random.normal(0, 0.01, input_size)
        elif self.weight_init == "zeros":
            self.weights = np.zeros(input_size)
        elif self.weight_init == "uniform":
            self.weights = np.random.uniform(-0.05, 0.05, input_size)
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

class MultiClassPerceptron:
    def __init__(self, learning_rate, max_epochs, error_patience_threshold, num_classes, rgb_or_gray, seed, weight_init):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.error_threshold = error_patience_threshold
        self.num_classes = num_classes
        self.perceptrons = [BinaryPerceptron(learning_rate, max_epochs, weight_init) for _ in range(num_classes)]
        self.training_history = {'train_accuracy': [], 'val_accuracy': []}
        self.data = None
        self.training_set_size = None
        self.validation_set_size = None
        self.rgb_or_gray = rgb_or_gray
        self.y_pred_classes = []
        self.test_labels = []
        self.test_predictions = []
        np.random.seed = seed # for reproducibility
        random.seed = seed # for reproducibility
        self.weight_init = weight_init

    def transform_images_to_vectors(self, image_files, label, normalize):
        """Add this method to MultiClassPerceptron class"""
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

    def preprocess_data(self, train_split, shuffle, normalize):
        # adapted from the BinaryPerceptron, extended to 10 folders
        folders = [
            ("./dataset/"+ self.rgb_or_gray +"/train/bart_simpson", 0),
            ("./dataset/"+ self.rgb_or_gray +"/train/lisa_simpson", 1),
            ("./dataset/"+ self.rgb_or_gray +"/train/homer_simpson", 2),
            ("./dataset/"+ self.rgb_or_gray +"/train/marge_simpson", 3),
            ("./dataset/"+ self.rgb_or_gray +"/train/ned_flanders", 4),
            ("./dataset/"+ self.rgb_or_gray +"/train/milhouse_van_houten", 5),
            ("./dataset/"+ self.rgb_or_gray +"/train/principal_skinner", 6),
            ("./dataset/"+ self.rgb_or_gray +"/train/charles_montgomery_burns", 7),
            ("./dataset/"+ self.rgb_or_gray +"/train/moe_szyslak", 8),
            ("./dataset/"+ self.rgb_or_gray +"/train/krusty_the_clown", 9)
        ]

        all_train_imgs, all_train_lbls = [], []
        all_val_imgs, all_val_lbls = [], []

        for folder_path, label in folders:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if shuffle:
                random.shuffle(files)

            split_idx = int(len(files) * train_split)
            train_files, val_files = files[:split_idx], files[split_idx:]

            train_imgs, train_lbls = self.perceptrons[0].transform_images_to_vectors(train_files, label, normalize)
            val_imgs, val_lbls = self.perceptrons[0].transform_images_to_vectors(val_files, label, normalize)

            all_train_imgs.extend(train_imgs)
            all_train_lbls.extend(train_lbls)
            all_val_imgs.extend(val_imgs)
            all_val_lbls.extend(val_lbls)

        train_data = np.array(all_train_imgs)
        train_labels = np.array(all_train_lbls)
        val_data = np.array(all_val_imgs)
        val_labels = np.array(all_val_lbls)

        if shuffle:
            perm_train = np.random.permutation(len(train_data))
            train_data, train_labels = train_data[perm_train], train_labels[perm_train]
            perm_val = np.random.permutation(len(val_data))
            val_data, val_labels = val_data[perm_val], val_labels[perm_val]

        self.data = {
            'train': {'images': train_data, 'labels': train_labels},
            'val': {'images': val_data, 'labels': val_labels}
        }
        self.training_set_size = len(train_data)
        self.validation_set_size = len(val_data)

        print("Train data shape:", train_data.shape)
        print("Val data shape:", val_data.shape)
        return self.data

    def fit(self):

        self.preprocess_data(train_split=0.8, shuffle=True, normalize=True)

        x_train, y_train = self.data['train']['images'], self.data['train']['labels']
        x_val, y_val = self.data['val']['images'], self.data['val']['labels']

        input_size = x_train.shape[1]
        for p in self.perceptrons:
            p.initialize_weights(input_size)

        best_val_acc = 0
        epochs_no_improvement = 0

        for epoch in range(self.max_epochs):
            print("Epoch", epoch)
            correct_train = 0
            for i in range(len(x_train)):
                x, y = x_train[i], y_train[i]
                scores = []
                for class_idx, p in enumerate(self.perceptrons):
                    target = 1 if y == class_idx else 0
                    score = np.dot(x, p.weights) + p.bias
                    y_pred = p.activation_function(score)
                    update = p.learning_rate * (target - y_pred)
                    p.weights += update * x
                    p.bias += update
                    scores.append(score)
                if np.argmax(scores) == y:
                    correct_train += 1

            acc_train = (correct_train / len(x_train)) * 100
            self.training_history['train_accuracy'].append(acc_train)
            print("Train acc:", acc_train)
            self.shuffle_training_data()

            correct_val = 0
            for i in range(len(x_val)):
                pred_class = self.predict(x_val[i])
                if  pred_class == y_val[i]:
                    correct_val += 1
            acc_val = (correct_val / len(x_val)) * 100
            self.training_history['val_accuracy'].append(acc_val)
            print("Val acc:", acc_val)
            print("")

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= self.error_threshold:
                print(f"Early stopping at epoch {epoch}, no validation accuracy improvement for {self.error_threshold} epochs")
                break


        # evaluate model using final params
        for i in range(len(x_val)):
            pred_class = self.predict(x_val[i])
            self.y_pred_classes.append(pred_class)


    def predict(self, x):
        scores = [np.dot(x, p.weights) + p.bias for p in self.perceptrons]
        return np.argmax(scores)

    def shuffle_training_data(self):
        # get random permutation indices
        train_indices = np.random.permutation(self.training_set_size)

        # apply same shuffle to images and labels
        self.data['train']['images'] = self.data['train']['images'][train_indices]
        self.data['train']['labels'] = self.data['train']['labels'][train_indices]

    def plot_pca(self):
        x = self.data['train']['images']
        y = self.data['train']['labels']

        # reduce to 2D
        pca = PCA(n_components=2)
        xp = pca.fit_transform(x)

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(xp[:, 0], xp[:, 1], c=y, cmap='tab10', alpha=0.6)

        # add legend for 10 classes
        plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("PCA of train set (10 classes)")
        # plt.tight_layout()
        plt.show()

    def plot_train_val_accuracy(self):
        train_acc_history = self.training_history['train_accuracy']
        val_acc_history = self.training_history['val_accuracy']
        epochs = range(1, len(train_acc_history) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_acc_history, label="Train Accuracy")
        plt.plot(epochs, val_acc_history, label="Validation Accuracy")

        plt.title("Training vs Validation Accuracy {"+ self.rgb_or_gray.upper()+"}")
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

    def load_test_data(self, normalize=True):
        test_images = []
        test_labels = []
        print("Loading test data from ./dataset/"+self.rgb_or_gray+"/test")
        for class_label, class_name in enumerate(sorted(os.listdir("./dataset/"+self.rgb_or_gray+"/test"))):
            folder_path = os.path.join("./dataset/"+self.rgb_or_gray+"/test", class_name)
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

            imgs, labels = self.transform_images_to_vectors(image_files, label=class_label, normalize=normalize)
            test_images.extend(imgs)
            test_labels.extend(labels)

        test_data = np.array(test_images)
        test_labels = np.array(test_labels)

        print("Test data shape:", test_data.shape)
        print("Test labels shape:", test_labels.shape)

        return test_data, test_labels

    def evaluate_test_set(self, normalize=True):


        test_images, test_labels = self.load_test_data(normalize=normalize)

        # Make predictions on test set
        test_predictions = []
        print("Making predictions on test set...")

        for i in range(len(test_images)):
            pred = self.predict(test_images[i])
            test_predictions.append(pred)

        # Store predictions for evaluation
        self.test_predictions = test_predictions
        self.test_labels = test_labels




    def get_classification_report(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix: {"+self.rgb_or_gray.upper()+"}\n", cm)

        print("Classification Report: {"+self.rgb_or_gray.upper()+"} \n" + classification_report(y_true, y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


def main():
    # multi_perc = MultiClassPerceptron(learning_rate=0.03, max_epochs=100, error_patience_threshold=25, num_classes=10, rgb_or_gray="grayscale", seed=42, weight_init="constant")
    # multi_perc = MultiClassPerceptron(learning_rate=0.06, max_epochs=100, error_patience_threshold=25, num_classes=10, rgb_or_gray= "rgb", seed=42, weight_init="constant")
    #
    # multi_perc.plot_pca()
    # multi_perc.fit()
    # multi_perc.plot_train_val_accuracy()
    # multi_perc.get_classification_report(multi_perc.data['val']['labels'], multi_perc.y_pred_classes)
    # multi_perc.evaluate_test_set(normalize=True)
    # multi_perc.get_classification_report(multi_perc.test_labels, multi_perc.test_predictions)

    learning_rates = [0.01, 0.02, 0.05, 0.08, 0.1, 0.4]
    max_epochs_list = [50, 80, 100, 120, 150]
    val_error_threshold = [10, 20, 35, 45]
    weight_inits = ["zeros", "constant", "gaussian", "uniform"]

    results = []

    # grid search for parameters and weight initializations
    for lr in learning_rates:
        for max_epochs in max_epochs_list:
            for patience in val_error_threshold:
                for init_name in weight_inits:

                    print(f"\n Training with lr={lr}, max_epochs={max_epochs}, patience={patience}, init={init_name}")


                    multi_perc = MultiClassPerceptron(
                        learning_rate=lr,
                        max_epochs=max_epochs,
                        error_patience_threshold=patience,
                        num_classes=10,
                        rgb_or_gray="grayscale",
                        seed=42,
                        weight_init=init_name
                    )

                    multi_perc.fit()

                    y_true = multi_perc.data['val']['labels']
                    y_pred = multi_perc.predict(multi_perc.data['val']['images'])
                    val_acc = (y_true == y_pred).mean() * 100

                    results.append({
                        "lr": lr,
                        "epochs": max_epochs,
                        "patience": patience,
                        "init": init_name,
                        "val_acc": val_acc
                    })
    results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    print("\n=== Best Hyperparameter Combos ===")
    for r in results[:5]:
        print(r)


if __name__ == "__main__":
    main()


