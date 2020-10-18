import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class RandomModelTrainer:
    def __init__(self, num_hidden_layers=2, num_epochs=15, num_models=4):
        self.num_hidden_layers = num_hidden_layers
        self.num_epochs = num_epochs
        self.num_models = num_models
        self.BATCH_SIZES = [2**i for i in range(0, 10)]
        self.ACTIVATIONS = ["relu", "cosine", "softplus", "sigmoid", "tanh"]
        self.models = []
        self.batch_sizes = []
        self.train_dataset, self.test_dataset, self.dataset_info = None, None, None
        # self.__load_datasets()

    def __load_datasets(self):
        def _normalize_img(img, label):
            img = tf.cast(img, tf.float32) / 255.
            return (img, label)

        [self.train_dataset, self.test_dataset], self.dataset_info = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True, with_info=True)
        self.train_dataset = self.train_dataset.map(_normalize_img)
        self.test_dataset = self.test_dataset.map(_normalize_img)
    
    def generate_random_models(self):
        random.seed(666)
        for model_number in range(self.num_models):
            print("[INFO] Creating model number %d" % (model_number))

            batch_size = self.BATCH_SIZES[random.randint(0, len(self.BATCH_SIZES)-1)]
            print("\t[INFO] Using batch size of %d" % (batch_size))
            self.batch_sizes.append(batch_size)

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

            num_of_hidden_layers = random.randint(0, self.num_hidden_layers)
            print("\t[INFO] Using %d hidden layers" % (num_of_hidden_layers))

            for layer in range(num_of_hidden_layers):
                activation_fn = self.ACTIVATIONS[random.randint(0, len(self.ACTIVATIONS)-1)]
                print("\t[INFO] Using hidden layer %d activation : %s" % (layer, activation_fn))
                if activation_fn == "cosine":
                    activation_fn = tf.math.cos
                hidden_units = random.randint(10, 784)
                print("\t[INFO] Using %d neurons in hidden layer %d" % (hidden_units, layer))
                model.add(tf.keras.layers.Dense(hidden_units, activation=activation_fn))

            model.add(tf.keras.layers.Dense(10, activation='softmax'))
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

            self.models.append(model)

    def view_models(self, models=None):
        if models is None:
            for model in self.models:
                model.summary()
        elif isinstance(models, int):
            try:
                self.models[models-1].summary()
            except IndexError as error:
                print(error)
        elif isinstance(models, list):
            for i in models:
                try:
                    self.models[int(i)-1].summary()
                except IndexError:
                    print("[ERROR] Please enter the correct index number(s) to view the model(s)")
        else:
            print("[ERROR] Type mismatch for argument 'models'")

    def train_models(self, models=None):
        if models is None:
            for i in range(0, len(self.models)-1):
                self.__train_model(i)
        elif isinstance(models, int):
            self.__train_model(models)
        elif isinstance(models, list):
            for i in models:
                self.__train_model(int(i)-1)
        else:
            print("[ERROR] Type mismatch for argument 'models'")
    
    def __train_model(self, index):
        tf.keras.backend.clear_session()
        try:
            model = self.models[index]
            if self.train_dataset is None or self.test_dataset is None:
                self.__load_datasets()
            train_dataset = self.train_dataset.shuffle(1024).batch(self.batch_sizes[index])
            test_dataset = self.test_dataset.batch(self.batch_sizes[index])
            
            print("[INFO] Starting training for model %d: " % (index))
            history = model.fit(train_dataset, epochs=self.num_epochs, 
                                validation_data=test_dataset)
            print(history)
            print("[INFO] Model %d trained!" % (index))

        except IndexError:
            print("[ERROR] Please enter the correct model index number(s)")
    
    def view_dataset_example(self, dataset='train'):
        if dataset not in ['train', 'test']:
            print("error")
        if self.train_dataset is None or self.test_dataset is None:
            self.__load_datasets()
        
        if dataset == "train":
            fig = tfds.show_examples(self.train_dataset, self.dataset_info)
            plt.show()
        else:
            fig = tfds.show_examples(self.test_dataset, self.dataset_info)
            plt.show()
