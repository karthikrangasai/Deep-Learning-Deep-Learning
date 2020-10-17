import random
import argparse
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

    def train_random_models(self):
        def _normalize_img(img, label):
            img = tf.cast(img, tf.float32) / 255.
            return (img, label)

        random.seed(666)
        for model_number in range(self.num_models):

            tf.keras.backend.clear_session()
            train_dataset, test_dataset = tfds.load(name="mnist", 
                                                    split=['train', 'test'], 
                                                    as_supervised=True)
            train_dataset = train_dataset.map(_normalize_img)
            test_dataset = test_dataset.map(_normalize_img)

            batch_size = self.BATCH_SIZES[random.randint(0, len(self.BATCH_SIZES)-1)]
            print("[INFO] Using batch size of %d" % (batch_size))
            train_dataset = train_dataset.shuffle(1024).batch(batch_size)
            test_dataset = test_dataset.batch(batch_size)

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

            nrof_hidden_layers = random.randint(0, self.num_hidden_layers)
            print("[INFO] Using %d hidden layers" % (nrof_hidden_layers))

            for layer in range(nrof_hidden_layers):
                activation_fn = self.ACTIVATIONS[random.randint(0, len(self.ACTIVATIONS)-1)]
                print("[INFO] Using hidden layer %d activation : %s" % (layer, activation_fn))
                if activation_fn == "cosine":
                    activation_fn = tf.math.cos
                hidden_units = random.randint(10, 784)
                print("[INFO] Using %d neurons in hidden layer %d" % (hidden_units, layer))
                model.add(tf.keras.layers.Dense(hidden_units, activation=activation_fn))

            model.add(tf.keras.layers.Dense(10, activation='softmax'))

            model.summary()
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy', tf.keras.metrics.Precision()])

            if self.num_models == 1:
                return model
            else:
                history = model.fit(train_dataset, epochs=self.num_epochs, 
                                    validation_data=test_dataset)
                print(history)
                print("[INFO] Model %d trained!" % (model_number))

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--hidden_layers", type=int, default=2, required=False,
                        help="Specifies maximum number of hidden layers in model")
    parser.add_argument("-e", "--epochs", type=int, default=15, required=False,
                        help="Specifies number of epochs for training")
    parser.add_argument("-m", "--nrof_models", type=int, default=4, required=False,
                        help="Specifies number of models to generate. If 1, returns an untrained model")
    args = vars(parser.parse_args())
    obj = RandomModelTrainer(num_hidden_layers=args['hidden_layers'],
                             num_epochs=args['epochs'],
                             num_models=args['nrof_models'])
    obj.train_random_models()