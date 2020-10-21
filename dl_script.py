import os
import numpy as np
from dl_models import get_models
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

def save_metric_graphical_history(history, folder, index):
    try:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        path = os.path.join(folder, ''.join(['model-', str(index), '-loss.png']))
        plt.savefig(path)
        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc='upper left')
        path = os.path.join(folder, ''.join(['model-', str(index), '-accuracy.png']))
        plt.savefig(path)
        plt.clf()

        return True
    except:
        return False

def main(folder, batch_size):
    train_dataset, test_dataset = tfds.load(name="mnist", 
                                            split=['train', 'test'], 
                                            as_supervised=True)
    train_dataset = train_dataset.map(_normalize_img)
    test_dataset = test_dataset.map(_normalize_img)

    train_dataset = train_dataset.shuffle(1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    if not os.path.exists(folder):
        os.mkdir(folder)

    models = get_models()
    for index, model in enumerate(models):
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=25, validation_data=test_dataset)
        print("[INFO] Trained model %d\n" % (index))
        if save_metric_graphical_history(history, folder, index) is not True:
            print("[ERROR] There was an error while saving the plots of tracked metrics during training")

    print("[INFO] All models trained!")
    return

if __name__ == '__main__':
    folder = os.path.join(os.getcwd(), "models")
    for batch_size in [256]: #[32, 64, 128, 256]:
        main(folder, batch_size)
