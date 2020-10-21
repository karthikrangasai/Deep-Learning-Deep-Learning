import tensorflow as tf

def get_models():
    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model1.add(tf.keras.layers.Dense(512, activation='relu'))
    model1.add(tf.keras.layers.Dense(10, activation='softmax'))

    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model2.add(tf.keras.layers.Dense(512, activation='sigmoid'))
    model2.add(tf.keras.layers.Dense(10, activation='softmax'))

    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model3.add(tf.keras.layers.Dense(512, activation='tanh'))
    model3.add(tf.keras.layers.Dense(10, activation='softmax'))

    model4 = tf.keras.models.Sequential()
    model4.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model4.add(tf.keras.layers.Dense(512, activation='relu'))
    model4.add(tf.keras.layers.Dense(256, activation='relu'))
    model4.add(tf.keras.layers.Dense(128, activation='relu'))
    model4.add(tf.keras.layers.Dense(10, activation='softmax'))

    model5 = tf.keras.models.Sequential()
    model5.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model5.add(tf.keras.layers.Dense(512, activation='tanh'))
    model5.add(tf.keras.layers.Dense(256, activation='tanh'))
    model5.add(tf.keras.layers.Dense(128, activation='tanh'))
    model5.add(tf.keras.layers.Dense(10, activation='softmax'))

    model6 = tf.keras.models.Sequential()
    model6.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model6.add(tf.keras.layers.Dense(512, activation='sigmoid'))
    model6.add(tf.keras.layers.Dense(256, activation='sigmoid'))
    model6.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    model6.add(tf.keras.layers.Dense(10, activation='softmax'))

    model7 = tf.keras.models.Sequential()
    model7.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model7.add(tf.keras.layers.Dense(10, activation='softmax'))

    model8 = tf.keras.models.Sequential()
    model8.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model8.add(tf.keras.layers.Dense(387, activation='relu'))
    model8.add(tf.keras.layers.Dense(10, activation='softmax'))

    model9 = tf.keras.models.Sequential()
    model9.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model9.add(tf.keras.layers.Dense(387, activation='relu'))
    model9.add(tf.keras.layers.Dense(387, activation='relu'))
    model9.add(tf.keras.layers.Dense(10, activation='softmax'))

    model10 = tf.keras.models.Sequential()
    model10.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model10.add(tf.keras.layers.Dense(387, activation='relu'))
    model10.add(tf.keras.layers.Dense(387, activation='relu'))
    model10.add(tf.keras.layers.Dense(387, activation='relu'))
    model10.add(tf.keras.layers.Dense(387, activation='relu'))
    model10.add(tf.keras.layers.Dense(10, activation='softmax'))

    model11 = tf.keras.models.Sequential()
    model11.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model11.add(tf.keras.layers.Dense(128, activation='tanh'))
    model11.add(tf.keras.layers.Dense(256, activation='tanh'))
    model11.add(tf.keras.layers.Dense(512,activation='tanh'))
    model11.add(tf.keras.layers.Dense(10, activation='softmax'))

    model12 = tf.keras.models.Sequential()
    model12.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model12.add(tf.keras.layers.Dense(128, activation='relu'))
    model12.add(tf.keras.layers.Dense(256, activation='relu'))
    model12.add(tf.keras.layers.Dense(512,activation='relu'))
    model12.add(tf.keras.layers.Dense(10, activation='softmax'))

    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model1, model12]

    return models

