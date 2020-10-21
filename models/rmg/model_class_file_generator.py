import os
import sys
import tensorflow as tf

def __write_class_start(output_file, class_name):
	if not isinstance(class_name, str):
		raise AssertionError("Attribute class_name has to be of the type String")

	output_file.write("import tensorflow as tf\n\n")
	output_file.write("class %s(tf.keras.Model):\n" % (class_name))
	output_file.write("    def __init__(self):\n")
	output_file.write("        super(%s, self).__init__()\n" % (class_name))
	output_file.write("        self.f = tf.keras.layers.Flatten(input_shape=(28, 28, 1))\n")
	return


def __write_class_init_method(output_file, layers):
	count = 1
	for layer in layers:
		if layer['type'] == "Dense":
			if layer['activation'] == "cos":
				output_file.write("        self.l%d = tf.keras.layers.Dense(%d, activation=tf.math.cos)\n" % (count, layer['units']))
			elif layer['activation'] == "softmax":
				output_file.write("        self.out = tf.keras.layers.Dense(%d, activation='softmax')\n\n" % (layer['units']))
			else:
				output_file.write("        self.l%d = tf.keras.layers.Dense(%d, activation='%s')\n" % (count, layer['units'], layer['activation']))
			count += 1
	output_file.write("\n")
	return

def __write_class_call_method(output_file, layers):
	output_file.write("    def call(self, x):\n")
	output_file.write("        x = self.f(x)\n")
	count = 1
	for layer in layers:
		if layer['type'] == "Dense" and not layer['activation'] == "softmax":
			output_file.write("        x = self.l%d(x)\n" % (count))
			count += 1
	output_file.write("        return self.out(x)\n")


def generate_class_file(model, file_path):
	if not isinstance(model, tf.keras.models.Sequential):
		raise AssertionError("Attribute model needs to be of type tensorflow.keras.models.Sequential")

	class_name = "MyModel"
	layers = []
	for layer in model.layers:
		l = {}
		if layer.__class__.__name__ == "Flatten":
			l['type'] = "Flatten"
			l['input_shape'] = layer.input_shape
			layers.append(l)
		elif layer.__class__.__name__ == "Dense":
			l['type'] = "Dense"
			l['units'] = layer.units
			l['activation'] = str(layer.activation).split(' ')[1]
			layers.append(l)
		else:
			raise Exception("Generation code for %s module hasn't been written yet." % (layer.__class__.__name__))

	with open(file_path, 'w') as output_file:
		__write_class_start(output_file, class_name)
		__write_class_init_method(output_file, layers)
		__write_class_call_method(output_file, layers)
