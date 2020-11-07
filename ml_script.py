# Drive link for trained models and the dataset
# https://drive.google.com/drive/folders/1BHXkD4BReFjodlyh8_PADFqxykT5sgkm?usp=sharing

#-----------------------------------------------------IMPORTING LIBRARIES-----------------------------------------------------#

import numpy as np
import pandas as pd
import pickle # For saving the trained model
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
# from pprint import pprint

#-----------------------------------------------------DATA PRE-PROCESSING-----------------------------------------------------#

train_data = pd.read_csv("./models/dataset/mnist_train.csv")
test_data = pd.read_csv("./models/dataset/mnist_test.csv")
# Training data
x_train = train_data.iloc[:, 1:].astype(float)
y_train = train_data.iloc[:, 0]
x_train = x_train/255.0
# Testing data
x_test = test_data.iloc[:, 1:].astype(float)
y_test = test_data.iloc[:, 0]
x_test = x_test/255.0
# Printing data dimensions
print('x_train shape =', x_train.shape)
print('y_train shape =', y_train.shape)
print('x_test shape =', x_test.shape)
print('y_test shape =', y_test.shape)

class_names = [0,1,2,3,4,5,6,7,8,9]

#-------------------------------------------------LEARNING CURVE PLOT FUNCTION------------------------------------------------#

## PLOT LEARNING CURVE
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
	if axes is None:
		_, axes = plt.subplots(1, 1, figsize=(6, 5))

	axes.set_title(title)
	axes.set_xlabel("Training examples")
	axes.set_ylabel("Score")

	train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
																			train_sizes=train_sizes,
																			return_times=True)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	fit_times_mean = np.mean(fit_times, axis=1)
	fit_times_std = np.std(fit_times, axis=1)
	axes.grid()
	axes.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color="r")
	axes.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha=0.1, color="g")
	axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
	axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
	axes.legend(loc="best")
	plt.savefig("models/" + title + ".png")

fig, axes = plt.subplots(1, 1, figsize=(10, 15))

#---------------------------------------------------SAVING CONFUSION MATRIX----------------------------------------------------#

def save_confusion_matrix(name, classifier):
	disp = plot_confusion_matrix(classifier, x_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
	disp.ax_.set_title(name)
	plt.savefig("models/" + name + "_confusion_matrix.png")

#------------------------------------------------GRID SEARCH TUNING HYPERPARAMS------------------------------------------------#

def hyperparameter_tuning(classifier, params_grid):
	model_gs = GridSearchCV(estimator = classifier, param_grid = params_grid, scoring = 'accuracy', cv = 3, verbose = 3)
	model_gs.fit(x_train, y_train)
	best_score = model_gs.best_score_
	best_params = model_gs.best_params_
	return best_score, best_params

#-------------------------------------------------SAVING AND LOADING THE MODEL-------------------------------------------------#

def save_model(model, filename):
	pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
	loaded_model = pickle.load(open(filename, 'rb'))
	result = loaded_model.score(x_test, y_test)
	print(result)
	return loaded_model

#---------------------------------------------------MACHINE LEARNING MODELS----------------------------------------------------#

ML_MODEL = input("Enter Model (SVM/LR/RF/DT): ")


## SUPORT VECTOR MACHINE
if ML_MODEL == "SVM":
	print("Running support vector machine model...")
	try:
		# Load the model from disk
		classifier = load_model("models/saved_ML_models/finalized_svm.sav")
		# pprint(vars(classifier))
	
	except:
		# Linear Model
		classifier = SVC(kernel = 'linear', random_state = 0)
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy for linear model =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")

		# Hyperparameter Tuning using Grid Search
		classifier = SVC(kernel = 'rbf')
		params = [{'C':[1, 10, 100], 'gamma':[0.01, 0.001, 0.0001]}]
		best_score, best_params = hyperparameter_tuning(classifier, params)
		classifier = SVC(C = best_params['C'], gamma = best_params['gamma'], kernel="rbf")
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy for model with tuned hyperparams =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")
		print(confusion_matrix(y_true = y_test, y_pred = y_pred))
		# Save model on disk
		save_model(classifier, "models/saved_ML_models/finalized_svm.sav")
		
	# Save confusion matrix
	save_confusion_matrix("SVM", classifier)

	# Learning curve code
	plot_learning_curve(classifier, "SVM", x_train, y_train, axes=None, ylim=(0.6, 1.01),  cv=3)


## LOGISTIC REGRESSION
elif ML_MODEL == "LR":
	print("Running logistic regression model...")
	try:
		# Load the model from disk
		classifier = load_model("models/saved_ML_models/finalized_lr.sav")
		# pprint(vars(classifier))

	except:
		# Tuning model hyperparameters with grid search gave the same model
		classifier = LogisticRegression(penalty = 'l2', solver = 'lbfgs', max_iter = 1000)
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy of LR =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")

		# Hyperparameter Tuning using Grid Search
		params = [{'solver':['lbfgs', 'sag', 'saga']}]
		best_score, best_params = hyperparameter_tuning(classifier, params)
		classifier = LogisticRegression(penalty = 'l2', solver = best_params['solver'], max_iter = 1000)
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy for LR model with tuned hyperparams =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")
		# Save model on disk
		save_model(classifier, "models/saved_ML_models/finalized_lr.sav")

	# Save confusion matrix
	save_confusion_matrix("LogisticRegression", classifier)
	
	# Learning curve code
	plot_learning_curve(classifier, "LogisticRegression", x_train, y_train, axes=None, ylim=(0.6, 1.01),  cv=3)


## RANDOM FOREST
elif ML_MODEL == "RF":
	print("Running random forest model...")
	try:
		# Load the model from disk
		classifier = load_model("models/saved_ML_models/finalized_rf.sav")
		# pprint(vars(classifier))

	except:
		# Tuning model hyperparameters with grid search gave the same model
		classifier = RandomForestClassifier()
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy for random forest model =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")

		# Hyperparameter Tuning using Grid Search
		params = [{'criterion':['gini','entropy'], 'max_depth':[15,20,None], 'min_samples_leaf':[1,3],
					'min_weight_fraction_leaf':[0., 0.0001], 'n_estimators':[10, 100]}]
		best_score, best_params = hyperparameter_tuning(classifier, params)
		classifier = RandomForestClassifier(criterion=best_params['criterion'],
											max_depth=best_params['max_depth'],
											min_samples_leaf=best_params['min_samples_leaf'],
											min_weight_fraction_leaf=best_params['min_weight_fraction_leaf'],
											n_estimators=best_params['n_estimators'])
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy for RF model with tuned hyperparams =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")
		# Save model on disk
		save_model(classifier, "models/saved_ML_models/finalized_rf.sav")

	# Save confusion matrix
	save_confusion_matrix("RandomForest", classifier)

	# Learning curve code
	plot_learning_curve(classifier, "RandomForest", x_train, y_train, axes=None, ylim=(0.6, 1.01),  cv=3)


## DECISION TREE
elif ML_MODEL == "DT":
	print("Running decision tree model...")
	try:
		# Load the model from disk
		classifier = load_model("models/saved_ML_models/finalized_dt.sav")
		# pprint(vars(classifier))

	except:
		# Default model
		classifier = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=3, min_weight_fraction_leaf=0.0001)
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy of DT =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")

		# Hyperparameter Tuning using Grid Search
		params = [{'criterion':['gini','entropy'], 'max_depth':[15,20,None], 'min_samples_leaf':[1,3], 'min_weight_fraction_leaf':[0., 0.0001]}]
		best_score, best_params = hyperparameter_tuning(classifier, params)
		classifier = DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'],
				min_samples_leaf=best_params['min_samples_leaf'], min_weight_fraction_leaf=best_params['min_weight_fraction_leaf'])
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		print("Accuracy for DT model with tuned hyperparams =", accuracy_score(y_true = y_test, y_pred = y_pred), "\n")
		# Save model on disk
		save_model(classifier, "models/saved_ML_models/finalized_dt.sav")

	# Save confusion matrix
	save_confusion_matrix("DT", classifier)

	# Learning curve code
	plot_learning_curve(classifier, "DT", x_train, y_train, axes=None, ylim=(0.6, 1.01),  cv=3)
	

else:
	print("INVALID INPUT...")
