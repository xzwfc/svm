# SVM not sym
# based on ml13 and become a function to be used on try_logistics.py
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix #not for continuous variables if split n is 5, 5 confusion matrix, 
#A confusion matrix is a summary of prediction results on a classification problem. The number of correct 
#cooland incorrect predictions are summarized with count values and broken down by each class. 
from sklearn.metrics import accuracy_score #classification scores

def run_kfold(split_number, data, target, machine, use_accu=0, use_confusion=0):
	# pass
	kfold_object=KFold(n_splits =split_number) #KFold only uses each value once, so we separate the dataset into 4
	kfold_object.get_n_splits(data)

	results_r2 =[]
	results_c=[]
	results_a=[]
	for training_index, test_index in kfold_object.split(data):

		# print(training_index)
		# print(test_index)
		data_training, data_test=data[training_index], data[test_index]
		target_training, target_test=target[training_index], target[test_index]
		# machine=linear_model.LogisticRegression()
		machine.fit (data_training, target_training)
		prediction =machine.predict(data_test)
		if use_accu ==1:
			results_a.append(accuracy_score(target_test,prediction))
		if use_confusion==1:
			 results_c.append(confusion_matrix(target_test,prediction))
		results_r2.append(metrics.r2_score(target_test, prediction))

	return results_r2, results_a, results_c


if __name__=="__main__": #when importing this template, the lines below will be skipped

	df=pd.read_csv("logistic_dataset.csv")
	target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
	data = df.iloc[:, 3:10].values
	r2_scores =run_kfold(5, data, target)
	print (r2_scores)
