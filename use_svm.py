# SVM not sym
#class after the logistics machine
import logistics1
# from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs #a fake dataset
from sklearn import svm #classification formula which could replace the logisitics machine. SVM gives prediction of target instead of the probability of being target as logistics does

data, target=make_blobs(n_samples=400, centers=2, cluster_std=1, random_state=0) #center is the number of levels now we have 0, and 1; 
# target[target ==0] =-1

# print (data, target)
# plt.scatter(data[:,0], data[:,1], c=target)
# plt.savefig("plot.png")

r2_scores, accuracy_scores, confusion_matrices= logistics1.run_kfold(5, data, target, svm.SVC(kernel="linear"), 1,1)


print(r2_scores) #r2_scores is for evaluationg how good is the machine (blue line in the middle)
print (accuracy_scores)
for i in confusion_matrices:
	print(i)

# SVM An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate
#categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space 
#and predicted to belong to a category based on the side of the gap on which they fall.

