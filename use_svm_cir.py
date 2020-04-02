import logistics1
# from matplotlib import pyplot as plt
from sklearn.datasets import make_circles #a fake dataset
from sklearn import svm

data, target =make_circles(n_samples=500, noise=0.03)
# if circle, the old svm is a line cannot handle it, so change the linear to rbf kernel could help us to classify the circle.

# print (data, target)
# plt.scatter(data[:,0], data[:,1], c=target)
# plt.savefig("plot.png")

# r2_scores, accuracy_scores, confusion_matrices= logistics1.run_kfold(5, data, target, svm.SVC(kernel="rbf", gamma=5), 1,1)


# print(r2_scores) #r2_scores is for evaluationg how good is the machine (blue line in the middle)
# print (accuracy_scores)
# for i in confusion_matrices:
# 	print(i)

#_________________________________________________
#if still using linear kernel, we could define the circle
#x^2+y^2=r^2
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
data1 =data[:,0].reshape((-1,1))
data2 =data[:,1].reshape((-1,1))
# here if we do not know the data structure, we could use a loop here for different patterns such as cube.
# when the accuracy score is similar, then consider the number of features (less is better), and then try to increase
# the kfold number
data3 =(data1**2+data2**2)

data =np.hstack ((data,data3))
# print(data)

#3D plots
# fig =plt.figure()
# axes=fig.add_subplot(111, projection="3d")
# axes.scatter(data1, data2, data3, c=target, depthshade=True)
# plt.savefig('plot3d.png')


# r2_scores, accuracy_scores, confusion_matrices= logistics1.run_kfold(5, data, target, svm.SVC(kernel="linear"), 1,1)

# print(r2_scores) #r2_scores is for evaluationg how good is the machine (blue line in the middle)
# print (accuracy_scores)
# for i in confusion_matrices:
# 	print(i)
# # scores are very high

#now the machine is a plain instead of a line



# extra "hyperplane" but this is not for judgeing if the model is good or not
machine=svm.SVC(kernel="linear")
machine.fit(data, target)
coeff =machine.coef_
intercept =machine.intercept_

plane =coeff[0][0]*data1+coeff[0][1]*data2 +intercept
plot_surface()



