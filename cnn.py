from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split




arr=np.zeros((3200,60),np.float64)
output=np.zeros((3200,1))
#output=[]
for i in range(8):
    user ="U0"+ str(i+1)
    for j in range(20):
        gesture = str('{:02d}'.format(j+1))
        for k in range(20):
            iteration = str('{:02d}'.format(k+1))
            for x in range(10):
                arr[(i+1)*(j+1)*k,x:(x+6)]=np.loadtxt(fname="gestures-dataset"+"\\"+user+"\\"+gesture+"\\"+iteration+".txt", max_rows=1, skiprows=x, dtype=float)
            output[(i+1)*(j+1)*k]=j+1
            #output.append(j+1)
X_train, X_test, y_train, y_test = train_test_split(
    arr, output, test_size=0.3,random_state=109)


svc = svm.SVC(gamma=0.001)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)


print("Classification report for - \n{}:\n{}\n".format(
    svc, metrics.classification_report(y_test, y_pred)))