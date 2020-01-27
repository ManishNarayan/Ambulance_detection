from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import csv
import pickle
from sklearn import preprocessing


def convert(X,length):
    """
    A function to convert the given data [which has 128 dimensional embeddings] to a data which represents each video in one row.
    We are trying to combine the values of every ten rows into a list for each column.

    """
    X.values.T.tolist()
    x = [list(l) for l in zip(*X.values)]
    # Transform the dataframe into a list. First we transpose it and combine them into list.
    #Here x has 128 columns with the data from all the videos. This x has to be divided for each video [10 seconds/rows]. 
    
    xt = []
    for vid in range(length):
        temp = []
        for i in x :
            z = "".join(str(num) for num in i[:10])
            
            temp.append(int(z))
            del i[:10]
        #print("\nTEMP :  ",temp)
        xt.append(temp)   
    return xt
    



def SVM_train():
    X_train = pd.read_csv('amb_new_train.csv') # Read the csv file which has the required embeddings from the vggish model
    #[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Y_train = np.array(([1]*164) + [0]*140)
    # Y_train is the labels for each video [we have 161 videos of 10 secs each.] 
    
    # we are trying to convert the data into the required form by calling the predefined function.
    X_train = convert(X_train,304)       
    
    X_scaled = preprocessing.scale(X_train)#Scale the Data 
    df = pd.DataFrame(X_scaled)
    X_train = df.apply(preprocessing.LabelEncoder().fit_transform)#Encode.
       
    
    x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size = 0.2)       

    
    model = SVC(kernel = 'linear')#Define the SVM model. linear - since we need either 0 or 1.
    
    model.fit(x_train,y_train)# Fit the data onto the model.

    pred = model.predict(x_test)#perform the prediction and print it.
    print("Predictions :",pred)
       
    score = accuracy_score(y_test,pred)#Get the Accuracy score and print it.
    print("Accuracy = ", score)
    
    
    filename = 'final_svm_model.sav'
    pickle.dump(model,open(filename,'wb'))

SVM_train()

def SVM_test(X_test,Y_test) :
    filename = 'final_svm_model.sav'
    loaded_model = pickle.load(open(filename,'rb'))
    result = loaded_model.score(X_test,Y_test)
    print("Accuracy for the Test Dataset :",result)

x_test = pd.read_csv('amb_try_test.csv')

y_test = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x_test = convert(x_test,53)

SVM_test(x_test,y_test)