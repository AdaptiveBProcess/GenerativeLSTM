# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 04:43:48 2020

@author: Mauricio Díaz
"""

from tensorflow import keras
import numpy as np
import random as rd
import os
import csv

#check if file exist if not create it
def check_file(file_name):
    if not os.path.exists(file_name):
        csv_file = open(file_name,'w+')
        data = ['Folder','Model_File','Batch_size','Epochs','optim','Event_Log',
                       'Next_Event_Prediction_Accuracy','TP','TN','FP','FN',
                       'Amount_of_Anomalies','Accuracy', 'Precision', 'Recall', 'F1']
        csv_Writer = csv.writer(csv_file, delimiter=',')
        csv_Writer.writerow(data)
        csv_file.close()

def random(a):
    rad = rd.randint(1,100)
    if rad <= a:
        return True
    return False

def randomNumb(a):
    rad = rd.randint(1,len(a))
    return a[rad]
    

#Returns the Global Threshold  and  dictionary with training data proportions of dataset
def useful(a, b):
    propAct = {}
    for i in range(b.shape[1]):
        if i != 0 and i != (b.shape[1]-1):
            propAct[i] = 0
    arr = []
    cont = 0
    total = 0
    while cont < len(b):
        #Dictionary
        act = a[cont][4]
        if act != 0 and act != b.shape[1]-1:
            propAct[act] += 1
            total += 1
            
        #Threshold 
        if (cont == 0 
            or b[cont-1].argmax() == a[cont][4] 
            or (b[cont-1].argmax() == b.shape[1]-1 and a[cont][4] == 0)):
            cont+=1
            continue
        #Wrong predict
        arr.append(b[cont-1][a[cont][4]])
        cont+=1
    #Activities Dictionary
    for act in propAct:
        propAct[act] = propAct[act]/total
        
    #Threshold     
    threshold  = np.mean(arr)- np.std(arr)
    
    return propAct, threshold 

def createDict(a, b, diction):
    for i in range(a.shape[0]):
        if i != 0:
            key = str(a[i])
            if diction.get(key) != None:
                if (b[i-1].argmax() == a[i][4]
                    or (b[i-1].argmax() == b.shape[1]-1 and a[i][4] == 0)):
                    diction[key][0].append(b[i-1][a[i][4]])
                else:
                    diction[key][1].append(b[i-1][a[i][4]])
            else:
                if (b[i-1].argmax() == a[i][4]
                    or (b[i-1].argmax() == b.shape[1]-1 and a[i][4] == 0)):
                    diction[key] = [[b[i-1][a[i][4]]],[]]
                else:
                    diction[key] = [[],[b[i-1][a[i][4]]]]
    
    
    return diction

def specThreshold(xT, xPT, xV, xVT):
    diction = {}
    diction = createDict(xT, xPT, diction)
    diction = createDict(xV, xVT, diction)
    
    for i in diction:
        wrong = diction[i][1]
        specThreshold = 0
        if  wrong:
            specThreshold = np.mean(wrong) - np.std(wrong)
            
        diction[i] = specThreshold
            
    return diction

def anomalyGen(arr, anomAct):
    print("***************** CREATING ANOMALIES *****************")
    truth = []
    flag = [False, False]
    anom = None
    pos = 5
    cont = 0
    anoms = 0
    
    while cont < len(arr):
        act = arr[cont]
        
        if flag[0]:
            if flag[1] and act[0] != 0:
                flag[0] = False
                flag[1] = False
                pos = 0
                truth.append(1)
                cont+=1
                continue
            
            elif not flag[1] and act[0] != 0:
                truth.append(1)
                pos = 5
                cont+=1
                continue
                
            elif not flag[1] and not np.all(act==0):
                arr[cont][4] = anom
                pos = 4
                truth.append(0)
                anoms+=1
                flag[1] = True
                cont+=1
                continue
            elif flag[1] and not np.all(act==0):
                arr[cont][pos-1]=anom
                pos -= 1
                truth.append(1)
                cont+=1
                continue
            elif flag[1] and np.all(act==0):
                flag[0] = False
                flag[1] = False
                pos = 0
                truth.append(1)
                cont+=1
                continue
                
                
        elif random(30):
            flag[0] = True
            anom = randomNumb(anomAct)
            if not np.all(act==0):
                arr[cont][4] = anom
                anoms+=1
                pos = 4
                truth.append(0)
                flag[1] = True
                cont+=1
                continue
            
        truth.append(1)
        pos = 5
        cont+=1
        
    return arr, truth, anoms
                
def detecAnoms(x_test, x_pred, truth, globalThreshold, dicT=None):
    print("***************** DETECTING ANOMALIES *****************")
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    print("Global", globalThreshold)
    if  dicT == None:
        cont = 0
        while cont < len(x_pred):
            act = x_test[cont][4]
            if x_test[cont][4] == 0:
                act = x_pred.shape[1]-1
            if x_pred[cont-1][act] < globalThreshold:
                if truth[cont] == 0:
                    TP += 1
                else:
                    FP += 1
            elif x_pred[cont-1][act] >= globalThreshold:
                if truth[cont] == 1:
                    TN += 1
                else:
                    FN += 1
            cont+=1
         
    else:
        cont = 0
        while cont < len(x_pred):
            act = x_test[cont][4]
            key = str(x_test[cont])
            if x_test[cont][4] == 0:
                act = x_pred.shape[1]-1
                
            if dicT.get(key) != None:
                specThreshold = dicT[key]
                
                if x_pred[cont-1][act] < specThreshold:
                    if truth[cont] == 0:
                        TP += 1
                    else:
                        FP += 1
                elif x_pred[cont-1][act] >= specThreshold:
                    if truth[cont] == 1:
                        TN += 1
                    else:
                        FN += 1                    
                
            else:
                if x_pred[cont-1][act] < globalThreshold:
                    if truth[cont] == 0:
                        TP += 1
                    else:
                        FP += 1
                elif x_pred[cont-1][act] >= globalThreshold:
                    if truth[cont] == 1:
                        TN += 1
                    else:
                        FN += 1
            
            
            #time.sleep(5)
            cont+=1
    
        
    
    return TP,TN,FP,FN
                
                


def _detector(args):
    
    pTr = 0.6
    pTe = 0.2 
    model = None
    
    output_folder = os.path.join('output_files', args['folder'])
    
    try:
        model = keras.models.load_model(output_folder+"/transf_model")
        print("Model loaded correctly")
    except:
        print("Model not found")
        return
    
    x_train = np.load(output_folder+'/parameters/x_train.npy')
    x_val = np.load(output_folder+'/parameters/x_val.npy')
    x_test = np.load(output_folder+'/parameters/x_test.npy')
    y_test = np.load(output_folder+'/parameters/y_test.npy')
    
    
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=args['batch_size'])
    print("test loss, test acc:", results)         
    
    
    x_predT = model.predict(x_train)
    x_predV = model.predict(x_val)
    
    
    prop_train, threshold_train = useful(x_train, x_predT)
    prop_val, threshold_val = useful(x_val, x_predV)
    
    #Weighted average of the two dictionaries of the activities proportions
    for act in prop_train:
        prop_train[act] = (prop_train[act]*pTr + prop_val[act]*(1-pTr-pTe))/(1-pTe)
    
    propAct = prop_train    
    
    #Weighted average of global Thresholds
    globalThreshold = ((threshold_train*pTr) + (threshold_val*(1-pTr-pTe)))/(1-pTe)
    
    print("Threshold", globalThreshold)
    print("Prop Act array", propAct)
    
    diction = specThreshold(x_train, x_predT, x_val ,x_predV)
    
    #Anoms *********************
    anomAct = {}
    cont = 1
    for act in propAct:
        if propAct[act] < globalThreshold:
            anomAct[cont] = act
            cont+=1
    
    x_test, truth, anoms = anomalyGen(x_test, anomAct)
    
    print("Size x_test", len(x_test))
    
    x_pred = model.predict(x_test)
    
    TP,TN,FP,FN = detecAnoms(x_test, x_pred, truth, globalThreshold, diction)
    
    acc = (TP+TN)/len(x_pred)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*((precision*recall)/(precision+recall))
    
    datafile = open(output_folder+'/parameters/data.txt')
    data = next(csv.reader(datafile))
    data1 = [results[1],TP,TN,FP,FN,anoms,acc,precision,recall,f1]
    data.extend(data1)
        
    csv_file = 'output_files/anomaly_detection.csv'
    check_file(csv_file)
    csv_file = open(csv_file, 'a+')
    csv_Writer = csv.writer(csv_file, delimiter=',')
    csv_Writer.writerow(data)
    
    datafile.close()
    csv_file.close()
    
    print("Amount of anomalies", anoms)
    print("TP",TP, "TN",TN, "FP",FP, "FN",FN, "total", TP+TN+FP+FN)
    print("Correct:", TP+TN, "Wrong:", FP+FN, "Accuracy:", acc)
    print("Precision:",precision)
    print("Recall:", recall)
    print("F1 Score:",f1)