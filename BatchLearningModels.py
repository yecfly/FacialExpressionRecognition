# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-18 10:09:25
# @Last Modified by:   yys
# @Last Modified time: 2019-09-20 

import time
import logging
import numpy as np

from sklearn import linear_model
from sklearn import svm 
from sklearn import neighbors
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import metrics

def overAllAccuracy(conf_m, afc=None, cts=None):#modify in 20190524 adding cts record the counts
    accuracy_for_every_category=[]
    counts_for_every_category=[]
    CM=conf_m
    if type(conf_m) is np.ndarray:
        conf_m=conf_m.astype(np.float)
        r,c=conf_m.shape
        counts=np.sum(conf_m, axis=1)
        for i in range(r):
            for j in range(c):
                conf_m[i][j]=conf_m[i][j]/counts[i]
    elif type(conf_m) is list:
        r=len(conf_m)
        if r>0:
            c=len(conf_m[0])
        else:
            raise RuntimeError('ERROR: Confusion Matrix is unexpected.')
    else:
        raise RuntimeError('ERROR: Confusion Matrix is unexpected.')
    assert r==c, ('ERROR: Confusion Matrix is unexpected for its unequal rows and cols: %d %d'%(r,c))
    ac=0.0
    for i in range(r):
        ac=ac+conf_m[i][i]
        accuracy_for_every_category.append(conf_m[i][i])
        counts_for_every_category.append(CM[i][i])
    ac=ac/r
    if not afc is None:
        afc.clear()
        afc=afc.extend(accuracy_for_every_category)
    if not cts is None:
        cts.clear()
        cts=cts.extend(counts_for_every_category)
    return ac
def logfileBLM(file_record, fold, OAA=None, TA=None, afc=None, TC=None,
                 input=None, CM=None, T=None):
    if type(CM) is np.ndarray:
        tem=list(CM)
        CMS='[ '
        for v in range(len(tem)-1):
            CMS=CMS+str(tem[v]).replace(' ', ', ')+', '
        CMS=CMS+str(tem[-1]).replace(' ', ', ')+']'
        b=CMS.find(', , ')
        while b>-1:
            CMS=CMS.replace(', , ', ', ')
            b=CMS.rfind(', , ')
        CMS=CMS.replace(', ]', ']')
        CMS=CMS.replace('[, ', '[')
    elif type(CM) is list:
        CMS=str(CM)
    file_record='Fold%d\tOverAllACC: %s\tTA: %s\tACs: %s\tTimeComsumed:%s\tInput:%s\t%s\tTime:%s\t'%(fold, 
                                                    str(OAA), str(TA), str(afc), str(TC), str(input),str(CMS),time.strftime('%Y%m%d%H%M%S',T))
    return file_record

class BatchLearning:
    def __init__(self, fdata):
        self.X=fdata.train.X
        self.Y=np.argmax(fdata.train.Y, axis=1)
        self._T_X=fdata.test.X
        self._T_Y=np.argmax(fdata.test.Y, axis=1)
        print('length of feature {0}'.format(len(self.X[0])))

    def train(self, generate_model = None, logfile = None, fold=-1, argv=None):
        if logfile == None:
            print('specify the path of logfile firstly')
            return
        logger = logging.getLogger(logfile.split()[0])
        logger.setLevel(logging.DEBUG)
        file_handle = logging.FileHandler(logfile)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s  %(message)s"))
        logger.addHandler(file_handle)
        
        #logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

        if generate_model == None:
            print('specify the model firstly')
            return

        start_time = time.time()
        y_predict = []

        model = generate_model()
        train_X, train_Y= [], []
        test_X, test_Y = self._T_X[:], self._T_Y.copy()
        #for j in range(len(self.Y)):
            #train_X.extend(self.X[j])
            #train_Y.extend(self.Y[j])
        train_X.extend(self.X)
        train_Y.extend(self.Y)

        model.fit(train_X, train_Y)
        y_predict = model.predict(test_X)
        cross_val_score = metrics.accuracy_score(y_predict, test_Y)
            
        tpre = model.predict(train_X)
        TA = metrics.accuracy_score(tpre, train_Y)

        logger.info('training model description \n {0}'.format(model))
        logger.info('length of feature {0}'.format(len(self.X[0])))
        logger.info('time consuming: {0}'.format(time.time() - start_time))
        print('time consuming: %fs'%(time.time() - start_time))
        logger.info('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        print('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        logger.info('Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(test_Y, y_predict)))
        logger.info('Train Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(train_Y, tpre)))
        
        file_record=None
        cm=metrics.confusion_matrix(test_Y, y_predict)
        afc=[]
        cts=[]
        oaa=overAllAccuracy(cm, afc, cts)
        tt=time.time()
        timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(tt))
        file_record = logfileBLM(file_record, fold=fold, OAA=oaa, TA=cross_val_score, afc=afc,
                                TC=(tt-start_time), input=argv, CM=cm, T=time.localtime(tt))
        filelog2=logfile.replace('.txt','_Results.txt')
        filelog=open(filelog2,'a')
        filelog.write('%s\t\t Timestamp: %s\tClassifier: %s\t%s\n'%(file_record, timestamp, str(type(model).__name__), logfile))
        filelog.close()

    def trainWithFeatureSelectionFirst(self, generate_model = None, logfile = None, fold=-1, argv=None):
        if logfile == None:
            print('specify the path of logfile firstly')
            return
        logger = logging.getLogger(logfile.split()[0])
        logger.setLevel(logging.DEBUG)
        file_handle = logging.FileHandler(logfile)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s  %(message)s"))
        logger.addHandler(file_handle)
        
        if generate_model == None:
            print('specify the model firstly')
            return

        
        start_time = time.time()
        y_predict = []
        
        model = generate_model()
        train_X, train_Y= [], []
        test_X, test_Y = self._T_X[:], self._T_Y.copy()

        #for j in range(len(self.Y)):
            #train_X.extend(self.X[j])
            #train_Y.extend(self.Y[j])
        train_X.extend(self.X)
        train_Y.extend(self.Y)
        
        lsvc = svm.LinearSVC(C=0.5,penalty='l1',dual=False).fit(train_X, train_Y)
        sm=SelectFromModel(lsvc, prefit=True)
        train_NX=sm.transform(train_X)
        test_NX=sm.transform(test_X)

        model.fit(train_NX, train_Y)
        y_predict = model.predict(test_NX)
        cross_val_score = metrics.accuracy_score(y_predict, test_Y)

        tpre = model.predict(train_NX)
        TA = metrics.accuracy_score(tpre, train_Y)

        logger.info('training model description \n {0}'.format(model))
        logger.info('length of feature {0}'.format(len(self.X[0])))
        logger.info('time consuming: {0}'.format(time.time() - start_time))
        print('time consuming: %fs'%(time.time() - start_time))
        logger.info('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        print('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        logger.info('Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(test_Y, y_predict)))
        logger.info('Train Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(train_Y, tpre)))

        
        file_record=None
        cm=metrics.confusion_matrix(test_Y, y_predict)
        afc=[]
        cts=[]
        oaa=overAllAccuracy(cm, afc, cts)
        tt=time.time()
        timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(tt))
        file_record = logfileBLM(file_record, fold=fold, OAA=oaa, TA=cross_val_score, afc=afc,
                                TC=(tt-start_time), input=argv, CM=cm, T=time.localtime(tt))
        filelog2=logfile.replace('.txt','_Results.txt')
        filelog=open(filelog2,'a')
        filelog.write('%s\t\t Timestamp: %s\tClassifier: %s\t%s\n'%(file_record, timestamp, str(type(model).__name__), logfile))
        filelog.close()
    def libsvmtrain(self, kernel='linear', degree=1, logfile = None, fold=-1, argv=None):
        #added by yys
        if logfile == None:
            print('specify the path of logfile firstly')
            return
        logger = logging.getLogger(logfile.split()[0])
        logger.setLevel(logging.DEBUG)
        file_handle = logging.FileHandler(logfile)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s  %(message)s"))
        logger.addHandler(file_handle)
        
        start_time = time.time()
        y_predict = []
        model=svm.SVC(kernel=kernel, degree=degree)
        train_X, train_Y= [], []
        test_X, test_Y = self._T_X[:], self._T_Y.copy()
        #for j in range(len(self.Y)):
            #train_X.extend(self.X[j])
            #train_Y.extend(self.Y[j])
        train_X.extend(self.X)
        train_Y.extend(self.Y)
        
        b=model.fit(train_X, train_Y)
        y_predict = model.predict(test_X)
        cross_val_score = metrics.accuracy_score(y_predict, test_Y)

        tpre = model.predict(train_X)
        TA = metrics.accuracy_score(tpre, train_Y)

        logger.info('training model description \n {0}'.format(model))
        logger.info('length of feature {0}'.format(len(self.X[0])))
        logger.info('time consuming: {0}'.format(time.time() - start_time))
        print('time consuming: %fs'%(time.time() - start_time))
        logger.info('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        print('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        logger.info('Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(test_Y, y_predict)))
        logger.info('Train Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(train_Y, tpre)))

        
        file_record=None
        cm=metrics.confusion_matrix(test_Y, y_predict)
        afc=[]
        cts=[]
        oaa=overAllAccuracy(cm, afc, cts)
        tt=time.time()
        timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(tt))
        file_record = logfileBLM(file_record, fold=fold, OAA=oaa, TA=cross_val_score, afc=afc,
                                TC=(tt-start_time), input=argv, CM=cm, T=time.localtime(tt))
        filelog2=logfile.replace('.txt','_Results.txt')
        filelog=open(filelog2,'a')
        filelog.write('%s\t\t Timestamp: %s\tClassifier: %s\t%s\n'%(file_record, timestamp, str(type(model).__name__), logfile))
        filelog.close()
    def libsvmtrainFeatureSelectionFirst(self, kernel='linear', degree=1, logfile = None, fold=-1, argv=None):
        #added by yys
        if logfile == None:
            print('specify the path of logfile firstly')
            return
        logger = logging.getLogger(logfile.split()[0])
        logger.setLevel(logging.DEBUG)
        file_handle = logging.FileHandler(logfile)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s  %(message)s"))
        logger.addHandler(file_handle)
        
        start_time = time.time()
        y_predict = []
        
        model=svm.SVC(kernel=kernel, degree=degree)
        train_X, train_Y= [], []
        test_X, test_Y = self._T_X[:], self._T_Y.copy()
        #for j in range(len(self.Y)):
            #train_X.extend(self.X[j])
            #train_Y.extend(self.Y[j])
        train_X.extend(self.X)
        train_Y.extend(self.Y)
        
        lsvc = svm.LinearSVC(C=0.5, penalty='l1',dual=False).fit(train_X, train_Y)
        sm=SelectFromModel(lsvc, prefit=True)
        train_NX=sm.transform(train_X)
        test_NX=sm.transform(test_X)

        model.fit(train_NX, train_Y)
        y_predict = model.predict(test_NX)
        cross_val_score = metrics.accuracy_score(y_predict, test_Y)
            
        tpre = model.predict(train_NX)
        TA = metrics.accuracy_score(tpre, train_Y)

        logger.info('training model description \n {0}'.format(model))
        logger.info('length of feature {0}'.format(len(self.X[0])))
        logger.info('time consuming: {0}'.format(time.time() - start_time))
        print('time consuming: %fs'%(time.time() - start_time))
        logger.info('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        print('Accuracy: {0}\tTrain Accuracy: {1}'.format(cross_val_score, TA))
        logger.info('Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(test_Y, y_predict)))
        logger.info('Train Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(train_Y, tpre)))
        
        file_record=None
        cm=metrics.confusion_matrix(test_Y, y_predict)
        afc=[]
        cts=[]
        oaa=overAllAccuracy(cm, afc, cts)
        tt=time.time()
        timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(tt))
        file_record = logfileBLM(file_record, fold=fold, OAA=oaa, TA=cross_val_score, afc=afc,
                                TC=(tt-start_time), input=argv, CM=cm, T=time.localtime(tt))
        filelog2=logfile.replace('.txt','_Results.txt')
        filelog=open(filelog2,'a')
        filelog.write('%s\t\t Timestamp: %s\tClassifier: %s\t%s\n'%(file_record, timestamp, str(type(model).__name__), logfile))
        filelog.close()
        
def logistic_regression():
    return linear_model.LogisticRegression(penalty = 'l1', n_jobs = -1)   


def svm_model():
    # return svm.SVC()
    return svm.LinearSVC()

def knn_model():
    return neighbors.KNeighborsClassifier(n_neighbors = 10)


def decision_tree():
    return tree.DecisionTreeClassifier()

def bagging_classifier():
    #base_model = linear_model.LogisticRegression(penalty = 'l1', n_jobs = -1)
    base_model = svm.LinearSVC()
    return BaggingClassifier(base_model, max_samples=0.5, max_features=0.5)


def boosting_classfier():
    base_model = linear_model.LogisticRegression(penalty = 'l1', n_jobs = -1)
    return AdaBoostClassifier(base_estimator = base_model)

def mainCall(log, fdata):#####deprecated and unfinished codes
    
    batch_model = BatchLearning(fdata=fdata)
    ## logistic regression
    print('\nLogisticRegression')
    logfile = '%s_LogisticRegression.txt'%(log)
    batch_model.train(generate_model = logistic_regression, logfile = logfile)
    tt=time.time()
    print('Time consumed: %fs'%(tt-t1))
    
    ## logistic regression
    print('\nSELECT_LogisticRegression')
    logfile = './logs/{0}_SelectFeatureFirst_LogisticRegression.log'.format(data_file.split('/')[-1].split('.')[0])
    batch_model.trainWithFeatureSelectionFirst(generate_model = logistic_regression, logfile = logfile)
    tt=time.time()
    print('Time consumed: %fs'%(tt-t1))

    #svm
    print('\nSVM rbf kernel')
    logfile = './logs/{0}_libsvm.log'.format(data_file.split('/')[-1].split('.')[0])
    batch_model.libsvmtrain(kernel='rbf', logfile = logfile)
    tt=time.time()
    print('Time consumed: %fs'%(tt-t1))

    #svm
    print('\nSELECT_SVM rbf kernel')
    logfile = './logs/{0}_SelectFeatureFirst_libsvm.log'.format(data_file.split('/')[-1].split('.')[0])
    batch_model.libsvmtrainFeatureSelectionFirst(kernel='rbf', logfile = logfile)
    tt=time.time()
    print('Time consumed: %fs'%(tt-t1))

    ##svm
    #logfile = './logs/{0}_svm.log'.format(data_file.split('/')[-1].split('.')[0])
    #batch_model.train(generate_model = svm_model, logfile = logfile)
   
    # knn
    #logfile = 'D20_knn.log'
    #batch_model.train(generate_model = knn_model, logfile = logfile)
    
    # decision tree
    #logfile = 'D20_dt.log'
    #batch_model.train(generate_model = decision_tree, logfile = logfile)

    # bagging
    #logfile = 'D20_bagging.log'
    #batch_model.train(generate_model = bagging_classifier, logfile = logfile)
    
    # boosting
    #logfile = 'D10_boosting.log'
    #batch_model.train(generate_model = boosting_classfier, logfile = logfile)
    
    # neural network
    #logfile = 'D20_neuralnetwork.log'
    #batch_model.train(generate_model = neural_network, logfile = logfile)
    
    t2=time.time()
    print('\n\nTime consumed: %fs'%(t2-t1))