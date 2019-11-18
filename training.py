####
# Platform windows 10 with Tensorflow 1.13.1, CUDA 10.0.130, python 3.7.1 64 bit, MSC v.1915 64bit,
###################
import collections
import os
import pickle
import sys
import time
import traceback
import warnings

import keras.backend as K
import numpy as np
import tensorflow as tf
import win_unicode_console
from keras import optimizers as OPT
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix)
from tensorflow.python.framework import dtypes

import FERNetworks as FERN
from NetworkClass import FERFN
from SelectedPresavedFusionData import getSelectedDataFile

win_unicode_console.enable()

lr_drate=0.8
times=40 #which control the decay learning rate decays at every %times% epochs
#ovt_drate=0.85#0.3 for runs5 and runs6, 0.5 for runs 7, 1 for runs 4 runs8-run13. 0.8 for runs14 and runs16, 1.0 for runs15
test_bat=200
TestNumLimit = 200
Mini_Epochs = 140#300; 140 for M1-M4, 240 for ovt
Acc_Oriented=True#wether the final model is based on highest Accuracy or lowest loss
Summary=False####unfinished for KerasV
EG1=[0, 2, 5, 6]
EG2=[1,3,4]
SaveModel=False
ForVisFea=False
scale_factor=1.0/255.0
M23CPL=False
smFOR1=True
RADAM=True
LOOKAHEAD=False
BatchLearning=False#be carefual, it only interacts with M23 Fusion option now
#FUSION=True#for module 11 with NewC as 2 or 3


####
def getOVTSM(Module, sm=False):
    if Module==1 or Module==3 or Module==4 or Module==23 or Module==211 or Module==411:
        sm=False
    if ForVisFea:
        sm=True
    return sm
##for training
def getOVTDR(Module):
    ovt_dr=1
    if Module==11:
        ovt_dr=0.85
    elif Module==211:
        ovt_dr=0.85
    elif Module==2:
        ovt_dr=0.85
    elif Module==3 or Module==23:
        ovt_dr=0.8
    elif Module==1 or Module==4:
        ovt_dr=0.8
    elif Module==411:
        ovt_dr=0.8
    return ovt_dr
def getOVTME(Module, me=60, DataSet=None):
    if Module==1 or Module==3 or Module==4 or Module==23:
        me=140
        if DataSet==690 or DataSet==4090 or DataSet==4190:
            me=100
    elif Module==2 or Module==11 or Module==211:
        me=140
    elif Module==411:
        if DataSet is not None:
            if DataSet%10<7 and DataSet%10>0:
                me=100
            elif DataSet%10==9:
                me=100
            elif DataSet==690 or DataSet==4090 or DataSet==4190:
                me=60
            else:
                me=120
        else:
            me=120#for SFEW2
    print('Total Epoch %d'%me)
    return me
def getOVTTimes(Module, t=40, DataSet=None):
    if Module==1 or Module==3 or Module==4 or Module==23:
        t=40
    elif Module==11 or Module==2 or Module==211 or Module==411:
        t=40
    if DataSet is not None:
        if  DataSet==690 or DataSet==4090 or DataSet==4190:
            t=30
    print('Decayed Epoch %d'%t)
    return t
def getOVTBS(Module, bs=30, DataSet=None):
    if Module==1 or Module==3 or Module==4:
        bs=30
        if DataSet is not None:
            if DataSet%10==9:
                bs=720###480, 640, 
            elif DataSet==690 or DataSet==4090 or DataSet==4190:
                bs=640
    elif Module==23:
        bs=30
    elif Module==11 or Module==211:
        bs=32
        if DataSet==690 or DataSet==4090 or DataSet==4190:
            bs=640
        elif DataSet%10==7:
            bs=32
    elif Module==411:
        bs=32
        if DataSet is not None:
            if DataSet%10==9:
                bs=720###480, 640, 
            elif DataSet==690 or DataSet==4090 or DataSet==4190:####for M411 modelfit 32 is actual used before 20191007
                #bs=640###bad performance with 63% top
                bs=32
    print('BatchSize: %d'%bs)
    return bs
def getsaveMV(Module, DataSet):
    sm=0.85
    if Module==3:
        if DataSet%10<4 and DataSet%10>0:
            sm=0.97
        elif DataSet%10==4:
            sm=0.93
    elif Module==1 or Module==4:
        if DataSet%10<4 and DataSet%10>0:
            sm=0.90
        elif DataSet%10==4:
            sm=0.85
    if smFOR1:
        sm=1
    if ForVisFea:
        sm=0.85
    return sm


def initialize_dirs(ovt=False):
    if ovt:
        if not os.path.exists('./logs/ovt/VL'):
            os.makedirs('./logs/ovt/VL')
        if not os.path.exists('./saves/ovt'):
            os.makedirs('./saves/ovt')
        if not os.path.exists('./logs/KerasV/VL'):
            os.makedirs('./logs/KerasV/VL')
        if not os.path.exists('./saves/KerasV'):
            os.makedirs('./saves/KerasV')
        if not os.path.exists('./modelfigures'):
            os.makedirs('./modelfigures')
        if not os.path.exists('./MMFeatures'):
            os.makedirs('./MMFeatures')
    else:
        if not os.path.exists('./logs/VL'):
            os.makedirs('./logs/VL')
        if not os.path.exists('./saves'):
            os.makedirs('./saves')
    if BatchLearning:
        if not os.path.exists('./logs/BLM'):
            os.makedirs('./logs/BLM')
    return True
class LOSS_ANA:
    '''The LOSS_ANA class collects the training losses and analyzes them.
        The initial length should be divided by 50 with no remainder.'''
    def __init__(self, step=10):
        self.__Validation_Loss_List = []
        self.__Current_Length = 0#indicates whether the Validation_Loss_List has reach the maximum Length
        self.__Min_Loss = 10000.0
        self.__accuracy = 0.0
        self.__iteration = -1
        self.__epoch = -1
        self.__Acc_List=[]

    @property
    def highestAcc(self):
        return self.__accuracy
    @property
    def minimun_loss(self):
        return self.__Min_Loss
    @property
    def loss_length(self):
        return self.__Current_Length
    @property
    def bestIteration(self):
        return self.__iteration
    @property
    def bestEpoch(self):
        return self.__epoch
    def preSetOutputList(self, lossl, accl, lr):
        '''Preset the output contents with loss, acc, and lr.'''
        if len(lossl)==len(accl) and len(lossl)==len(lr):
            self.out=[]
            self.__Current_Length=-10
            for i in range(len(lossl)):
                self.out.append('%.16f\t%.16f\t%.16f\n'%(lossl[i], accl[i], lr[i]))
        else:
            print('Inconsistent length %d %d. Exit without setting.'%(len(lossl), len(accl)))
            return

    def analyzeLossVariation(self, loss, iter, epo, va):
        '''
        Inputs:
            loss: float type, the current loss of the validation set
            
        Outputs:
            boolean type: indicates whether the input is less than all others before it
        '''
        self.__Current_Length = self.__Current_Length + 1
        flag=False
        if loss < self.__Min_Loss:
            self.__Min_Loss = loss
            if not Acc_Oriented:
                flag=True
                self.__iteration=iter
                self.__epoch=epo
            else:
                if va==self.__accuracy:
                    flag=True
        if self.__accuracy<va:
            self.__accuracy=va
            if Acc_Oriented:
                flag=True
                self.__iteration=iter
                self.__epoch=epo

        self.__Validation_Loss_List.append(loss)
        return flag

    def outputlosslist(self, logfilename):
        '''input the file name to log out all the validation losses in the current training'''
        if self.__Current_Length==0:
            return
        else:
            fw=open(logfilename,'w')
            if self.__Current_Length==-10:
                for v in self.out:
                    fw.write(v)
            else:
                for v in self.__Validation_Loss_List:
                    fw.write('%.16f\n'%(v))
            fw.close()
def calR(predict_labels_in, groundtruth_labels_in, cn=7):
    #print(len(predict_labels_in.shape))
    #print(len(predict_labels_in))
    #print(len(np.asarray(groundtruth_labels_in).shape))
    #print(len(groundtruth_labels_in))
    #exit()
    if len(np.asarray(predict_labels_in).shape)==1:
        predict_labels=DataSetPrepare.dense_to_one_hot(predict_labels_in, cn)
        #print(predict_labels.shape)
    else:
        predict_labels=predict_labels_in
    if len(np.asarray(groundtruth_labels_in).shape)==1:
        groundtruth_labels=DataSetPrepare.dense_to_one_hot(groundtruth_labels_in, cn)
        #print(groundtruth_labels.shape)
    else:
        groundtruth_labels=groundtruth_labels_in
    assert len(predict_labels)==len(groundtruth_labels), ('predict_labels length: %d groundtruth_labels length: %d' % (len(predict_labels), len(groundtruth_labels)))
    nc=len(groundtruth_labels)
    g_c=np.zeros([cn])
    #confusion_mat=[[0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0],
    #        [0,0,0,0,0,0,0]]
    confusion_mat=list(np.zeros([cn,cn]))
    for i in range(nc):
        cmi=list(groundtruth_labels[i]).index(max(groundtruth_labels[i]))
        g_c[cmi]=g_c[cmi]+1
        pri=list(predict_labels[i]).index(max(predict_labels[i]))
        confusion_mat[cmi][pri]=confusion_mat[cmi][pri]+1
    for i in range(len(g_c)):
        if g_c[i]>0:
            confusion_mat[i]=list(np.asarray(confusion_mat[i])/g_c[i])
    return confusion_mat
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
def Valid_on_TestSet(cn, sess, accuracy, sum_test, loss, softmax,
                       placeholder1, placeholder1_input, 
                       placeholder_labels, placeholder_labels_input,afc=None):
    '''Evalute the data with 1 network input in the session input
Inputs:
    sess:
    accuracy:
    sum_test:
    loss:
    softmax:

    
Outputs:
    v_accuracy:
    valid_loss:
    oaa:
    confu_mat'''
    ncount=len(placeholder_labels_input)
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[start:end],
                                                            placeholder_labels:placeholder_labels_input[start:end]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
            tlabels.extend(tlab)
        if ncount%test_bat>0:
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[test_bat*test_iter:ncount], 
                                        placeholder_labels:placeholder_labels_input[test_bat*test_iter:ncount]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
        tlabels.extend(tlab)
    else:
        v_accuracy, valid_loss, tlab = sess.run([accuracy, loss, softmax], feed_dict={placeholder1:placeholder1_input, 
                                                                       placeholder_labels:placeholder_labels_input})
        tlabels.extend(tlab)
    confu_mat=calR(tlabels, placeholder_labels_input, cn)
    oaa=overAllAccuracy(confu_mat,afc=afc)
    return v_accuracy, valid_loss, oaa, confu_mat
def Valid_on_TestSet_3NI(cn, sess, accuracy, sum_test, loss, softmax,
                       placeholder1, placeholder1_input, 
                       placeholder2, placeholder2_input,
                       placeholder3, placeholder3_input,
                       placeholder_labels, placeholder_labels_input, afc=None):
    '''Evalute the data with 3 network inputs in the session input
Inputs:
    sess:
    accuracy:
    sum_test:
    loss:
    softmax:

    
Outputs:
    v_accuracy:
    valid_loss:
    oaa:
    confu_mat'''
    ncount=len(placeholder_labels_input)
    tlabels=[]
    if ncount>TestNumLimit:               
        test_iter=np.floor_divide(ncount,test_bat)
        v_accuracy=0
        valid_loss=0
        for ite in range(test_iter):
            start=test_bat*ite
            end=test_bat*(ite+1)
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[start:end],
                                                                            placeholder2:placeholder2_input[start:end],
                                                                            placeholder3:placeholder3_input[start:end],
                                                                            placeholder_labels:placeholder_labels_input[start:end]})
            v_accuracy=v_accuracy+st
            valid_loss=valid_loss+v_loss
            tlabels.extend(tlab)
        if ncount%test_bat>0:
            st, v_loss, tlab=sess.run([sum_test, loss, softmax], feed_dict={placeholder1:placeholder1_input[test_bat*test_iter:ncount],
                                                                        placeholder2:placeholder2_input[test_bat*test_iter:ncount],
                                                                        placeholder3:placeholder3_input[test_bat*test_iter:ncount],
                                                                        placeholder_labels:placeholder_labels_input[test_bat*test_iter:ncount]})
            tlabels.extend(tlab)
        v_accuracy=v_accuracy+st
        valid_loss=valid_loss+v_loss
        v_accuracy=v_accuracy/ncount
        valid_loss=valid_loss/(test_iter+1)
    else:
        v_accuracy, valid_loss, tlab = sess.run([accuracy, loss, softmax], feed_dict={placeholder1:placeholder1_input, 
                                                                                         placeholder2:placeholder2_input,
                                                                                         placeholder3:placeholder3_input,
                                                                                         placeholder_labels:placeholder_labels_input})
        tlabels.extend(tlab)
    confu_mat=calR(tlabels, placeholder_labels_input, cn)
    oaa=overAllAccuracy(confu_mat, afc=afc)
    return v_accuracy, valid_loss, oaa, confu_mat
def logfile(file_record, runs, OAA, afc, valid_loss, valid_min_loss, final_train_loss, train_min_loss, 
            TA, TC, ILR, FLR, LS, ites, Epo, cBS, iBS, input, CM, T, df, lossa):
    file_record='Run%02d\tOverAllACC:%0.8f\tTestAccuracy:%.8f\tACs: %s\tFinalLoss:%.10f\tMinimunLoss:%.10f\tFinaltrainloss:%.10f\tMinimumtrainloss:%.10f\tTimeComsumed:%08.6f\tInitialLearningRate:%.8f\tFinalLearningRate:%.8f\tLearningStepForDroppingMagnitude:%08d\tTotalIterations:%08d\tEpoches:%08d\tcurrentBatchSize:%05d\tinitialBatchSize:%05d\tInput:%s\t%s\tTime:%s\tDataFile:%s\t%.10f'%(runs, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              OAA, TA, str(afc), valid_loss, valid_min_loss, final_train_loss, train_min_loss, TC, ILR,FLR, LS,ites,Epo,cBS,iBS,str(input),str(CM),time.strftime('%Y%m%d%H%M%S',T),df,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              lossa.minimun_loss)
    return file_record
def logOVTFileKeras(filepath, runs, CM=None, TargetACC=None,mafa=None,
                    TAC=None, OAA=None, TA=None, BN=None, TC=None, 
                    ILR=None, FLR=None, LS=None, iBS=None, 
                    input=None, T=None, df=None, params=-1, Module=None):
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
    if Module==23:
        file_record='Run%02d\tOverAllACC: %s\tTA: %s\tPart1hacc: %s\tPart1oaa: %s\tPart2hacc: %s\tPart2oaa: %s\tbatchNorm:%s\tACCLIST: %s\tTimeComsumed:%s\tInitialLearningRate:%s\tFinalLearningRate:%s\tLearningStepForDroppingMagnitude:%s\tinitialBatchSize:%s\tInput:%s\tTime:%s\tDataFile:%s\tLinearmodelcombinationParameters:%d\n'%(runs, 
                        str(OAA), str(TA), str(mafa[0][0]), str(mafa[0][1]), str(mafa[1][0]), str(mafa[1][1]), str(BN), CMS, str(TC), str(ILR), str(FLR), str(LS),
                        str(iBS),str(input),time.strftime('%Y%m%d%H%M%S',T),df, params)
    else:
        file_record='Run%02d\tOverAllACC: %s\tTA: %s\tTargetACC: %s\tTAC: %s\tbatchNorm:%s\tACCLIST: %s\tTimeComsumed:%s\tInitialLearningRate:%s\tFinalLearningRate:%s\tLearningStepForDroppingMagnitude:%s\tinitialBatchSize:%s\tInput:%s\tTime:%s\tDataFile:%s\tLinearmodelcombinationParameters:%d\n'%(runs, 
                        str(OAA), str(TA), str(TargetACC), str(TAC), str(BN), CMS, str(TC), str(ILR), str(FLR), str(LS),
                        str(iBS),str(input),time.strftime('%Y%m%d%H%M%S',T),df, params)
    fin=open(filepath,'a')
    fin.write(file_record)
    fin.close()
    return True
def logfileKeras(file_record, runs, OAA=None, TA=None, afc=None, BN=None, valid_loss=None, 
                 valid_min_loss=None, final_train_loss=None, train_min_loss=None, TC=None, 
                 ILR=None, FLR=None, LS=None, ites=None, Epo=None, cBS=None, iBS=None, 
                 input=None, CM=None, T=None, df=None):
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
    file_record='Run%02d\tOverAllACC: %s\tTA: %s\tACs: %s\tbatchNorm:%s\tFinalLoss:%s\tMinimunLoss:%s\tFinaltrainloss:%s\tMinimumtrainloss:%s\tTimeComsumed:%s\tInitialLearningRate:%s\tFinalLearningRate:%s\tLearningStepForDroppingMagnitude:%s\tTotalIterations:%s\tEpoches:%s\tcurrentBatchSize:%s\tinitialBatchSize:%s\tInput:%s\t%s\tTime:%s\tDataFile:%s'%(runs, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              str(OAA), str(TA), str(afc), str(BN), str(valid_loss), str(valid_min_loss), str(final_train_loss), 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              str(train_min_loss), str(TC), str(ILR), str(FLR), str(LS), str(ites), str(Epo), str(cBS), 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              str(iBS),str(input),str(CMS),time.strftime('%Y%m%d%H%M%S',T),df)
    return file_record
def tranL3T(input, label, e):
    if int(label) == e:
        input=np.array([1.0, 0.0, 0.0])
    elif int(label) in EG1:
        if e in EG1:
            input=np.array([0.0, 1.0, 0.0])
        else:
            input=np.array([0.0, 0.0, 1.0])
    elif int(label) in EG2:
        if e in EG2:
            input=np.array([0.0, 1.0, 0.0])
        else:
            input=np.array([0.0, 0.0, 1.0]);
    else:
        raise RuntimeError('Error in tranL3T label: %d express: %d'%(label, e))

    return input
def tranL2T(input, label, e):
    if int(label) == e:
        input=np.array([1.0, 0.0]);
    else:
        input=np.array([0.0, 1.0]);
    return input
def transferLabels(labels, newcn, ei):
    '''label should be dense vector instead of onehot vector
    if newcn==3, transfers of labels are as follows:
       Group 1: angry, contempt, disgust, and sad
       Group 2: fear, happy and surprise
       the current expression ei as [0, 0, 1]; expressions from the same group with ei as [0, 1, 0]; 
       expressions from different group as [1, 0, 0];
    if newcn==2, transfers of labels are as follows:
       the current expression ei as [0, 1]; other expression as [1, 0]
    '''
    count=len(labels)
    newlabel=np.zeros((count, newcn))
    if newcn==3:
        for i in range(count):
            newlabel[i]=tranL3T(newlabel[i], labels[i], ei)
    elif newcn==2:
        for i in range(count):
            newlabel[i]=tranL2T(newlabel[i], labels[i], ei)
    else:
        raise RuntimeError('Error in transferLabels with unexpected label setting.')
    return newlabel
#############
#############

##for data loading
Label_Dictionary={0:'angry', 1:'surprise', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'contempt'}
Dataset_Dictionary={
    1:'./Datasets/CK+86.pkl',
    2:'./Datasets/CK+106.pkl',
    3:'./Datasets/CK+107.pkl',
    4:'./Datasets/KDEF6.pkl',
    5:'./Datasets/OuluCASIAVN6.pkl',
    6:'./Datasets/OuluCASIANIR6.pkl',
    8:'H:/Datasets/SFEW2.0_7_M1_M3_M4_M11s_TestID_0_TrainID_1_vs1311.pkl',
    11:'./Datasets/CK+86_M11.pkl',
    12:'./Datasets/CK+106_M11.pkl',
    13:'./Datasets/CK+107_M11.pkl',
    14:'./Datasets/KDEF6_M11.pkl',
    15:'./Datasets/OuluCASIAVN6_M11.pkl',
    16:'./Datasets/OuluCASIANIR6_M11.pkl',
    #17:'./Datasets/KDEF6_M11s.pkl', #old version and discarded on 20190610
    #18:'./Datasets/OuluCASIAVN6_M11s.pkl', #old version and discarded on 20190610
    #19:'./Datasets/KDEF6_M11withoutWavelet.pkl', #old version and discarded on 20190610
    #20:'./Datasets/OuluCASIAVN6_M11withoutWavelet.pkl', #old version and discarded on 20190610
    21:'./Datasets/CK+86_M11s.pkl',
    22:'./Datasets/CK+106_M11s.pkl',
    23:'./Datasets/CK+107_M11s.pkl',
    24:'./Datasets/KDEF6_M11s.pkl',
    25:'./Datasets/OuluCASIAVN6_M11s.pkl',
    26:'./Datasets/OuluCASIANIR6_M11s.pkl',
    27:'H:/Datasets/MNIST_M1_M4_M11s.pkl',
    28:'H:/Datasets/CIFAR10_gray_M1_M4_M11s.pkl',
    29:'H:/Datasets/CIFAR10_rgb_M1_M4_M11s.pkl',
    31:'./Datasets/CK+86_M11withoutWavelet.pkl',#
    32:'./Datasets/CK+106_M11withoutWavelet.pkl',#
    33:'./Datasets/CK+107_M11withoutWavelet.pkl',#
    34:'./Datasets/KDEF6_M11withoutWavelet.pkl',#D3x is poorer than D2x
    35:'./Datasets/OuluCASIAVN6_M11withoutWavelet.pkl',#D3x is better than D2x
    36:'./Datasets/OuluCASIANIR6_M11withoutWavelet.pkl',#D3x is better than D2x
    37:'H:/Datasets/MNIST_M1_M4_M11withoutWavelet.pkl',
    38:'H:/Datasets/CIFAR10_gray_M1_M4_M11withoutWavelet.pkl',
    39:'H:/Datasets/CIFAR10_rgb_M1_M4_M11withoutWavelet.pkl',
    #41:'./Datasets/CK+86_M11withLogarithm.pkl',###very poor performance
    #42:'./Datasets/CK+106_M11withLogarithm.pkl',###very poor performance
    #43:'./Datasets/CK+107_M11withLogarithm.pkl',###very poor performance
    #44:'./Datasets/KDEF6_M11withLogarithm.pkl',###very poor performance
    #45:'./Datasets/OuluCASIAVN6_M11withLogarithm.pkl',###very poor performance
    #46:'./Datasets/OuluCASIANIR6_M11withLogarithm.pkl',###very poor performance
    51:'H:/Datasets/CK+86_M2_M3Wavelet.pkl',
    52:'H:/Datasets/CK+106_M2_M3Wavelet.pkl',
    53:'H:/Datasets/CK+107_M2_M3Wavelet.pkl',
    54:'H:/Datasets/KDEF6_M2_M3Wavelet.pkl',
    55:'H:/Datasets/OuluCASIAVN6_M2_M3Wavelet.pkl',
    56:'H:/Datasets/OuluCASIANIR6_M2_M3Wavelet.pkl',
    58:'H:/Datasets/SFEW2.0_7_M1Wavelet_M3Wavelet_M4Wavelet_M11s_TestID_0_TrainID_1_vs1311.pkl',
    61:'H:/Datasets/CK+86_M1WaL1C3_M2_M3WaL1C3_M4WaL1C3_M11withoutWavelet_vs927.pkl',
    62:'H:/Datasets/CK+106_M1WaL1C3_M2_M3WaL1C3_M4WaL1C3_M11withoutWavelet_vs922.pkl',
    63:'H:/Datasets/CK+107_M1WaL1C3_M2_M3WaL1C3_M4WaL1C3_M11withoutWavelet_vs975.pkl',
    64:'H:/Datasets/KDEF6_M1WaL1C3_M2_M3WaL1C3_M4WaL1C3_M11withoutWavelet_vs840.pkl',
    65:'H:/Datasets/OuluCASIAVN6_M1WaL1C3_M2_M3WaL1C3_M4WaL1C3_M11withoutWavelet_vs1440.pkl',
    66:'H:/Datasets/OuluCASIANIR6_M2_M3WaL1C3_vs4320.pkl',
    67:'H:/Datasets/SFEW2_7v2_M1WaL1C3_M3WaL1C3_M4WaL1C3_M11withoutWavelet_vs1719.pkl',
    68:'H:/Datasets/SFEW2.0_7_M1WaL1C3_M3WaL1C3_M4WaL1C3_M11withoutWavelet_TestID_0_TrainID_1_vs1311.pkl',
    69:'H:/Datasets/FER2013plus_48x48_TestID0_trainID1_M4WaL1C3_M11withoutWavelet_vs32110.pkl',
    690:'H:/Datasets/FER2013plus_48x48_TestID0_trainID1_equal_M4WaL1C3_M11withoutWavelet_vs86585.pkl',
    71:'H:/Datasets/CK+86_FLIP_enlarge_M2_M3WaL1C3_vs1854.pkl',
    72:'H:/Datasets/CK+106_FLIP_enlarge_M2_M3WaL1C3_vs1844.pkl',
    73:'H:/Datasets/CK+107_FLIP_enlarge_M2_M3WaL1C3_vs1950.pkl',
    74:'H:/Datasets/KDEF6_FLIP_enlarge_M2_M3WaL1C3_vs1680.pkl',
    75:'H:/Datasets/OuluCASIAVN6_FLIP_enlarge_M2_M3WaL1C3_vs2877.pkl',
    76:'H:/Datasets/OuluCASIANIR6_FLIP_enlarge_M2_M3WaL1C3_vs8640.pkl',
    ##########PairDiff
    81:'H:/Datasets/CK+86_pair_M2PAIRDiff_M11sPAIRDiff_vs927.pkl',
    82:'H:/Datasets/CK+106_pair_M2PAIRDiff_M11sPAIRDiff_vs922.pkl',
    83:'H:/Datasets/CK+107_pair_M2PAIRDiff_M11sPAIRDiff_vs975.pkl',
    84:'H:/Datasets/KDEF6_pair_M2PAIRDiff_M11sPAIRDiff_vs838.pkl',
    85:'H:/Datasets/OuluCASIAVN6_pair_M2PAIRDiff_M11sPAIRDiff_vs1440.pkl',
    86:'H:/Datasets/OuluCASIANIR6_pair_M2PAIRDiff_M11sPAIRDiff_vs4320.pkl',
    91:'H:/Datasets/CK+86_pair_M2PAIRDiff_M11withoutWaveletPAIRDiff_vs927.pkl',
    92:'H:/Datasets/CK+106_pair_M2PAIRDiff_M11withoutWaveletPAIRDiff_vs922.pkl',
    93:'H:/Datasets/CK+107_pair_M2PAIRDiff_M11withoutWaveletPAIRDiff_vs975.pkl',
    94:'H:/Datasets/KDEF6_pair_M2PAIRDiff_M11withoutWaveletPAIRDiff_vs838.pkl',
    95:'H:/Datasets/OuluCASIAVN6_pair_M2PAIRDiff_M11withoutWaveletPAIRDiff_vs1440.pkl',
    96:'H:/Datasets/OuluCASIANIR6_pair_M2PAIRDiff_M11withoutWaveletPAIRDiff_vs4320.pkl',
    101:'H:/Datasets/CK+86_pair_M2_M11withoutWaveletPAIRDiff_vs927.pkl',#####To tested on M211
    102:'H:/Datasets/CK+106_pair_M2_M11withoutWaveletPAIRDiff_vs922.pkl',#####To tested on M211
    103:'H:/Datasets/CK+107_pair_M2_M11withoutWaveletPAIRDiff_vs975.pkl',#####To tested on M211
    104:'H:/Datasets/KDEF6_pair_M2_M11withoutWaveletPAIRDiff_vs840.pkl',#####To tested on M211
    105:'H:/Datasets/OuluCASIAVN6_pair_M2_M11withoutWaveletPAIRDiff_vs1440.pkl',#####To tested on M211
    106:'H:/Datasets/OuluCASIANIR6_pair_M2_M11withoutWaveletPAIRDiff_vs4320.pkl',#####To tested on M211
    ##########
    #######enlarge data corresponding to 51-56
    151:'H:/Datasets/CK+86_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_M2_M3Wavelet_vs9270.pkl',
    152:'H:/Datasets/CK+106_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_M2_M3Wavelet_vs9220.pkl',
    153:'H:/Datasets/CK+107_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_M2_M3Wavelet_vs9750.pkl',
    154:'H:/Datasets/KDEF6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_M2_M3Wavelet_vs8400.pkl',
    155:'H:/Datasets/OuluCASIAVN6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_M2_M3Wavelet_vs13493.pkl',
    156:'H:/Datasets/OuluCASIANIR6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_M2_M3Wavelet_vs40626.pkl',
    #########
    121:'H:/Datasets/CK+86_FLIPNS_M11s.pkl',
    122:'H:/Datasets/CK+106_FLIPNS_M11s.pkl',
    123:'H:/Datasets/CK+107_FLIPNS_M11s.pkl',
    124:'H:/Datasets/KDEF6_FLIPNS_M11s.pkl',
    #125:'./Datasets/OuluCASIAVN6_M11s.pkl',
    #126:'./Datasets/OuluCASIANIR6_M11s.pkl',
    #127:'./Datasets/MNIST_M1_M4_M11s.pkl',
    #128:'./Datasets/CIFAR10_gray_M1_M4_M11s.pkl',
    #129:'./Datasets/CIFAR10_rgb_M1_M4_M11s.pkl',
    131:'H:/Datasets/CK+86_FLIPNS_M11withoutWavelet.pkl',#
    132:'H:/Datasets/CK+106_FLIPNS_M11withoutWavelet.pkl',#
    133:'H:/Datasets/CK+107_FLIPNS_M11withoutWavelet.pkl',#
    134:'H:/Datasets/KDEF6_FLIPNS_M11withoutWavelet.pkl',#
    #135:'./Datasets/OuluCASIAVN6_M11withoutWavelet.pkl',
    #136:'./Datasets/OuluCASIANIR6_M11withoutWavelet.pkl',
    221:'H:/Datasets/CK+86_FLIPNST10_M11s.pkl',
    222:'H:/Datasets/CK+106_FLIPNST10_M11s.pkl',
    223:'H:/Datasets/CK+107_FLIPNST10_M11s.pkl',
    224:'H:/Datasets/KDEF6_FLIPNST10_M11s.pkl',
    231:'H:/Datasets/CK+86_FLIPNST10_M11withoutWavelet.pkl',#
    232:'H:/Datasets/CK+106_FLIPNST10_M11withoutWavelet.pkl',#
    233:'H:/Datasets/CK+107_FLIPNST10_M11withoutWavelet.pkl',#
    234:'H:/Datasets/KDEF6_FLIPNST10_M11withoutWavelet.pkl',#
    321:'H:/Datasets/CK+86_FLIPNST6_ACIT10_M11s.pkl',
    322:'H:/Datasets/CK+106_FLIPNST6_ACIT10_M11s.pkl',
    323:'H:/Datasets/CK+107_FLIPNST6_ACIT10_M11s.pkl',
    324:'H:/Datasets/KDEF6_FLIPNST6_ACIT10_M11s.pkl',
    331:'H:/Datasets/CK+86_FLIPNST6_ACIT10_M11withoutWavelet.pkl',#
    332:'H:/Datasets/CK+106_FLIPNST6_ACIT10_M11withoutWavelet.pkl',#
    333:'H:/Datasets/CK+107_FLIPNST6_ACIT10_M11withoutWavelet.pkl',#
    334:'H:/Datasets/KDEF6_FLIPNST6_ACIT10_M11withoutWavelet.pkl',#
    ####
    401:'H:/Datasets/CK+86_M1_M3_M4_M11withoutWavelet_vs927.pkl',
    402:'H:/Datasets/CK+106_M1_M3_M4_M11withoutWavelet_vs922.pkl',
    403:'H:/Datasets/CK+107_M1_M3_M4_M11withoutWavelet_vs975.pkl',
    404:'H:/Datasets/KDEF6_M1_M3_M4_M11withoutWavelet_vs840.pkl',
    405:'H:/Datasets/OuluCASIAVN6_M1_M3_M4_M11withoutWavelet_vs1440.pkl',
    407:'H:/Datasets/SFEW2_7v2_M1_M2_M3_M4_M11withoutWavelet_vs1719.pkl',
    408:'H:/Datasets/SFEW2.0_7_M1_M3_M4_M11withoutWavelet_TestID_0_TrainID_1_vs1311.pkl',
    409:'H:/Datasets/FER2013plus_48x48_TestID0_trainID1_M4_M11withoutWavelet_vs32110.pkl',
    4090:'H:/Datasets/FER2013plus_48x48_TestID0_trainID1_equal_M4_M11withoutWavelet_vs86585.pkl',
    411:'H:/Datasets/CK+86_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_vs927.pkl',
    412:'H:/Datasets/CK+106_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_vs922.pkl',
    413:'H:/Datasets/CK+107_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_vs975.pkl',
    414:'H:/Datasets/KDEF6_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_vs840.pkl',
    415:'H:/Datasets/OuluCASIAVN6_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_vs1440.pkl',
    417:'H:/Datasets/SFEW2_7v2_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_vs1719.pkl',
    418:'H:/Datasets/SFEW2.0_7_M1Wavelet_M3Wavelet_M4Wavelet_M11withoutWavelet_TestID_0_TrainID_1_vs1311.pkl',
    419:'H:/Datasets/FER2013plus_48x48_TestID0_trainID1_M4Wavelet_M11withoutWavelet_vs32110.pkl',
    4190:'H:/Datasets/FER2013plus_48x48_TestID0_trainID1_equal_M4Wavelet_M11withoutWavelet_vs86585.pkl',
    }

#Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])
Datasets = collections.namedtuple('Datasets', ['train', 'test'])
def dense_to_one_hot(labels_dense, num_classes):
    '''Convert class labels from scalars to one-hot vectors.'''
    if type(labels_dense[0]) is not int and type(labels_dense[0]) is not np.int32:
        if len(labels_dense[0])==num_classes:
            return labels_dense
        elif len(labels_dense[0])>1:
            raise RuntimeError('Unexpected data format.')
    num_labels = len(labels_dense)
    #index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #print('Total %d'%num_labels)
    for i in range(num_labels):
        labels_one_hot[i,int(labels_dense[i])]=1
        #print('%d\t%s\t%s'%(i, labels_dense[i], str(labels_one_hot[i])))
    return labels_one_hot
#def listShuffle(listV, ind):
#    tm=listV[:]
#    for i, v in enumerate(ind):
#        tm[i]=listV[v]
#    return tm
def listShuffle(listV, ind):
    tm=[]
    for v in ind:
        tm.append(listV[v])
    listV.clear()
    return tm
class DataSetFor3KindsDataV4(object):
    '''nextbatch function returns a tuple of lists (imgs, geometry, eyep, foreheadp, mouthp, labels),
each with length of batch size
Class Adapted from tensorflow.mnist 
Input imgs, geometry, eyep, foreheadp, mouthp, and labels must be list objects
imgs datatype should be unit8

the following parameters are deprecated:
    reshape_size, Df, reshape
    leave them to their default values.

Patching: 20190423 >>>>add component wcs and corresponding implementations
Patch: 20190514 >>>> all datatype are transfered into np.float64

'''
    def __init__(self, imgs=None, labels=None, geometry=None, eyep=None, foreheadp=None, mouthp=None, cropf=None, wcs=None, 
                 one_hot=True, num_Classes=7, Df=False, loadImg=False, loadGeo=False, loadPat=False, loadInnerF=False, loadWcs=False,
                 reshape=False, reshape_size=(64, 64)):
        '''Becareful, the inputs of this function will be deleted.'''
        if loadImg or loadGeo or loadPat or loadInnerF or loadWcs:
            self._num_examples = len(labels)
            if loadImg:
                assert len(imgs) == len(labels), ('imgs length: %d labels length: %d' % (len(imgs), len(labels)))
                self._res_images = imgs[:]
                del imgs

                if str(self._res_images[0].dtype) == 'uint8':#modified in 20190227
                    print('Original Data Type: Uint8, Transfering into Float64 with Range in [0, 1]')
                    # Convert from [0, 255] -> [0.0, 1.0].
                    if len(self._res_images[0].shape)==2:
                        for i in range(self._num_examples):
                            self._res_images[i] = self._res_images[i].astype(np.float64)
                            self._res_images[i] = np.multiply(self._res_images[i], scale_factor)
                            r,c=self._res_images[i].shape
                            self._res_images[i] = np.reshape(self._res_images[i], [r, c, 1])
                    else:#modified in 20190514
                        for i in range(self._num_examples):
                            self._res_images[i] = self._res_images[i].astype(np.float64)
                            self._res_images[i] = np.multiply(self._res_images[i], scale_factor)
                else:#modified in 20190227, 20190516
                    if str(self._res_images[0].dtype) == 'float64':
                        print('Original Data shape: %s, Transfering into 3 channels'%(str(self._res_images[0].shape)))
                        if len(self._res_images[0].shape)==2:
                            for i in range(self._num_examples):
                                r,c=self._res_images[i].shape
                                self._res_images[i] = np.reshape(self._res_images[i], [r, c, 1])
                    else:
                        print('Original Data Type: %s, Transfering into Float64'%(str(self._res_images[0].dtype)))
                        if len(self._res_images[0].shape)==2:
                            for i in range(self._num_examples):
                                self._res_images[i] = self._res_images[i].astype(np.float64)
                                r,c=self._res_images[i].shape
                                self._res_images[i] = np.reshape(self._res_images[i], [r, c, 1])
                        else:#modified in 20190514
                            for i in range(self._num_examples):
                                self._res_images[i] = self._res_images[i].astype(np.float64)
            else:
                self._res_images = None
            if loadGeo:
                assert len(geometry) == len(labels), ('geometry length: %d labels length: %d' % (len(geometry), len(labels)))
                self._geometry = geometry[:]
                del geometry
            else:
                self._geometry = None
            if loadPat:
                assert len(eyep) == len(labels), ('eye_patch length: %d labels length: %d' % (len(eyep), len(labels)))
                assert len(foreheadp) == len(labels), ('middle_patch length: %d labels length: %d' % (len(foreheadp), len(labels)))
                assert len(mouthp) == len(labels), ('mouth_patch length: %d labels length: %d' % (len(mouthp), len(labels)))
                self._eyep = eyep[:]
                del eyep
                self._foreheadp = foreheadp[:]
                del foreheadp
                self._mouthp = mouthp[:]
                del mouthp

                if str(self._eyep[0].dtype) == 'uint8':#modified in 20180709
                    print('Original Data Type: Uint8, Transfering into Float64 with Range in [0, 1]')
                    # Convert from [0, 255] -> [0.0, 1.0].
                    if len(self._eyep[0].shape)==2:
                        for i in range(self._num_examples):
                            self._eyep[i] = self._eyep[i].astype(np.float64)
                            self._eyep[i] = np.multiply(self._eyep[i], scale_factor)
                            r,c=self._eyep[i].shape
                            self._eyep[i] = np.reshape(self._eyep[i], [r, c, 1])
                
                            self._foreheadp[i] = self._foreheadp[i].astype(np.float64)
                            self._foreheadp[i] = np.multiply(self._foreheadp[i], scale_factor)
                            r,c=self._foreheadp[i].shape
                            self._foreheadp[i] = np.reshape(self._foreheadp[i], [r, c, 1])
                
                            self._mouthp[i] = self._mouthp[i].astype(np.float64)
                            self._mouthp[i] = np.multiply(self._mouthp[i], scale_factor)
                            r,c=self._mouthp[i].shape
                            self._mouthp[i] = np.reshape(self._mouthp[i], [r, c, 1])
                    else:#modified in 20190514
                            self._eyep[i] = self._eyep[i].astype(np.float64)
                            self._eyep[i] = np.multiply(self._eyep[i], scale_factor)
                            self._foreheadp[i] = self._foreheadp[i].astype(np.float64)
                            self._foreheadp[i] = np.multiply(self._foreheadp[i], scale_factor)
                            self._mouthp[i] = self._mouthp[i].astype(np.float64)
                            self._mouthp[i] = np.multiply(self._mouthp[i], scale_factor)
                else:#modified in 20180709, 20190516
                    if str(self._eyep[0].dtype) == 'float64':
                        print('Original Data shape: %s, Transfering into 3 channels'%(str(self._eyep[0].shape)))
                        if len(self._eyep[0].shape)==2:
                            for i in range(self._num_examples):
                                r,c=self._eyep[i].shape
                                self._eyep[i] = np.reshape(self._eyep[i], [r, c, 1])
                
                                r,c=self._foreheadp[i].shape
                                self._foreheadp[i] = np.reshape(self._foreheadp[i], [r, c, 1])
                
                                r,c=self._mouthp[i].shape
                                self._mouthp[i] = np.reshape(self._mouthp[i], [r, c, 1])
                    else:
                        print('Original Data Type: %s, Transfering into Float64'%(str(self._eyep[0].dtype)))
                        if len(self._eyep[0].shape)==2:
                            for i in range(self._num_examples):
                                self._eyep[i] = self._eyep[i].astype(np.float64)
                                r,c=self._eyep[i].shape
                                self._eyep[i] = np.reshape(self._eyep[i], [r, c, 1])
                
                                self._foreheadp[i] = self._foreheadp[i].astype(np.float64)
                                r,c=self._foreheadp[i].shape
                                self._foreheadp[i] = np.reshape(self._foreheadp[i], [r, c, 1])
                
                                self._mouthp[i] = self._mouthp[i].astype(np.float64)
                                r,c=self._mouthp[i].shape
                                self._mouthp[i] = np.reshape(self._mouthp[i], [r, c, 1])
                        else:#modified in 20190514
                            for i in range(self._num_examples):
                                self._eyep[i] = self._eyep[i].astype(np.float64)
                                self._foreheadp[i] = self._foreheadp[i].astype(np.float64)
                                self._mouthp[i] = self._mouthp[i].astype(np.float64)

            else:
                self._eyep = None
                self._foreheadp = None
                self._mouthp = None
            if loadInnerF:
                if loadImg or loadGeo or loadPat:
                    raise RuntimeError('ERROR in __init__ of DataSetFor3KindsDataV4: Unexpected case for nextbatch logic')
                assert len(cropf) == len(labels), ('inner_face length: %d labels length: %d' % (len(cropf), len(labels)))
                self._cropf = cropf[:]
                del cropf

                if str(self._cropf[0].dtype) == 'uint8':#modified in 20190303
                    print('Original Data Type: Uint8, Transfering into Float64 with Range in [0, 1]')
                    if len(self._cropf[0].shape)==2:
                        for i in range(self._num_examples):
                            self._cropf[i] = self._cropf[i].astype(np.float64)
                            self._cropf[i] = np.multiply(self._cropf[i], scale_factor)
                            r,c=self._cropf[i].shape
                            self._cropf[i] = np.reshape(self._cropf[i], [r, c, 1])
                    else:#modified in 20190514
                        for i in range(self._num_examples):
                            self._cropf[i] = self._cropf[i].astype(np.float64)
                            self._cropf[i] = np.multiply(self._cropf[i], scale_factor)

                else:#modified in 20190303, 20190516, 20191005
                    if str(self._cropf[0].dtype) == 'float64':
                        print('Original Data shape: %s, Transfering into 3 channels'%(str(self._cropf[0].shape)))
                        if len(self._cropf[0].shape)==2:
                            for i in range(self._num_examples):
                                r,c=self._cropf[i].shape
                                self._cropf[i] = np.reshape(self._cropf[i], [r, c, 1])
                    else:
                        if len(self._cropf[0].shape)==2:
                            for i in range(self._num_examples):
                                self._cropf[i] = self._cropf[i].astype(np.float64)
                                r,c=self._cropf[i].shape
                                self._cropf[i] = np.reshape(self._cropf[i], [r, c, 1])
                        else:#modified in 20190514
                            for i in range(self._num_examples):
                                self._cropf[i] = self._cropf[i].astype(np.float64)
            else:
                self._cropf = None
            if loadWcs:#patched in 201904
                if loadImg or loadPat:
                    raise RuntimeError('ERROR in __init__ of DataSetFor3KindsDataV4: Unexpected case for nextbatch logic')
                assert len(wcs) == len(labels), ('inner_face length: %d labels length: %d' % (len(wcs), len(labels)))
                self._wcs = wcs[:]
                del wcs
                #print(self._wcs[0].dtype)
                #exit()
                if str(self._wcs[0].dtype) == 'uint8':#modified in 20190303
                    print('Original Data Type: Uint8, Transfering into Float64 with Range in [0, 1]')
                    if len(self._wcs[0].shape)==2:
                        for i in range(self._num_examples):
                            self._wcs[i] = self._wcs[i].astype(np.float64)
                            self._wcs[i] = np.multiply(self._wcs[i], scale_factor)
                            r,c=self._wcs[i].shape
                            self._wcs[i] = np.reshape(self._wcs[i], [r, c, 1])
                    else:#modified in 20190514
                        for i in range(self._num_examples):
                            self._wcs[i] = self._wcs[i].astype(np.float64)
                            self._wcs[i] = np.multiply(self._wcs[i], scale_factor)
                else:#modified in 20190516 20191005
                    if str(self._wcs[0].dtype)=='float64':
                        if len(self._wcs[0].shape)==2:
                            print('Original Data shape: %s, reshape into 3 channels'%(str(self._wcs[0].shape)))
                            for i in range(self._num_examples):
                                r,c=self._wcs[i].shape
                                self._wcs[i] = np.reshape(self._wcs[i], [r, c, 1])
                    else:
                        print('Original Data Type: %s, Transfering into Float64 and reshape into 3 channels'%(str(self._wcs[0].dtype)))
                        if len(self._wcs[0].shape)==2:
                            for i in range(self._num_examples):
                                self._wcs[i] = self._wcs[i].astype(np.float64)
                                r,c=self._wcs[i].shape
                                self._wcs[i] = np.reshape(self._wcs[i], [r, c, 1])
                        else:#modified in 20190514
                            for i in range(self._num_examples):
                                self._wcs[i] = self._wcs[i].astype(np.float64)
            else:
                self._wcs = None
            if one_hot and num_Classes>3:
                self._labels=dense_to_one_hot(labels, num_Classes)
            else:
                self._labels = np.asarray(labels[:])
            del labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            self.__loadGeo=loadGeo
            self.__loadPat=loadPat   
            self.__loadImg=loadImg
            self.__loadInnerF=loadInnerF
            self.__loadWcs=loadWcs
        else:
            raise RuntimeError('%*%*%**%*%*%*ERROR: Must load one of the feature Module')


    def reset(self, labels=None, imgs=None, geometry=None, eyep=None, foreheadp=None, mouthp=None, cropf=None, wcs=None):
        if labels is None:
            return False
        if self.__loadImg and not(imgs is None):
            assert len(imgs) == len(labels), ('imgs length: %d labels length: %d' % (len(imgs), len(labels)))
            self._res_images = imgs[:]
        if self.__loadGeo and not(geometry is None):
            assert len(geometry) == len(labels), ('geometry length: %d labels length: %d' % (len(geometry), len(labels)))
            self._geometry = geometry[:]
        if self.__loadPat and not(eyep is None) and not(foreheadp is None) and not(mouthp is None):
            assert len(eyep) == len(labels), ('eye_patch length: %d labels length: %d' % (len(eyep), len(labels)))
            assert len(foreheadp) == len(labels), ('middle_patch length: %d labels length: %d' % (len(foreheadp), len(labels)))
            assert len(mouthp) == len(labels), ('mouth_patch length: %d labels length: %d' % (len(mouthp), len(labels)))
            self._eyep = eyep[:]
            self._foreheadp = foreheadp[:]
            self._mouthp = mouthp[:]
        if self.__loadInnerF and not(cropf is None):
            assert len(cropf) == len(labels), ('inner_face length: %d labels length: %d' % (len(cropf), len(labels)))
            self._cropf = cropf[:]
        if self.__loadWcs and not(wcs is None):
            assert len(wcs) == len(labels), ('wcs length: %d labels length: %d' % (len(wcs), len(labels)))
            self._wcs = wcs[:]
        if imgs is None and geometry is None and eyep is None and foreheadp is None and mouthp is None and cropf is None and wcs is None:
            assert self._num_examples == len(labels), ('Inconsistent lengths. orginal length: %d labels length: %d' % (len(cropf), len(labels)))
            self._labels = labels[:]
        else:
            self._num_examples = len(labels)
            self._labels = labels[:]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        return True

    def resetIndex(self):
        self._index_in_epoch = 0
        print('_index_in_epoch has been reset.')
        return True
    
    @property
    def imgs(self):
        return self._res_images

    @property
    def geo(self):
        return self._geometry

    @property
    def eyep(self):
        return self._eyep

    @property
    def foreheadp(self):
        return self._foreheadp

    @property
    def mouthp(self):
        return self._mouthp
    
    @property
    def cropf(self):
        return self._cropf

    @property
    def wcs(self):
        return self._wcs

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=False):
        '''nextbatch function returns a tuple of lists (imgs, geometry, eyep, foreheadp, mouthp, labels), each with length of batch_size from this data set.'''
        start = self._index_in_epoch
        if self.__loadGeo and self.__loadPat and self.__loadImg:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._res_images = listShuffle(self.imgs, perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._eyep = listShuffle( self.eyep, perm0)
                self._foreheadp = listShuffle( self.foreheadp, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._cropf = listShuffle( self._cropf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])


                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                innerf_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._foreheadp[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])
                innerf_rest_part.extend(self._cropf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._res_images = listShuffle(self.imgs, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._foreheadp = listShuffle( self.foreheadp, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                    self._cropf = listShuffle( self._cropf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._foreheadp[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])
                innerf_rest_part.extend(self._cropf[start:end])

                return images_rest_part, geometry_rest_part, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], self._geometry[start:end], self._eyep[start:end], self._foreheadp[start:end], self._mouthp[start:end], self._labels[start:end], self._cropf[start:end]
        elif self.__loadPat and self.__loadImg:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._res_images = listShuffle(self.imgs, perm0)
  
                self._eyep = listShuffle( self.eyep, perm0)
                self._foreheadp = listShuffle( self.foreheadp, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._cropf = listShuffle( self._cropf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                innerf_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._foreheadp[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])
                innerf_rest_part.extend(self._cropf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._res_images = listShuffle(self.imgs, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._foreheadp = listShuffle( self.foreheadp, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                    self._cropf = listShuffle( self._cropf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._foreheadp[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])
                innerf_rest_part.extend(self._cropf[start:end])

                return images_rest_part, None, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], None, self._eyep[start:end], self._foreheadp[start:end], self._mouthp[start:end], self._labels[start:end], self._cropf[start:end]
        elif self.__loadGeo and self.__loadImg:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._res_images = listShuffle(self.imgs, perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._res_images = listShuffle(self.imgs, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                return images_rest_part, geometry_rest_part, None, None, None, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], self._geometry[start:end], None, None, None, self._labels[start:end], None
        elif self.__loadGeo and self.__loadPat:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._eyep = listShuffle( self.eyep, perm0)
                self._foreheadp = listShuffle( self.foreheadp, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)

                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._foreheadp[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._foreheadp = listShuffle( self.foreheadp, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._foreheadp[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])

                return None, geometry_rest_part, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                return None, self._geometry[start:end], self._eyep[start:end], self._foreheadp[start:end], self._mouthp[start:end], self._labels[start:end], None
        elif self.__loadGeo and self.__loadWcs:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._wcs = listShuffle(self.wcs, perm0)
  
                self._geometry = listShuffle(self.geometry, perm0)
  
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                wcs_rest_part = []
                labels_rest_part = []
                wcs_rest_part.extend(self._wcs[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._wcs = listShuffle(self.wcs, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._geometry = listShuffle(self.geometry, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                wcs_rest_part.extend(self._wcs[start:end])
                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                return None, geometry_rest_part, None, None, None, labels_rest_part, None, wcs_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                return None, self._geometry[start:end], None, None, None, self._labels[start:end], None, self._wcs[start:end]
        elif self.__loadInnerF and self.__loadWcs:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._wcs = listShuffle(self.wcs, perm0)
  
                self._cropf = listShuffle(self.cropf, perm0)
  
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                wcs_rest_part = []
                labels_rest_part = []
                wcs_rest_part.extend(self._wcs[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                cropf_rest_part = []
                cropf_rest_part.extend(self._cropf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._wcs = listShuffle(self.wcs, perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._cropf = listShuffle(self.cropf, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                wcs_rest_part.extend(self._wcs[start:end])
                labels_rest_part.extend(self._labels[start:end])

                cropf_rest_part.extend(self._cropf[start:end])

                return None, None, None, None, None, labels_rest_part, cropf_rest_part, wcs_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                return None, None, None, None, None, self._labels[start:end], self._cropf[start:end], self._wcs[start:end]
        elif self.__loadGeo:
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._labels = listShuffle( self.labels, perm0)
                self._geometry = listShuffle(self.geometry, perm0)
  
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                geometry_rest_part = []
                geometry_rest_part.extend(self._geometry[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)
                    self._geometry = listShuffle(self.geometry, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                geometry_rest_part.extend(self._geometry[start:end])

                return None, geometry_rest_part, None, None, None, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                return None, self._geometry[start:end], None, None, None, self._labels[start:end], None
        elif self.__loadImg:
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._res_images = listShuffle(self.imgs, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                images_rest_part = []
                labels_rest_part = []
                images_rest_part.extend(self._res_images[start:self._num_examples])
                labels_rest_part.extend(self._labels[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._res_images = listShuffle(self.imgs, perm)
                    self._labels = listShuffle( self.labels, perm)

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                images_rest_part.extend(self._res_images[start:end])
                labels_rest_part.extend(self._labels[start:end])

                return images_rest_part, None, None, None, None, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return self._res_images[start:end], None, None, None, None, self._labels[start:end], None
        elif self.__loadPat:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._eyep = listShuffle( self.eyep, perm0)
                self._foreheadp = listShuffle( self.foreheadp, perm0)
                self._mouthp = listShuffle( self.mouthp, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                eyep_rest_part = []
                middlep_rest_part = []
                mouthp_rest_part = []
                eyep_rest_part.extend(self._eyep[start:self._num_examples])
                middlep_rest_part.extend(self._foreheadp[start:self._num_examples])
                mouthp_rest_part.extend(self._mouthp[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._eyep = listShuffle( self.eyep, perm)
                    self._foreheadp = listShuffle( self.foreheadp, perm)
                    self._mouthp = listShuffle( self.mouthp, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                eyep_rest_part.extend(self._eyep[start:end])
                middlep_rest_part.extend(self._foreheadp[start:end])
                mouthp_rest_part.extend(self._mouthp[start:end])

                return None, None, eyep_rest_part, middlep_rest_part, mouthp_rest_part, labels_rest_part, None
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return None, None, self._eyep[start:end], self._foreheadp[start:end], self._mouthp[start:end], self._labels[start:end], None
        elif self.__loadInnerF:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._cropf = listShuffle( self._cropf, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                innerf_rest_part = []
                innerf_rest_part.extend(self._cropf[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._cropf = listShuffle( self._cropf, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                innerf_rest_part.extend(self._cropf[start:end])

                return None, None, None, None, None, labels_rest_part, innerf_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return None, None, None, None, None, self._labels[start:end], self._cropf[start:end]
        elif self.__loadWcs:
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0 and shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._wcs = listShuffle( self._wcs, perm0)
                self._labels = listShuffle( self.labels, perm0)
            # Go to the next epoch
            if start + batch_size > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1
                #print('Epoche: %d'%self._epochs_completed)
                # Get the rest examples in this epoch
                rest_num_examples = self._num_examples - start

                labels_rest_part = []
                labels_rest_part.extend(self._labels[start:self._num_examples])

                wcs_rest_part = []
                wcs_rest_part.extend(self._wcs[start:self._num_examples])

                # Shuffle the data
                if shuffle:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._labels = listShuffle( self.labels, perm)

                    self._wcs = listShuffle( self._wcs, perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch

                labels_rest_part.extend(self._labels[start:end])

                wcs_rest_part.extend(self._wcs[start:end])

                return None, None, None, None, None, labels_rest_part, None, wcs_rest_part
            else:
                self._index_in_epoch += batch_size
                end = self._index_in_epoch
                #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
                return None, None, None, None, None, self._labels[start:end], None, self._wcs[start:end]
        return 
class DataSetForAnonymousData(object):
    '''nextbatch function returns a tuple of lists (X, Y),
each with length of batch size
Class Adapted from tensorflow.mnist 

Patching: 20190423 >>>>add component wcs and corresponding implementations
Patch: 20190514 >>>> all datatype are transfered into np.float64
'''
    def __init__(self, X:np.ndarray=None, Y:np.ndarray=None):
        '''Becareful, the inputs of this function will be deleted.'''
        self._num_examples = len(Y)
        assert len(X) == len(Y), ('X length: %d Y length: %d' % (len(X), len(Y)))
        self.X=list(X)
        self.Y=list(Y)
                
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def reset(self, X:np.ndarray, Y:np.ndarray):
        self.X=list(X)
        self.Y=list(Y)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        return True

    def resetIndex(self):
        self._index_in_epoch = 0
        print('_index_in_epoch has been reset.')
        return True
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=False):
        '''nextbatch function returns a tuple of lists (imgs, geometry, eyep, foreheadp, mouthp, labels), each with length of batch_size from this data set.'''
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.X = listShuffle( self.X, perm0)
            self.Y = listShuffle( self.Y, perm0)
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            #print('Epoche: %d'%self._epochs_completed)
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start

            Y_rest_part = []
            Y_rest_part.extend(self.Y[start:self._num_examples])
            X_rest_part = []
            X_rest_part.extend(self.X[start:self._num_examples])
            
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.Y = listShuffle( self.Y, perm)
                self.X = listShuffle( self.X, perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            Y_rest_part.extend(self.Y[start:end])
            X_rest_part.extend(self.X[start:end])

            return X_rest_part, Y_rest_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            #print('%d\t%d'%(len(self._res_images[start:end]),len(self._labels[start:end])))
            return self.X[start:end], self.Y[start:end]

def loaddata_v4(datafilepath, validation_no=0, test_no=0, Module=0, Df=False, one_hot=True, reshape=False, cn=7):
    '''return the CKplus preprocessed data in a class with convinient function to access it
    validation_no and test_no must be integer values from 0 to 9
    Patching: 20190423 >>>>add Module 11 and corresponding wcs'''
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        if datafilepath.find('H:/Datasets')==0:
            datafilepath=datafilepath.replace('H:/Datasets','./Datasets')
            if(os.path.exists(datafilepath)):
                print('Loading data from file: %s'%datafilepath)
            else:
                raise RuntimeError('Cannot find the data file: %s'%datafilepath)
        else:
            raise RuntimeError('Cannot find the data file: %s'%datafilepath)
    with open(datafilepath,'rb') as datafile:
        ckplus10g=pickle.load(datafile)
    if datafilepath.find('MNIST')>0 or datafilepath.find('CIFAR10')>0:
        test_no=1
        one_hot=False
    nL=len(ckplus10g)
    print('Preprocessed data is loaded from %s with %d groups.'%(datafilepath,nL))
    if Module==3:
        tl_eye_patch=[]
        tl_mouth_patch=[]
        tl_middle_patch=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], eyep=ckplus10g[i]['eye_patch'],
                                            foreheadp=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=True, loadInnerF=False)
            #    if test_no==validation_no:
            #        valid=test
            #elif i==validation_no:
            #    valid=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], eyep=ckplus10g[i]['eye_patch'],
            #                                foreheadp=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
            #                                one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
            #                                loadGeo=False, loadImg=False, loadPat=True, loadInnerF=False)
            else:
                tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
                del ckplus10g[i]['eye_patch']
                tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
                del ckplus10g[i]['mouth_patch']
                tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
                del ckplus10g[i]['middle_patch']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                #del ckplus10g[i]['imgs']
                #del ckplus10g[i]['geometry']
                #del ckplus10g[i]['inner_face']
                #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,
                                    eyep= tl_eye_patch, foreheadp= tl_middle_patch,
                                    mouthp= tl_mouth_patch,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=False, loadPat=True, loadInnerF=False)
    elif Module==1:
        tl_rescaleimgs=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(imgs=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
                                            num_Classes=cn,Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
            #    if test_no==validation_no:
            #        valid=test
            #elif i==validation_no:
            #    valid=DataSetFor3KindsDataV4(imgs=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
            #                                 num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
            #                                 loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
            else:
                tl_rescaleimgs.extend(ckplus10g[i]['imgs'])
                del ckplus10g[i]['imgs']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                #del ckplus10g[i]['geometry']
                #del ckplus10g[i]['eye_patch']
                #del ckplus10g[i]['mouth_patch']
                #del ckplus10g[i]['middle_patch']
                #del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(imgs=tl_rescaleimgs, labels=tl_labels,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
    elif Module==2:
        tl_geo=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(geometry=ckplus10g[i]['geo'], labels=ckplus10g[i]['labels'],
                                            num_Classes=cn,Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=True, loadImg=False, loadPat=False, loadInnerF=False)
            #    if test_no==validation_no:
            #        valid=test
            #elif i==validation_no:
            #    valid=DataSetFor3KindsDataV4(imgs=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
            #                                 num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
            #                                 loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
            else:
                tl_geo.extend(ckplus10g[i]['geo'])
                del ckplus10g[i]['geo']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                #del ckplus10g[i]['geometry']
                #del ckplus10g[i]['eye_patch']
                #del ckplus10g[i]['mouth_patch']
                #del ckplus10g[i]['middle_patch']
                #del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(geometry=tl_geo, labels=tl_labels,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadImg=False, loadPat=False, loadInnerF=False)
    elif Module==4:
        tl_innerf=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],cropf=ckplus10g[i]['inner_face'], 
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
            #    if test_no==validation_no:
            #        valid=test
            #elif i==validation_no:
            #    valid=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],cropf=ckplus10g[i]['inner_face'], 
            #                                one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
            #                                loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
            else:
                tl_innerf.extend( ckplus10g[i]['inner_face'])
                del ckplus10g[i]['inner_face']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                #del ckplus10g[i]['imgs']
                #del ckplus10g[i]['geometry']
                #del ckplus10g[i]['eye_patch']
                #del ckplus10g[i]['mouth_patch']
                #del ckplus10g[i]['middle_patch']
                #print(ckplus10g[i]['labels'])
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,cropf= tl_innerf, 
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
    elif Module==11:
        tl_wc=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],wcs=ckplus10g[i]['wcs'], 
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=False,
                                            loadWcs=True)
            else:
                tl_wc.extend( ckplus10g[i]['wcs'])
                del ckplus10g[i]['wcs']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,wcs= tl_wc, 
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=False,
                                    loadWcs=True)
    elif Module==23:
        tl_eye_patch=[]
        tl_mouth_patch=[]
        tl_middle_patch=[]
        tl_geo=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], eyep=ckplus10g[i]['eye_patch'],
                                            foreheadp=ckplus10g[i]['middle_patch'], mouthp=ckplus10g[i]['mouth_patch'],
                                            geometry=ckplus10g[i]['geo'],
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=True, loadPat=True)
            else:
                tl_eye_patch.extend(ckplus10g[i]['eye_patch'])
                del ckplus10g[i]['eye_patch']
                tl_mouth_patch.extend(ckplus10g[i]['mouth_patch'])
                del ckplus10g[i]['mouth_patch']
                tl_middle_patch.extend(ckplus10g[i]['middle_patch'])
                del ckplus10g[i]['middle_patch']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                tl_geo.extend(ckplus10g[i]['geo'])
                del ckplus10g[i]['geo']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,
                                    eyep= tl_eye_patch, foreheadp= tl_middle_patch,
                                    mouthp= tl_mouth_patch, geometry=tl_geo,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadPat=True)
    elif Module==211:
        tl_wcs=[]
        tl_geo=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], wcs=ckplus10g[i]['wcs'],
                                            geometry=ckplus10g[i]['geo'],
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=True, loadWcs=True)
            else:
                tl_wcs.extend(ckplus10g[i]['wcs'])
                del ckplus10g[i]['wcs']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                tl_geo.extend(ckplus10g[i]['geo'])
                del ckplus10g[i]['geo']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,
                                    wcs=tl_wcs, geometry=tl_geo,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadGeo=True, loadWcs=True)
    elif Module==411:
        tl_wcs=[]
        tl_cropf=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'], wcs=ckplus10g[i]['wcs'],
                                            cropf=ckplus10g[i]['inner_face'],
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape, 
                                            loadInnerF=True, loadWcs=True)
            else:
                tl_wcs.extend(ckplus10g[i]['wcs'])
                del ckplus10g[i]['wcs']
                tl_labels.extend(ckplus10g[i]['labels'])
                del ckplus10g[i]['labels']
                tl_cropf.extend(ckplus10g[i]['inner_face'])
                del ckplus10g[i]['inner_face']
        print('Initializing train dataset......')
        train=DataSetFor3KindsDataV4(labels=tl_labels,
                                    wcs=tl_wcs, cropf=tl_cropf,
                                    num_Classes=cn, Df=Df, one_hot=one_hot, reshape=reshape,
                                    loadInnerF=True, loadWcs=True)
    else:
        raise RuntimeError('ERROR: Unexpected Module in loadPKLData_V4')
  
    #return Datasets(train=train, test = test, validation = valid)
    return Datasets(train=train, test = test)
def loadAnonymousData(datafilepath):
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        raise FileNotFoundError('Cannot find the data file: %s'%datafilepath)
    with open(datafilepath,'rb') as datafile:
        fusiondata=pickle.load(datafile)
    return Datasets(train=DataSetForAnonymousData(X=fusiondata['trainX'], Y=fusiondata['trainY']), 
                                test=DataSetForAnonymousData(X=fusiondata['testX'], Y=fusiondata['testY']))
#############
####customized keras callbacks
class EvaCMSM(Callback):
    '''Perform validation, compute the ConfusionMatrix, and save model
    Return the confusion matrix after every epoch.
    passed in `on_epoch_end`

    # Arguments
        validation: tupe in (x, y) form, used for validation
        filepath: path to save the model
        monitor: quantity to monitor, only restrict to c_val_acc and c_val_loss.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if it's True, only save the model weights in model saving
        period: Interval (number of epochs) between checkpoints.
    '''

    def __init__(self, validation, filepath, monitor='c_val_acc', verbose=0, save_model=True,
                 save_best_only=False, save_weights_only=False, period=1, Acc_Oriented=True, tag='Tag',
                 GetffmmFlag=False, funcops=None, traindata=None, Module=None):
        super(EvaCMSM, self).__init__()
        if monitor not in ['c_val_acc','c_val_loss']:
            warnings.warn('EvaCMSM monitor %s is unknown, fall back to c_val_acc.'%(monitor), RuntimeWarning)
            monitor = 'c_val_acc'
        if monitor=='c_val_loss' and Acc_Oriented:
            warnings.warn('EvaCMSM unexpected setting between monitor and Acc_Oriented: %s %s.'%(monitor, str(Acc_Oriented)), RuntimeWarning)
        elif monitor=='c_val_acc' and not Acc_Oriented:
            warnings.warn('EvaCMSM unexpected setting between monitor and Acc_Oriented: %s %s.'%(monitor, str(Acc_Oriented)), RuntimeWarning)
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.x_val, self.y_val=validation
        self.best_cm=None
        self.best_epoch=None
        self.best=[.0, .0]
        self.tag=tag
        self.save_model=save_model
        self.ffmmflag=GetffmmFlag
        self.M=Module
        if GetffmmFlag:
            if funcops is None or traindata is None:
                raise RuntimeError('\nERROR\nERROR\nUnexpected Error in EvaCMSM(Callback) initialization.\n\n')
            self.getC1=funcops
            self.best_predy=None
            self.x_train=traindata
            self.testffmm=None
            self.trainffmm=None
            
        
        if 'c_val_acc' == self.monitor:
            self.monitor_op = np.greater
            self.monitor_op2 = np.less
            self.best[0] = -np.Inf
            self.currentIndx=1#index for matching the outputs of keras model evaluate() function and the accuracy in self.best list
        else:
            self.monitor_op = np.less
            self.monitor_op2 = np.greater
            self.best[0] = np.Inf
            self.currentIndx=0#index for matching the outputs of keras model evaluate() function and the accuracy in self.best list
        self.best[1]=-self.best[0]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            hr = self.model.evaluate(x=self.x_val, y=self.y_val, batch_size=None)
            logs['c_val_acc'] = hr[1]
            logs['c_val_loss'] = hr[0]
            if self.save_best_only:
                current = hr[self.currentIndx]
                current2 = hr[1-self.currentIndx]
                if self.monitor_op(current, self.best[0]):
                    if self.monitor_op2(current2, self.best[1]):
                        self.best[1]=current2
                    if self.verbose > 0:
                        print('\nExpression: %s Epoch %05d: %s improved from %0.5f to %0.5f'%(self.tag, epoch + 1, self.monitor, self.best[0], current))
                    self.best[0] = current
                    pred_y=self.model.predict(self.x_val, verbose=0)
                    if self.ffmmflag:
                        if self.M==23:
                            self.best_predy=pred_y
                        else:
                            self.best_predy=pred_y[:,0]
                        self.testffmm=self.getC1([self.x_val, 0])[0]
                        self.testffmm=np.concatenate([self.testffmm, pred_y], axis=-1)
                        trainy=self.model.predict(np.array(self.x_train), verbose=0)
                        self.trainffmm=self.getC1([self.x_train, 0])[0]
                        self.trainffmm=np.concatenate([self.trainffmm, trainy], axis=-1)
                    truel=np.argmax(self.y_val, axis=1)
                    predl=np.argmax(pred_y, axis=1)
                    self.best_cm=confusion_matrix(y_true=truel, y_pred=predl)
                    self.best_epoch=epoch
                    if self.save_weights_only and self.save_model:
                        self.model.save_weights(self.filepath, overwrite=True)
                    elif self.save_model:
                        self.model.save(self.filepath, overwrite=True)
                else:
                    if current==self.best[0] and self.monitor_op2(current2, self.best[1]):###
                        self.best[1]=current2
                        pred_y=self.model.predict(self.x_val, verbose=0)
                        if self.ffmmflag:
                            self.best_predy=pred_y[:,0]
                            self.testffmm=self.getC1([self.x_val, 0])[0]
                            self.testffmm=np.concatenate([self.testffmm, pred_y], axis=-1)
                            trainy=self.model.predict(np.array(self.x_train), verbose=0)
                            self.trainffmm=self.getC1([self.x_train, 0])[0]
                            self.trainffmm=np.concatenate([self.trainffmm, trainy], axis=-1)
                        truel=np.argmax(self.y_val, axis=1)
                        predl=np.argmax(pred_y, axis=1)
                        self.best_cm=confusion_matrix(y_true=truel, y_pred=predl)
                        self.best_epoch=epoch
                        if self.save_weights_only and self.save_model:
                            self.model.save_weights(self.filepath, overwrite=True)
                        elif self.save_model:
                            self.model.save(self.filepath, overwrite=True)
                    if self.verbose > 0:
                        print('\nExpression: %s Epoch %05d: %s did not improve from %0.5f'%(self.tag, epoch + 1, self.monitor, self.best[0]))
                if self.ffmmflag:
                    logs['predy']=self.best_predy
                    logs['testffmm']=self.testffmm
                    logs['trainffmm']=self.trainffmm
                    #print(self.testffmm.shape, self.trainffmm.shape)
                    #exit()
                logs['cm']=self.best_cm
                logs['epoch']=self.best_epoch
                logs['b_v_loss']=self.best[self.currentIndx]
                logs['b_v_acc']=self.best[1-self.currentIndx]
            else:
                if self.verbose > 0:
                    print('\nExpression: %s Epoch %05d ' % (self.tag, epoch + 1))
        return
###############


def run(GPU_Device_ID, 
        DataSet,ValidID,TestID, 
        NetworkType, runs, Module=3
        ,cLR=0.0001,batchSize=30, ME=140, reshape=False):
    '''
    DataSet: 1-6, see Dataset_Dictionary above
    ValidID==TestID: 0-9 or 0-7
    NetworkType: 1-3 correspond to ARCFN BRCFN CRCFN
    runs: 0-9, runs for each data selection
    for example: call it by
        run(0, 1, 0, 0, 1, 0)
        run(0, 3, 5, 5, 3, 8)
    '''
    try:
        initialize_dirs()
        '''GPU Option---------------------------------------------------------------------------------------------
        Determine which GPU is going to be used
        ------------------------------------------------------------------------------------------------------------'''
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==GPU_Device_ID) or (1==GPU_Device_ID):
            os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_Device_ID)
            errorlog='./logs/errors_gpu'+str(GPU_Device_ID)+'.txt'
            templog='./logs/templogs_newSC_gpu'+str(GPU_Device_ID)+'_M'+str(Module)+'_D'+str(DataSet)+'.txt'
        else:
            raise RuntimeError('Usage: python finetune.py <GPUID> <Module> <NetworkType>\nGPUID must be 0 or 1\nModule must be 1, 2, or 3\nNetworkType must be 0, 1, 2, 3')
        '''GPU Option ENDS---------------------------------------------------------------------------------------'''
        if DataSet%10==3:
            cn=7
        elif DataSet%10==7:
            cn=10
        else:
            cn=6
        mini_loss=10000
        loss_a=LOSS_ANA()
        file_record=None
        t1=time.time()
        timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))
        logprefix='./logs/'
        model_save_path=''

        m1shape= [None, 128, 128, 1]
        global Mini_Epochs
        #if not ME==300:
        #    Mini_Epochs=ME
        #
        #
        #
        Mini_Epochs = 140
        '''Input Data-------------------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        #
        ##data set loading
        #
        dfile=Dataset_Dictionary.get(DataSet, None)
        if dfile is None:
            raise RuntimeError('\nERROR: check the DataSet again. It must be one of %s'%(str(Dataset_Dictionary.keys())))
        logprefix='./logs/D%d_gpu'%(DataSet)
        
        data = loaddata_v4(dfile, ValidID, TestID, Module=Module, cn=cn)

        #
        lrstep=int(data.train.num_examples/batchSize*times)
        print('\nlearning rate decay steps: %d'%lrstep)
        #
        tt=time.time()
        
        log='%s%d_M%d_D%d_N%d_noPretrain_newSCV3_%depochs_upDLoLo.txt'%(logprefix,GPU_Device_ID,Module,DataSet,NetworkType,Mini_Epochs)

        print('Time used for loading data: %fs'%(tt-t1))

        if os.path.exists('J:/Models/saves/'):
            model_save_path='J:/Models/saves/M%d/D%d/N%d/'%(Module, DataSet, NetworkType)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
        else:
            model_save_path='./saves/M%d/D%d/N%d/'%(Module, DataSet, NetworkType)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
        model_save_path='%sD%d_M%d_N%d_T%d_V%d_R%d%s.ckpt'%(model_save_path,DataSet,Module,NetworkType,TestID,ValidID
                            ,runs,timestamp)

        '''Input Data Ends-----------------------------------------------------------------------------------------'''

        if Module==3:
            stcmwvlilttv=0.0002154#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value

            '''MODULE3---------------------------------------------------------------------------------------------------- 
            Options for the RCFN
            -------------------------------------------------------------------------------------------------------------'''
            print('RCFN: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            
            tr,tc,tk=data.train.eyep[0].shape
            eye_p_shape=[None, tr, tc, tk]
            tr, tc, tk=data.train.foreheadp[0].shape
            midd_p_shape=[None, tr,tc,tk]
            tr, tc, tk=data.train.mouthp[0].shape
            mou_p_shape=[None, tr,tc,tk]
                
            print('EYE shape: %s'%(str(eye_p_shape)))
            print('Forehead shape: %s'%(str(midd_p_shape)))
            print('Mouth shape: %s'%(str(mou_p_shape)))
            g=tf.Graph()
            with g.as_default():
                eye_p = tf.placeholder(tf.float32, eye_p_shape)
                midd_p = tf.placeholder(tf.float32, midd_p_shape)
                mou_p = tf.placeholder(tf.float32, mou_p_shape)
                #Holder for labels in a batch size of batch_size, number of labels are to be determined

            ferm=FERFN([eye_p, midd_p, mou_p], NetworkType, cn, g, Runs=runs, initialLearningRate=cLR, learningDecayRate=lr_drate,
                        learningDecayStep=lrstep)

            with tf.Session(graph=ferm.graph) as sess:
                ferm.session=sess
                sess.run(tf.global_variables_initializer())
                
                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                clr=cLR
                for i in range(iters):
                    afc=[]
                    batch=data.train.next_batch(batchSize, shuffle=False)
                    if Summary:
                        tloss, _, clr, trainSum=ferm.train(sess, [ferm._T_loss, ferm._T_train_op, ferm._T_learningRate, ferm._T_SummaryOPS], [eye_p, midd_p, mou_p, ferm._T_label], [batch[2], batch[3], batch[4], batch[5]])
                        ferm.Train_addSummary(trainSum, i)
                    else:
                        tloss, _, clr=ferm.train(sess, [ferm._T_loss, ferm._T_train_op, ferm._T_learningRate], [eye_p, midd_p, mou_p, ferm._T_label], [batch[2], batch[3], batch[4], batch[5]])
                    
                    if tloss<mini_loss:
                        mini_loss=tloss
                    v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet_3NI(cn, sess, ferm._T_accuracy, ferm._T_sum_test, ferm._T_loss, ferm._T_logits,
                                                                                  eye_p, data.test.eyep, midd_p, data.test.foreheadp,
                                                                                  mou_p, data.test.mouthp, ferm._T_label, data.test.labels, afc=afc)
                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, oaa)
                    tt=time.time()
                    print('LR:%.8f Ite:%05d Bs:%03d Epo:%03d Los:%.8f mLo:%0f>>mVL:%.8f VL:%.8f VA:%f OAA:%f Best:%f T:%fs'%
                          (clr,i,batchSize,data.train.epochs_completed, tloss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, oaa, loss_a.highestAcc, (tt-t1)))
                    if laflag:
                        file_record = logfile(file_record, runs=runs, OAA=oaa, afc=afc, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=tloss, train_min_loss=mini_loss, TA=v_accuracy, TC=(tt-t1),ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=batchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile, lossa=loss_a)
                        if loss_a.minimun_loss < stcmwvlilttv and SaveModel:
                            print(loss.a.minimun_loss)
                            saver.save(sess=sess, save_path=model_save_path)
     
                '''MODULE3 ENDS---------------------------------------------------------------------------------------------'''

        
        elif Module==1:
            stcmwvlilttv=0.002154#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value

            '''MODULE1---------------------------------------------------------------------------------------------------- 
            Options for the RCFN
            -------------------------------------------------------------------------------------------------------------'''
            print('RCFN: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            
            tr,tc,tk=data.train.imgs[0].shape
            whole_shape=[None, tr, tc, tk]
                
            print('face shape: %s'%(str(whole_shape)))
            g=tf.Graph()
            with g.as_default():
                whole_p = tf.placeholder(tf.float32, whole_shape)
                #Holder for labels in a batch size of batch_size, number of labels are to be determined

            ferm=FERFN([whole_p], NetworkType, cn, g, Runs=runs, initialLearningRate=cLR, learningDecayRate=lr_drate,
                        learningDecayStep=lrstep)

            with tf.Session(graph=ferm.graph) as sess:
                ferm.session=sess
                sess.run(tf.global_variables_initializer())
                
                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                clr=cLR
                for i in range(iters):
                    afc=[]
                    batch=data.train.next_batch(batchSize, shuffle=False)
                    if Summary:
                        tloss, _, clr, trainSum=ferm.train(sess, [ferm._T_loss, ferm._T_train_op, ferm._T_learningRate, ferm._T_SummaryOPS], [whole_p, ferm._T_label], [batch[0], batch[5]])
                        ferm.Train_addSummary(trainSum, i)
                    else:
                        tloss, _, clr=ferm.train(sess, [ferm._T_loss, ferm._T_train_op, ferm._T_learningRate],  [whole_p, ferm._T_label], [batch[0], batch[5]])
                    
                    if tloss<mini_loss:
                        mini_loss=tloss
                    v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet(cn, sess, ferm._T_accuracy, ferm._T_sum_test, ferm._T_loss, ferm._T_logits,
                                                                                  whole_p, data.test.imgs, 
                                                                                  ferm._T_label, data.test.labels, afc=afc)
                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed,oaa)
                    tt=time.time()
                    print('LR:%.8f Ite:%05d Bs:%03d Epo:%03d Los:%.8f mLo:%0f>>mVL:%.8f VL:%.8f VA:%f OAA:%f Best:%f T:%fs'%
                          (clr,i,batchSize,data.train.epochs_completed, tloss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, oaa, loss_a.highestAcc, (tt-t1)))
                    if laflag:
                        file_record = logfile(file_record, runs=runs, OAA=oaa, afc=afc, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=tloss, train_min_loss=mini_loss, TA=v_accuracy, TC=(tt-t1),ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=batchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile, lossa=loss_a)
                        if loss_a.minimun_loss < stcmwvlilttv and SaveModel:
                            saver.save(sess=sess, save_path=model_save_path)
     
                '''MODULE1 ENDS---------------------------------------------------------------------------------------------'''

        elif Module==4:
            stcmwvlilttv=0.002154#value need to be determined. save_the_current_model_when_validation_loss_is_less_than_this_value

            '''MODULE4---------------------------------------------------------------------------------------------------- 
            Options for the RCFN
            -------------------------------------------------------------------------------------------------------------'''
            print('RCFN: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            
            tr,tc,tk=data.train._cropf[0].shape
            crop_shape=[None, tr, tc, tk]
                
            print('face shape: %s'%(str(crop_shape)))
            g=tf.Graph()
            with g.as_default():
                crop_p = tf.placeholder(tf.float32, crop_shape)
                #Holder for labels in a batch size of batch_size, number of labels are to be determined

            ferm=FERFN([crop_p], NetworkType, cn, g, Runs=runs, initialLearningRate=cLR, learningDecayRate=lr_drate,
                        learningDecayStep=lrstep)

            with tf.Session(graph=ferm.graph) as sess:
                ferm.session=sess
                sess.run(tf.global_variables_initializer())
                
                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                clr=cLR
                for i in range(iters):
                    afc=[]
                    batch=data.train.next_batch(batchSize, shuffle=False)
                    if Summary:
                        tloss, _, clr, trainSum=ferm.train(sess, [ferm._T_loss, ferm._T_train_op, ferm._T_learningRate, ferm._T_SummaryOPS], [crop_p, ferm._T_label], [batch[6], batch[5]])
                        ferm.Train_addSummary(trainSum, i)
                    else:
                        tloss, _, clr=ferm.train(sess, [ferm._T_loss, ferm._T_train_op, ferm._T_learningRate],  [crop_p, ferm._T_label], [batch[6], batch[5]])
                    
                    if tloss<mini_loss:
                        mini_loss=tloss
                    v_accuracy, valid_loss, oaa, confu_mat = Valid_on_TestSet(cn, sess, ferm._T_accuracy, ferm._T_sum_test, ferm._T_loss, ferm._T_logits,
                                                                                  crop_p, data.test._cropf, 
                                                                                  ferm._T_label, data.test.labels, afc=afc)
                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed,oaa)
                    tt=time.time()
                    print('LR:%.8f Ite:%05d Bs:%03d Epo:%03d Los:%.8f mLo:%0f>>mVL:%.8f VL:%.8f VA:%f OAA:%f Best:%f T:%fs'%
                          (clr,i,batchSize,data.train.epochs_completed, tloss, mini_loss, loss_a.minimun_loss, valid_loss, v_accuracy, oaa, loss_a.highestAcc, (tt-t1)))
                    if laflag:
                        file_record = logfile(file_record, runs=runs, OAA=oaa, afc=afc, valid_loss=valid_loss, valid_min_loss=loss_a.minimun_loss, 
                            final_train_loss=tloss, train_min_loss=mini_loss, TA=v_accuracy, TC=(tt-t1),ILR=cLR, FLR=clr, LS=lrstep, ites=i,
                            Epo=data.train.epochs_completed, cBS=batchSize, iBS=batchSize,
                            input=sys.argv, CM=confu_mat, T=time.localtime(tt), df=dfile, lossa=loss_a)
                        if loss_a.minimun_loss < stcmwvlilttv and SaveModel:
                            saver.save(sess=sess, save_path=model_save_path)
                '''MODULE4 ENDS---------------------------------------------------------------------------------------------'''
        if Acc_Oriented:#####
            newmodelname=model_save_path.replace('.ckpt','_ACC%s_.ckpt'%(str(loss_a.highestAcc)))
        else:
            newmodelname=model_save_path.replace('.ckpt','_MiniLoss%s_.ckpt'%(str(loss_a.minimun_loss)))
        if os.path.exists(model_save_path+'.data-00000-of-00001'):
            os.rename((model_save_path+'.data-00000-of-00001'),(newmodelname+'.data-00000-of-00001'))
            os.rename((model_save_path+'.index'),(newmodelname+'.index'))
            os.rename((model_save_path+'.meta'),(newmodelname+'.meta'))
        optimizerName=str(type(ferm._T_optm).__name__)
        tt=time.time()
        log=log.replace('.txt',('_'+optimizerName+'.txt'))
        filelog=open(log,'a')

        print(log.split('.txt')[0])
        losslog=log.split('.txt')[0]+'_Runs%d_%d_%d'%(runs, ValidID, TestID)+'.validationlosslist'
        losslog=losslog.replace('./logs/','./logs/VL/')
        loss_a.outputlosslist(losslog)

        filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\n'%(file_record, (tt-t1), optimizerName, losslog))
        print(log)
        filelog.close()
    except:
        try:
            tt=time.time()
            log=log.replace('.txt',('_'+optimizerName+'.txt'))
            filelog=open(log,'a')

            print(log.split('.txt')[0])
            losslog=log.split('.txt')[0]+'_Runs%d_%d_%d'%(runs, ValidID, TestID)+'.validationlosslist'
            losslog=losslog.replace('./logs/','./logs/VL/')
            loss_a.outputlosslist(losslog)

            filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\n'%(file_record, (tt-t1), optimizerName, losslog))
            print('\n\n>>>>>> Saving current run info after it crrupted or interrupted.\n\n')
            print(log)
            filelog.close()
            print('>>>>>> Current run info has been saved after it crrupted or interrupted.\n\n')
        except:
            print('ERROR: Fail to save current run info. after it crrupted')
        ferror=open(errorlog,'w')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()
    return

##category 2 has better performance than category 3 (NewC=2 is better than NewC=3)
def runKerasV(GPU_Device_ID, 
        DataSet,TestID, NewC, 
        NetworkType, runs, Module, Mfit=False, batchNorm=True
        ,cLR=0.0001,batchSize=50, FusionNType=0, FusionMfit=True, FLdata=False, FUSION=False):
    '''
    DataSet: 11-16, see Dataset_Dictionary above
    ValidID==TestID: 0-9 or 0-7
    NetworkType: 11-
    runs: 0-9, runs for each data selection
    for example: call it by
        run(0, 1, 0, 0, 1, 0)
        run(0, 3, 5, 5, 3, 8)
    '''
    try:
        global Mini_Epochs, SaveModel, times, test_bat
        ovt_drate=getOVTDR(Module)
        Mini_Epochs = getOVTME(Module, Mini_Epochs, DataSet)
        times=getOVTTimes(Module, times, DataSet)
        batchSize=getOVTBS(Module, batchSize, DataSet)
        SaveModel=getOVTSM(Module, SaveModel)
        sf=False#shuffle flag
        if sf:
            mess='\nAlert! Alert! Alert! Using shuffle strategy in training.'
            for i in range(10):
                mess=mess+mess
            print(mess)
        initialize_dirs(ovt=True)
        '''GPU Option---------------------------------------------------------------------------------------------
        Determine which GPU is going to be used
        ------------------------------------------------------------------------------------------------------------'''
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==GPU_Device_ID) or (1==GPU_Device_ID):
            os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_Device_ID)
            errorlog='./logs/ovt/ovt_errors_gpu'+str(GPU_Device_ID)+'.txt'
            templog='./logs/ovt/ovt_templogs_newSC_gpu'+str(GPU_Device_ID)+'_M'+str(Module)+'_D'+str(DataSet)+'.txt'
        else:
            raise RuntimeError('Usage: python finetune.py <GPUID> <Module> <NetworkType>\nGPUID must be 0 or 1\nModule must be 1, 2, or 3\nNetworkType must be 0, 1, 2, 3')
        
        '''GPU Option ENDS---------------------------------------------------------------------------------------'''
        if DataSet%10 == 3:
            cn=7
        elif DataSet%10 == 8 or DataSet%10 == 7:#modified in 20191003
            cn=7
            if TestID>0:
                raise RuntimeWarning('Unexpected TestID Setting.')
        elif DataSet%10 == 9 or DataSet==690 or DataSet==4090 or DataSet==4190:#modified in 20191005
            cn=8
            if TestID>0:
                raise RuntimeWarning('Unexpected TestID Setting.')
        #elif DataSet%10==7:#modified in 20191003
        #    cn=10
        #    batchSize=1200
        #    test_bat=2000
        #    TestNumLimit = 2000
        #    #if not Mfit
        #    Mini_Epochs=70
        else:
            cn=6
        if Module==2:
            batchSize=30
            test_bat=3000
            if DataSet>150 and DataSet<160:
                times=15
                #batchSize=90
                Mini_Epochs=60
        if (DataSet>120 and DataSet<150) or (DataSet>220 and DataSet<340):
            batchSize=120
            print('batch size set to: %d'%batchSize)
        t0=time.time()
        if Module==11 and (NewC==2 or NewC==3):
            batchSize=120
            if os.path.exists('J:/Models/ovtsaves/'):
                model_save_path_prefix='J:/Models/ovtsaves/M%d/D%d/N%d/'%(Module, DataSet, NetworkType)
                if not os.path.exists(model_save_path_prefix):
                    os.makedirs(model_save_path_prefix)
            else:
                model_save_path_prefix='./saves/ovt/M%d/D%d/N%d/'%(Module, DataSet, NetworkType)
                if not os.path.exists(model_save_path_prefix):
                    os.makedirs(model_save_path_prefix)
            logprefix='./logs/ovt/D%d_ovt_gpu'%(DataSet)
            if ForVisFea:
                logprefix=logprefix.replace('_ovt_','_ovt_ForVisFea_')
            if not FLdata:
                lrstep=None
                SaveModel=False
                t1=time.time()
                timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))
                if Mfit:
                    ftag='_Modelfit'
                else:
                    ftag='_train_on_batch'
                FeaturesPath='./MMFeatures/OVT_D%d_M%d_N%d_T%d_R%d%s%s_lasttwolayerOutputs.pkl'%(DataSet,Module,NetworkType,TestID
                                            ,runs,timestamp,ftag)
                
                print(logprefix)
                mafa=np.zeros((2, cn))
                Lcount=0.0
                labels={}
                ffmms={}
                ffmms['testffmms']=[]#features from multi-models
                ffmms['trainffmms']=[]
                modelnamelist=[]
                for ei in range(cn):
                    mini_loss=10000
                    loss_a=LOSS_ANA()
                    file_record=None
                    model_save_path=''
                    '''Input Data-------------------------------------------------------------------------------------------------
                    -------------------------------------------------------------------------------------------------------------'''
                    #
                    ##data set loading
                    #
                    dfile=Dataset_Dictionary.get(DataSet, None)
                    if dfile is None:
                        raise RuntimeError('\nERROR: check the DataSet again. It must be one of %s'%(str(Dataset_Dictionary.keys())))
        
                    data = loaddata_v4(dfile, TestID, TestID, Module=Module, cn=NewC)
                    #print(data.test.labels)#check the original labels
                    Lcount=data.test.num_examples
                    if labels.get('test', None) is None:
                        labels['test']=dense_to_one_hot(data.test.labels, cn)
                        labels['train']=dense_to_one_hot(data.train.labels, cn)
                        labels['predict']=np.zeros((Lcount, cn))
                    data.train.reset(labels=transferLabels(data.train.labels, NewC, ei))
                    data.test.reset(labels=transferLabels(data.test.labels, NewC, ei))

                    tt=time.time()
                    if Mfit:
                        tag=('_noPretrain_keras_model_fit_%depochs_%sV2.txt'%(Mini_Epochs, Label_Dictionary.get(ei)))
                    else:
                        lrstep=int(data.train.num_examples/batchSize*times)
                        print('\nlearning rate decay steps: %d'%lrstep)
                        tag=('_noPretrain_newSCV3_%depochs_upDLoLo_%sV2.txt'%(Mini_Epochs, Label_Dictionary.get(ei)))
                    log='%s%d_M%d_D%d_N%d_C%d%s'%(logprefix, GPU_Device_ID, Module, DataSet, NetworkType, NewC, tag)
            
                    print('Time used for loading data: %fs'%(tt-t1))
                    
                    model_save_path='%sOVT_D%d_M%d_N%d_T%d_R%d%s_GPU%d%s.h5'%(model_save_path_prefix,DataSet,Module,NetworkType,TestID
                                            ,runs,timestamp,GPU_Device_ID,tag.replace('.txt',''))
                    #print(log, model_save_path)#check the log file path and model save path
                    '''Input Data Ends-----------------------------------------------------------------------------------------'''

                    '''MODULE11---------------------------------------------------------------------------------------------------- 
                    Options for the 
                    -------------------------------------------------------------------------------------------------------------'''
                    print('WCPCN: %s'%(NetworkType))
                    '''Here begins the implementation logic-------------------------------------------------------------------
                    -------------------------------------------------------------------------------------------------------------'''
            
                    data_shape=data.train._wcs[0].shape
                    print('data shape: %s'%(str(data_shape)))

                    ferm=FERN.GetNetworkKV(data_shape, NetworkType, Label_Dictionary.get(ei), NewC, batchNorm, DataSet)
                    plot_model(ferm, to_file='./modelfigures/M%dN%dD%dC%dmodel_params-%d.png'%(Module,NetworkType,DataSet,NewC,ferm.count_params()), show_shapes=True)
                    getConcatenate1=K.function([ferm.layers[0].input, K.learning_phase()],[ferm.layers[-2].output])#output ndarray

                    ####For the KDEF tests, the best option is use default Adam without decay lr, the second best is default Adadelta.
                    #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)#default lr 0.002 runs0-runs7, 0.001 for runs8 and runs9
                    #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)# default lr 1.0 rho0.95 for runs10, lr 1.0 rho 0.8 for runs11
                    op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                    cLR=0.001
                    ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

                    if Mfit:
                        print(Label_Dictionary.get(ei))
                        reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                        ccm=EvaCMSM(validation=(np.array(data.test.wcs), np.array(data.test.labels)), filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, 
                                    save_best_only=True, Acc_Oriented=Acc_Oriented, tag=Label_Dictionary.get(ei), GetffmmFlag=True, funcops=getConcatenate1, traindata=data.train.wcs)
                        cbl=[reduce_lr, ccm]
                        if Summary:
                            history=ferm.fit(x=np.array(data.train.wcs), y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                        else:
                            history=ferm.fit(x=np.array(data.train.wcs), y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                    
                        #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                        #exit()
                        cm=history.history['cm'][-1]
                        mini_v_loss=history.history['b_v_loss'][-1]
                        be=history.history['epoch'][-1]
                        floss=history.history['loss'][be-1]
                        hacc=history.history['b_v_acc'][-1]
                        predy=history.history['predy'][-1]
                        for indexL in range(Lcount):
                            labels['predict'][indexL][ei]=predy[indexL]
                        temtestffmm=history.history['testffmm'][-1]
                        temtrainffmm=history.history['trainffmm'][-1]
                        loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                        clr=history.history['lr'][-1]
                        tt=time.time()
                        afc=[]
                        cts=[]
                        oaa=overAllAccuracy(cm, afc, cts)
                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                    final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                    Epo=be, cBS=batchSize,iBS=batchSize, BN=batchNorm,
                                    input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                        mafa[0][ei]=afc[0]
                        mafa[1][ei]=cts[0]
                    else:
                        iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                        clr=K.get_value(ferm.optimizer.lr)
                        for i in range(iters):
                            if (i+1)%lrstep==0 and ovt_drate<1:
                                clr=clr*ovt_drate
                                K.set_value(ferm.optimizer.lr, clr)
                            batch=data.train.next_batch(batchSize, shuffle=False)
                            if Summary:
                                tloss, tac=ferm.train_on_batch(x=np.array(batch[7]), y=np.array(batch[5]))
                            else:
                                tloss, tac=ferm.train_on_batch(x=np.array(batch[7]), y=np.array(batch[5]))
                            #print(ferm.metrics_names)
                            if tloss<mini_loss:
                                mini_loss=tloss
                            valid_loss, ta = ferm.evaluate(x=np.array(data.test.wcs), y=np.array(data.test.labels), batch_size=test_bat)
                            laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                            tt=time.time()
                            print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs %s'%
                                    (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1), Label_Dictionary.get(ei)))
                            if laflag:
                                predy=ferm.predict(x=np.array(data.test.wcs))
                                temtestffmm=getConcatenate1([data.test.wcs, 0])[0]
                                #print(type(predy), predy.shape)
                                #print(type(temtestffmm),temtestffmm.shape)
                                temtestffmm=np.concatenate([temtestffmm, predy],axis=-1)
                                #print(type(temtestffmm),temtestffmm.shape)
                                #exit()
                                trainy=ferm.predict(x=np.array(data.train.wcs))
                                temtrainffmm=getConcatenate1([data.train.wcs, 0])[0]
                                temtrainffmm=np.concatenate([temtrainffmm, trainy],axis=-1)
                                for indexL in range(Lcount):
                                    labels['predict'][indexL][ei]=predy[indexL][0]
                                #print('\n\n\n')
                                ##plot_model(ferm, to_file='M%dN%dD%dC%dmodel.png'%(Module,NetworkType,DataSet,NewC), show_shapes=True)
                                #print('Model parameters count: %d'%(ferm.count_params()))
                                #print(ferm.layers[22].name, ferm.layers[-2].name)
                                ##print(type(temtestffmm))
                                ##print(labels['predict'])
                                #print('\n\n\n')
                                #exit()
                                predl=np.argmax(predy, axis=1)
                                truel=np.argmax(data.test.labels, axis=1)
                                cm=confusion_matrix(y_true=truel, y_pred=predl)
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                be=data.train.epochs_completed
                                hacc=ta
                                mini_v_loss=valid_loss
                                mafa[0][ei]=afc[0]
                                mafa[1][ei]=cts[0]
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                    final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                    Epo=be, cBS=batchSize, BN=batchNorm,
                                    input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                if SaveModel:
                                    ferm.save(model_save_path)
                                    print('Model saved!')
                    if Acc_Oriented:
                        newmodelname=model_save_path.replace('.h5','_Epoch%03d_ACC%f_MiniLoss%f_.h5'%(be, hacc, mini_v_loss))
                    else:
                        newmodelname=model_save_path.replace('.h5','_Epoch%03d_MiniLoss%f_ACC%f_.h5'%(be, mini_v_loss, hacc))
                    if os.path.exists(model_save_path):
                        os.rename((model_save_path),newmodelname)
                    ffmms['testffmms'].append(temtestffmm)
                    ffmms['trainffmms'].append(temtrainffmm)
                    modelnamelist.append(newmodelname)

                    optimizerName=str(type(op).__name__)
                    tt=time.time()
                    log=log.replace('.txt',('_'+optimizerName+'.txt'))
                    filelog=open(log,'a')

                    print(log.split('.txt')[0])
                    losslog=log.split('.txt')[0]+'_Runs%d_%d'%(runs, TestID)+'.validationlosslist'
                    losslog=losslog.replace('./logs/ovt/','./logs/ovt/VL/')
                    loss_a.outputlosslist(losslog)

                    filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\tParameters:%d\n'%(file_record, (tt-t1), optimizerName, losslog,ferm.count_params()))
                    print(log)
                    filelog.close()
                    '''Expression Loop ENDS---------------------------------------------------------------------------------------------'''
                #######linear voting using the maximum likelyhood probobilities of expressions
                aepredl=np.argmax(labels['predict'], axis=1)
                aetruel=np.argmax(labels['test'], axis=1)
                aecm=confusion_matrix(y_true=aetruel, y_pred=aepredl)
                aeoaa=balanced_accuracy_score(y_true=aetruel, y_pred=aepredl)
                aeacc=accuracy_score(y_true=aetruel, y_pred=aepredl)
                #######
                te=time.time()
                print('\nTotal time for all %d expressions: %fs'%(cn, (te-t0)))
                log='%s%d_M%d_D%d_N%d_C%d%s_0_MfAEV3_%s.txt'%(logprefix,GPU_Device_ID,Module,DataSet,NetworkType, NewC,tag[0:tag.rfind('_')],optimizerName)
                print('Summary file: %s'%log)
                mafam=np.sum(mafa, axis=1)/cn
                TargetAC=float(np.sum(mafa[1]))/float(Lcount)
                print('TargetAC: %f\tTargetExpACC: %f\tBAC: %f\tAC: %f'%(TargetAC, mafam[0], aeoaa, aeacc))
                if NewC>3:
                    paramsc=ferm.count_params()
                else:
                    paramsc=ferm.count_params()*cn
                logOVTFileKeras(log, runs, OAA=aeoaa, TA=aeacc, TargetACC=mafam[0], TAC=TargetAC, BN=batchNorm, CM=aecm, TC=(te-t0), LS=lrstep, ILR=cLR, FLR=clr, iBS=batchSize, input=sys.argv, T=time.localtime(te), df=dfile, params=paramsc)
                fusiondata={}
                fusiondata['trainX']=np.stack(ffmms['trainffmms'],axis=-1)
                fusiondata['trainY']=labels['train']
                fusiondata['testX']=np.stack(ffmms['testffmms'],axis=-1)
                fusiondata['testY']=labels['test']
                with open(FeaturesPath,'wb') as fpkl:
                    pickle.dump(fusiondata, fpkl, 4)
                    print('Features saved in %s'%(FeaturesPath))
                with open(FeaturesPath.replace('.pkl','.info'),'w') as finfo:
                    for v in modelnamelist:
                        finfo.write('%s\n'%(v))
                fdata=Datasets(train=DataSetForAnonymousData(X=fusiondata['trainX'], Y=fusiondata['trainY']), 
                                test=DataSetForAnonymousData(X=fusiondata['testX'], Y=fusiondata['testY']))
            else:
                t1=time.time()
                timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))
                dfile=getSelectedDataFile(DataSet, NetworkType, TestID)
                fdata=loadAnonymousData(dfile)
                file_record=None
            if FUSION:
                #if runs%3==0:
                #    SaveModel=True
                #else:
                #    SaveModel=False
                SaveModel=False
                if Mfit:
                    tag=('_noPretrain_keras_model_fit_%depochs_FUSIONHEAD_V2glorot'%(Mini_Epochs))
                else:
                    lrstep=int(fdata.train.num_examples/batchSize*times)
                    tag=('_noPretrain_nSCV3_%depochs_FUSIONHEAD_V2glorot'%(Mini_Epochs))
                if FusionMfit:
                    model_save_path='%sOVT_M%d_D%d_N%d_FN%d_T%d_FMfit%s_R%d_FLdata%s%s_GPU%d%s.h5'%(model_save_path_prefix,Module,DataSet,NetworkType,FusionNType,TestID
                                            ,str(FusionMfit),runs,str(FLdata),timestamp,GPU_Device_ID,tag)
                else:
                    model_save_path='%sOVT_M%d_D%d_N%d_FN%d_T%d_FMfit%sBS%d_R%d_FLdata%s%s_GPU%d%s.h5'%(model_save_path_prefix,Module,DataSet,NetworkType,FusionNType,TestID
                                            ,str(FusionMfit), batchSize,runs,str(FLdata),timestamp,GPU_Device_ID,tag)
                loss_f=LOSS_ANA()
                fusionGrap=tf.Graph()
                with fusionGrap.as_default():
                    fdata_shape=fdata.train.X[0].shape
                    ferfm=FERN.GetFusionNetworkKV(fdata_shape, FusionNType, cn, DataSet)
                    plot_model(ferfm, to_file='./modelfigures/M%dN%dFusionN%dD%dmodel_params-%d.png'%(Module,NetworkType,FusionNType,DataSet,ferfm.count_params()), show_shapes=True)
                    
                    op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                    cLR=0.001
                    ferfm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                    optimizerName=str(type(op).__name__)
                    mini_loss=np.inf
                    with tf.Session(graph=fusionGrap) as sess:
                        ferfm.session=sess
                        sess.run(tf.global_variables_initializer())
                        if FusionMfit:
                            reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                            ccm=EvaCMSM(validation=(np.array(fdata.test.X), np.array(fdata.test.Y)), filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='FusionHead%d'%(FusionNType))
                            cbl=[reduce_lr, ccm]
                            if Summary:
                                history=ferfm.fit(x=np.array(fdata.train.X), y=np.array(fdata.train.Y), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                            else:
                                history=ferfm.fit(x=np.array(fdata.train.X), y=np.array(fdata.train.Y), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                            cm=history.history['cm'][-1]
                            mini_v_loss=history.history['b_v_loss'][-1]
                            be=history.history['epoch'][-1]
                            floss=history.history['loss'][be-1]
                            hacc=history.history['b_v_acc'][-1]
                            loss_f.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                            clr=history.history['lr'][-1]
                            tt=time.time()
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                        final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                        Epo=be, cBS=batchSize, BN=batchNorm,
                                        input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                        else:
                            tag=tag.replace('_noPretrain_nSCV3_','_noPretrain_nSCV3_BS%d_'%(batchSize))
                            iters=int((fdata.train.num_examples*Mini_Epochs)/batchSize)+1
                            clr=K.get_value(ferfm.optimizer.lr)
                            for i in range(iters):
                                if (i+1)%lrstep==0 and ovt_drate<1:
                                    clr=clr*ovt_drate
                                    K.set_value(ferfm.optimizer.lr, clr)
                                batch=fdata.train.next_batch(batchSize, shuffle=False)
                                if Summary:
                                    tloss, tac=ferfm.train_on_batch(x=np.array(batch[0]), y=np.array(batch[1]))
                                else:
                                    tloss, tac=ferfm.train_on_batch(x=np.array(batch[0]), y=np.array(batch[1]))
                                #print(ferfm.metrics_names)
                                if tloss<mini_loss:
                                    mini_loss=tloss
                                valid_loss, ta = ferfm.evaluate(x=np.array(fdata.test.X), y=np.array(fdata.test.Y), batch_size=None)
                                laflag = loss_f.analyzeLossVariation(valid_loss, i, fdata.train.epochs_completed, ta)
                                tt=time.time()
                                print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                        (clr,i,batchSize,fdata.train.epochs_completed, tac, tloss, mini_loss, loss_f.minimun_loss, valid_loss, ta, loss_f.highestAcc, (tt-t1)))
                                if laflag:
                                    predy=ferfm.predict(x=np.array(fdata.test.X))
                                    predl=np.argmax(predy, axis=1)
                                    truel=np.argmax(fdata.test.Y, axis=1)
                                    cm=confusion_matrix(y_true=truel, y_pred=predl)
                                    afc=[]
                                    cts=[]
                                    oaa=overAllAccuracy(cm, afc, cts)
                                    be=fdata.train.epochs_completed
                                    hacc=ta
                                    mini_v_loss=valid_loss
                                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_f.minimun_loss, 
                                        final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                        Epo=be, cBS=batchSize, BN=batchNorm,
                                        input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                    if SaveModel:
                                        ferfm.save(model_save_path)
                                        print('Model saved!')

                        log='%s%d_M%d_D%d_N%d_FN%d_FMfit%s_FLdata%s%s_%s.txt'%(logprefix, GPU_Device_ID, Module, DataSet, NetworkType, FusionNType, str(FusionMfit), str(FLdata), tag, optimizerName)
                        tt=time.time()
                        filelog=open(log,'a')
                        filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\tParameters:%d\n'%(file_record, (tt-t1), optimizerName, log,ferfm.count_params()))
                        print(log)
                        filelog.close()

                        print(log.split('.txt')[0])
                        losslog=log.split('.txt')[0]+'_Runs%d_%d'%(runs, TestID)+'.validationlosslist'
                        losslog=losslog.replace('./logs/ovt/','./logs/ovt/VL/')
                        loss_f.outputlosslist(losslog)
                        if Acc_Oriented:
                            newmodelname=model_save_path.replace('.h5','_Epoch%03d_ACC%f_MiniLoss%f_.h5'%(be, hacc, mini_v_loss))
                        else:
                            newmodelname=model_save_path.replace('.h5','_Epoch%03d_MiniLoss%f_ACC%f_.h5'%(be, mini_v_loss, hacc))
                        if os.path.exists(model_save_path):
                            os.rename((model_save_path),newmodelname)
        elif Module==23 and FUSION:
            if os.path.exists('J:/Models/ovtsaves/'):
                model_save_path_prefix='J:/Models/ovtsaves/M%dFusion/D%d/N%d/'%(Module, DataSet, NetworkType)
                if not os.path.exists(model_save_path_prefix):
                    os.makedirs(model_save_path_prefix)
            else:
                model_save_path_prefix='./saves/ovt/M%dFusion/D%d/N%d/'%(Module, DataSet, NetworkType)
                if not os.path.exists(model_save_path_prefix):
                    os.makedirs(model_save_path_prefix)
            logprefix='./logs/ovt/D%d_ovt_gpu'%(DataSet)
            if ForVisFea:
                logprefix=logprefix.replace('_ovt_','_ovt_ForVisFea_')
            if not FLdata:
                lrstep=None
                SaveModel=False
                t1=time.time()
                timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))
                if Mfit:
                    ftag='_Modelfit'
                else:
                    ftag='_train_on_batch'
                FeaturesPath='./MMFeatures/M%d/OVT_D%d_M%d_N%d_T%d_R%d%s%s_TAGPART_lasttwolayerOutputs.pkl'%(Module, DataSet,Module,NetworkType,TestID
                                            ,runs,timestamp, ftag)
                fpdir=os.path.dirname(FeaturesPath)
                if not os.path.exists(fpdir):
                    os.makedirs(fpdir)
                
                print(logprefix)
                mafa=np.zeros((2, 2))
                Lcount=0.0
                labels={}
                ffmms={}
                ffmms['testffmms']=[]#features from multi-models
                ffmms['trainffmms']=[]
                modelnamelist=[]
                RNT, GNT=FERN.getSeperateNetwork(NetworkType)
                paramsc=0

                ##############Part1
                mini_loss=10000
                loss_a=LOSS_ANA()
                file_record=None
                model_save_path=''
                '''Input Data-------------------------------------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                #
                ##data set loading
                #
                dfile=Dataset_Dictionary.get(DataSet, None)
                if dfile is None:
                    raise RuntimeError('\nERROR: check the DataSet again. It must be one of %s'%(str(Dataset_Dictionary.keys())))
        
                data = loaddata_v4(dfile, TestID, TestID, Module=2, cn=cn)
                #print(data.test.labels)#check the original labels
                Lcount=data.test.num_examples
                if labels.get('test', None) is None:
                    labels['test']=dense_to_one_hot(data.test.labels, cn)
                    labels['train']=dense_to_one_hot(data.train.labels, cn)
                    labels['predict']=np.zeros((Lcount, cn))

                tt=time.time()
                if Mfit:
                    tag=('_noPretrain_keras_model_fit_%depochs_part1GeoN%d.txt'%(Mini_Epochs, GNT))
                else:
                    lrstep=int(data.train.num_examples/batchSize*times)
                    print('\nlearning rate decay steps: %d'%lrstep)
                    tag=('_noPretrain_newSCV3_%depochs_upDLoLo_part1GeoN%d.txt'%(Mini_Epochs, GNT))
                log='%s%d_M%d_D%d_N%d%s'%(logprefix, GPU_Device_ID, Module, DataSet, NetworkType, tag)
            
                print('Time used for loading data: %fs'%(tt-t1))
                    
                model_save_path='%sOVT_D%d_M%d_N%d_T%d_R%d%s_GPU%d%s.h5'%(model_save_path_prefix,DataSet,2,GNT,TestID
                                        ,runs,timestamp,GPU_Device_ID,tag.replace('.txt',''))
                #print(log, model_save_path)#check the log file path and model save path
                '''Input Data Ends-----------------------------------------------------------------------------------------'''

                '''MODULE2---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('GeoNetwork: %s'%(GNT))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                #print(type(data.train.geo), type(data.train.geo[0]), len(data.train.geo[0]))
                data_shape=data.train.geo[0].shape
                print('data shape: %s'%(str(data_shape)))
                GEOWeight=0.2#rejust the linear vote weight of GNT
                
                #g=tf.Graph()
                #with g.as_default():
                ferm=FERN.GetNetworkKV(data_shape, GNT, 'AE', cn, batchNorm, DataSet)
                plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(2,GNT,DataSet,ferm.count_params()), show_shapes=True)
                paramsc=paramsc+ferm.count_params()
                getConcatenate1=K.function([ferm.layers[0].input, K.learning_phase()],[ferm.layers[-2].output])#output ndarray
                ####
                #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                cLR=0.0001
                if M23CPL:
                    ferm.compile(optimizer=op, loss=FERN.CPL, metrics=['accuracy'])
                    log=log.replace('.txt','_CPL.txt')
                    tag=tag.replace('.txt','_CPL.txt')
                    print('\n\n\nUsing CPL in training.\n\n')
                else:
                    ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                #with tf.Session(graph=g) as sess:
                #    ferm.session=sess
                #    sess.run(tf.global_variables_initializer())
                if Mfit:
                    reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                    ccm=EvaCMSM(validation=(np.array(data.test.geo), np.array(data.test.labels)), 
                                filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, 
                                Acc_Oriented=Acc_Oriented, tag='AE', GetffmmFlag=True, funcops=getConcatenate1, traindata=data.train.geo, Module=Module)
                    cbl=[reduce_lr, ccm]
                    if Summary:
                        history=ferm.fit(x=np.array(data.train.geo), y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    else:
                        history=ferm.fit(x=np.array(data.train.geo), y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                
                    cm=history.history['cm'][-1]
                    mini_v_loss=history.history['b_v_loss'][-1]
                    be=history.history['epoch'][-1]
                    floss=history.history['loss'][be-1]
                    hacc=history.history['b_v_acc'][-1]
                    predy=history.history['predy'][-1]
                    for indexL in range(Lcount):
                        labels['predict'][indexL]=labels['predict'][indexL]+predy[indexL]*hacc*GEOWeight
                    temtestffmm=history.history['testffmm'][-1]
                    temtrainffmm=history.history['trainffmm'][-1]
                    loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                    clr=history.history['lr'][-1]
                    tt=time.time()
                    afc=[]
                    cts=[]
                    oaa=overAllAccuracy(cm, afc, cts)
                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                    mafa[0][0]=hacc
                    mafa[0][1]=oaa
                else:
                    iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                    clr=K.get_value(ferm.optimizer.lr)
                    for i in range(iters):
                        if (i+1)%lrstep==0 and ovt_drate<1:
                            clr=clr*ovt_drate
                            K.set_value(ferm.optimizer.lr, clr)
                        batch=data.train.next_batch(batchSize, shuffle=False)
                        if Summary:
                            tloss, tac=ferm.train_on_batch(x=np.array(batch[1]), y=np.array(batch[5]))
                        else:
                            tloss, tac=ferm.train_on_batch(x=np.array(batch[1]), y=np.array(batch[5]))
                        #print(ferm.metrics_names)
                        if tloss<mini_loss:
                            mini_loss=tloss
                        valid_loss, ta = ferm.evaluate(x=np.array(data.test.geo), y=np.array(data.test.labels), batch_size=None)
                        laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                        tt=time.time()
                        print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                        if laflag:
                            predy=ferm.predict(x=np.array(data.test.geo))
                            temtestffmm=getConcatenate1([data.test.geo, 0])[0]
                            temtestffmm=np.concatenate([temtestffmm, predy],axis=-1)
                                        
                            trainy=ferm.predict(x=np.array(data.train.geo))
                            temtrainffmm=getConcatenate1([data.train.geo, 0])[0]
                            temtrainffmm=np.concatenate([temtrainffmm, trainy],axis=-1)

                            predl=np.argmax(predy, axis=1)
                            truel=np.argmax(data.test.labels, axis=1)
                            cm=confusion_matrix(y_true=truel, y_pred=predl)
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            hacc=ta
                            mini_v_loss=valid_loss
                            be=data.train.epochs_completed
                            mafa[0][0]=hacc
                            mafa[0][1]=oaa
                            for indexL in range(Lcount):
                                labels['predict'][indexL]=labels['predict'][indexL]+predy[indexL]*hacc*GEOWeight
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            if SaveModel and hacc>getsaveMV(2, DataSet):
                                ferm.save(model_save_path)
                                print('Model saved!')
                if Acc_Oriented:
                    newmodelname=model_save_path.replace('.h5','_Epoch%03d_ACC%f_MiniLoss%f_.h5'%(be, hacc, mini_v_loss))
                else:
                    newmodelname=model_save_path.replace('.h5','_Epoch%03d_MiniLoss%f_ACC%f_.h5'%(be, mini_v_loss, hacc))
                if os.path.exists(model_save_path):
                    os.rename((model_save_path),newmodelname)
                ffmms['testffmms'].append(temtestffmm)
                ffmms['trainffmms'].append(temtrainffmm)
                modelnamelist.append(newmodelname)
                optimizerName=str(type(op).__name__)
                tt=time.time()
                log=log.replace('.txt',('_'+optimizerName+'.txt'))
                filelog=open(log,'a')

                print(log.split('.txt')[0])
                losslog=log.split('.txt')[0]+'_Runs%d_%d'%(runs, TestID)+'.validationlosslist'
                if losslog.find('/ovt/')>0:
                    losslog=losslog.replace('./logs/ovt/','./logs/ovt/VL/')
                else:
                    losslog=losslog.replace('./logs/KerasV/','./logs/KerasV/VL/')
                loss_a.outputlosslist(losslog)

                filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\tParameters:%d\n'%(file_record, (tt-t1), optimizerName, losslog,ferm.count_params()))
                print(log)
                filelog.close()


                ############Part2
                mini_loss=10000
                loss_a=LOSS_ANA()
                file_record=None
                model_save_path=''
                '''Input Data-------------------------------------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                #
                ##data set loading
                #
                dfile=Dataset_Dictionary.get(DataSet, None)
                if dfile is None:
                    raise RuntimeError('\nERROR: check the DataSet again. It must be one of %s'%(str(Dataset_Dictionary.keys())))
        
                data = loaddata_v4(dfile, TestID, TestID, Module=3, cn=cn)
                #print(data.test.labels)#check the original labels
                Lcount=data.test.num_examples
                if labels.get('test', None) is None:
                    labels['test']=dense_to_one_hot(data.test.labels, cn)
                    labels['train']=dense_to_one_hot(data.train.labels, cn)

                tt=time.time()
                if Mfit:
                    tag=('_noPretrain_keras_model_fit_%depochs_part2RCFNN%d.txt'%(Mini_Epochs, RNT))
                else:
                    lrstep=int(data.train.num_examples/batchSize*times)
                    print('\nlearning rate decay steps: %d'%lrstep)
                    tag=('_noPretrain_newSCV3_%depochs_upDLoLo_part2RCFNN%d.txt'%(Mini_Epochs, RNT))
                log='%s%d_M%d_D%d_N%d%s'%(logprefix, GPU_Device_ID, Module, DataSet, NetworkType, tag)
            
                print('Time used for loading data: %fs'%(tt-t1))
                    
                model_save_path='%sOVT_D%d_M%d_N%d_T%d_R%d%s_GPU%d%s.h5'%(model_save_path_prefix,DataSet,3,RNT,TestID
                                        ,runs,timestamp,GPU_Device_ID,tag.replace('.txt',''))
                #print(log, model_save_path)#check the log file path and model save path
                '''Input Data Ends-----------------------------------------------------------------------------------------'''
                '''MODULE3---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('RCFN: %s'%(RNT))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
            
                data_shape=[]
                data_shape.append(data.train.eyep[0].shape)
                data_shape.append(data.train.foreheadp[0].shape)
                data_shape.append(data.train.mouthp[0].shape)
                print('data shape: %s'%(str(data_shape)))
                
                #g2=tf.Graph()
                #with g2.as_default():
                ferm=FERN.GetNetworkKV(data_shape, RNT, 'AE', cn, batchNorm, DataSet)
                plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                paramsc=paramsc+ferm.count_params()
                #print(ferm.inputs)
                #for i in range(len(ferm.layers)):
                #    print(i, ferm.layers[i])
                #exit()
                
                ####
                #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                cLR=0.0001
                if M23CPL:
                    ferm.compile(optimizer=op, loss=FERN.CPL, metrics=['accuracy'])
                    log=log.replace('.txt','_CPL.txt')
                    tag=tag.replace('.txt','_CPL.txt')
                    print('\n\n\nUsing CPL in training.\n\n')
                else:
                    if RNT==2 or RNT==3:
                        ferm.compile(optimizer=op, loss=FERN.CPL, metrics=['accuracy'])
                        log=log.replace('.txt','_CPL.txt')
                        tag=tag.replace('.txt','_CPL.txt')
                        print('\n\n\nUsing CPL in training.\n\n')
                    else:
                        ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

                    
                #with tf.Session(graph=g2) as sess:
                #    ferm.session=sess
                #    sess.run(tf.global_variables_initializer())
                if RNT==2352 or RNT==2312:#without forehead patch
                    getConcatenate2=K.function([ferm.layers[0].input, ferm.layers[1].input, K.learning_phase()],[ferm.layers[-2].output])#output ndarray
                    if Mfit:
                        reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                        ccm=EvaCMSM(validation=([np.array(data.test.eyep), np.array(data.test.mouthp)], np.array(data.test.labels)), 
                                    filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, 
                                    Acc_Oriented=Acc_Oriented, tag='AE', GetffmmFlag=True, funcops=getConcatenate2, traindata=[np.array(data.train.eyep), np.array(data.train.mouthp)], Module=Module)
                        cbl=[reduce_lr, ccm]
                        if Summary:
                            history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                        else:
                            history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                        #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                        #exit()
                        cm=history.history['cm'][-1]
                        mini_v_loss=history.history['b_v_loss'][-1]
                        be=history.history['epoch'][-1]
                        floss=history.history['loss'][be-1]
                        hacc=history.history['b_v_acc'][-1]
                        temtestffmm=history.history['testffmm'][-1]
                        temtrainffmm=history.history['trainffmm'][-1]
                        predy=history.history['predy'][-1]
                        for indexL in range(Lcount):
                            labels['predict'][indexL]=labels['predict'][indexL]+predy[indexL]*hacc
                        loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                        clr=history.history['lr'][-1]
                        tt=time.time()
                        afc=[]
                        cts=[]
                        oaa=overAllAccuracy(cm, afc, cts)
                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                    final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                    Epo=be, cBS=batchSize, BN=batchNorm,
                                    input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                        mafa[1][0]=hacc
                        mafa[1][1]=oaa
                    else:
                        iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                        clr=K.get_value(ferm.optimizer.lr)
                        for i in range(iters):
                            if (i+1)%lrstep==0 and ovt_drate<1:
                                clr=clr*ovt_drate
                                K.set_value(ferm.optimizer.lr, clr)
                            batch=data.train.next_batch(batchSize, shuffle=False)
                            if Summary:
                                tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[4])], y=np.array(batch[5]))
                            else:
                                tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[4])], y=np.array(batch[5]))
                            #print(ferm.metrics_names)
                            if tloss<mini_loss:
                                mini_loss=tloss
                            valid_loss, ta = ferm.evaluate(x=[np.array(data.test.eyep), np.array(data.test.mouthp)], y=np.array(data.test.labels), batch_size=None)
                            laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                            tt=time.time()
                            print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                    (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                            if laflag:
                                predy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.mouthp)])
                                temtestffmm=getConcatenate2([np.array(data.test.eyep), np.array(data.test.mouthp), 0])[0]
                                temtestffmm=np.concatenate([temtestffmm, predy],axis=-1)
                                            
                                trainy=ferm.predict(x=[np.array(data.train.eyep), np.array(data.train.mouthp)])
                                temtrainffmm=getConcatenate2([np.array(data.train.eyep), np.array(data.train.mouthp), 0])[0]
                                temtrainffmm=np.concatenate([temtrainffmm, trainy],axis=-1)
                                predl=np.argmax(predy, axis=1)
                                truel=np.argmax(data.test.labels, axis=1)
                                cm=confusion_matrix(y_true=truel, y_pred=predl)
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                hacc=ta
                                mini_v_loss=valid_loss
                                be=data.train.epochs_completed
                                mafa[1][0]=hacc
                                mafa[1][1]=oaa
                                for indexL in range(Lcount):
                                    labels['predict'][indexL]=labels['predict'][indexL]+predy[indexL]*hacc
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                    final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                    Epo=be, cBS=batchSize, BN=batchNorm,
                                    input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                if SaveModel and hacc>getsaveMV(3, DataSet):
                                    ferm.save(model_save_path)
                                    print('Model saved!')
                else:
                    getConcatenate2=K.function([ferm.layers[0].input, ferm.layers[1].input, ferm.layers[2].input, K.learning_phase()],[ferm.layers[-2].output])#output ndarray
                    if Mfit:
                        reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                        ccm=EvaCMSM(validation=([np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)], np.array(data.test.labels)), 
                                    filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, 
                                    tag='AE', GetffmmFlag=True, funcops=getConcatenate2, traindata=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp)], Module=Module)
                        cbl=[reduce_lr, ccm]
                        if Summary:
                            history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                        else:
                            history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                        #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                        #exit()
                        cm=history.history['cm'][-1]
                        mini_v_loss=history.history['b_v_loss'][-1]
                        be=history.history['epoch'][-1]
                        floss=history.history['loss'][be-1]
                        hacc=history.history['b_v_acc'][-1]
                        temtestffmm=history.history['testffmm'][-1]
                        temtrainffmm=history.history['trainffmm'][-1]
                        predy=history.history['predy'][-1]
                        for indexL in range(Lcount):
                            labels['predict'][indexL]=labels['predict'][indexL]+predy[indexL]*hacc
                        loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                        clr=history.history['lr'][-1]
                        tt=time.time()
                        afc=[]
                        cts=[]
                        oaa=overAllAccuracy(cm, afc, cts)
                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                    final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                    Epo=be, cBS=batchSize, BN=batchNorm,
                                    input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                        mafa[1][0]=hacc
                        mafa[1][1]=oaa
                    else:
                        iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                        clr=K.get_value(ferm.optimizer.lr)
                        for i in range(iters):
                            if (i+1)%lrstep==0 and ovt_drate<1:
                                clr=clr*ovt_drate
                                K.set_value(ferm.optimizer.lr, clr)
                            batch=data.train.next_batch(batchSize, shuffle=False)
                            if Summary:
                                tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[3]), np.array(batch[4])], y=np.array(batch[5]))
                            else:
                                tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[3]), np.array(batch[4])], y=np.array(batch[5]))
                            #print(ferm.metrics_names)
                            if tloss<mini_loss:
                                mini_loss=tloss
                            valid_loss, ta = ferm.evaluate(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)], y=np.array(data.test.labels), batch_size=None)
                            laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                            tt=time.time()
                            print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                    (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                            if laflag:
                                predy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)])
                                temtestffmm=getConcatenate2([np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp), 0])[0]
                                temtestffmm=np.concatenate([temtestffmm, predy],axis=-1)
                                            
                                trainy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)])
                                temtrainffmm=getConcatenate2([np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp), 0])[0]
                                temtrainffmm=np.concatenate([temtrainffmm, trainy],axis=-1)
                                
                                predl=np.argmax(predy, axis=1)
                                truel=np.argmax(data.test.labels, axis=1)
                                cm=confusion_matrix(y_true=truel, y_pred=predl)
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                hacc=ta
                                mini_v_loss=valid_loss
                                be=data.train.epochs_completed
                                mafa[1][0]=hacc
                                mafa[1][1]=oaa
                                for indexL in range(Lcount):
                                    labels['predict'][indexL]=labels['predict'][indexL]+predy[indexL]*hacc
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                    final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                    Epo=be, cBS=batchSize, BN=batchNorm,
                                    input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                if SaveModel and hacc>getsaveMV(3, DataSet):
                                    ferm.save(model_save_path)
                                    print('Model saved!')
                if Acc_Oriented:
                    newmodelname=model_save_path.replace('.h5','_Epoch%03d_ACC%f_MiniLoss%f_.h5'%(be, hacc, mini_v_loss))
                else:
                    newmodelname=model_save_path.replace('.h5','_Epoch%03d_MiniLoss%f_ACC%f_.h5'%(be, mini_v_loss, hacc))
                if os.path.exists(model_save_path):
                    os.rename((model_save_path),newmodelname)
                ffmms['testffmms'].append(temtestffmm)
                ffmms['trainffmms'].append(temtrainffmm)
                modelnamelist.append(newmodelname)
                te=time.time()
                optimizerName=str(type(op).__name__)
                tt=time.time()
                log=log.replace('.txt',('_'+optimizerName+'.txt'))
                filelog=open(log,'a')

                print(log.split('.txt')[0])
                losslog=log.split('.txt')[0]+'_Runs%d_%d'%(runs, TestID)+'.validationlosslist'
                if losslog.find('/ovt/')>0:
                    losslog=losslog.replace('./logs/ovt/','./logs/ovt/VL/')
                else:
                    losslog=losslog.replace('./logs/KerasV/','./logs/KerasV/VL/')
                loss_a.outputlosslist(losslog)

                filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\tParameters: %d\n'%(file_record, (tt-t1), optimizerName, losslog, ferm.count_params()))
                print(log)
                filelog.close()

                '''Expression Loop ENDS---------------------------------------------------------------------------------------------'''
                #######linear voting using the weight probobilities of expressions
                aepredl=np.argmax(labels['predict'], axis=1)
                aetruel=np.argmax(labels['test'], axis=1)
                aecm=confusion_matrix(y_true=aetruel, y_pred=aepredl)
                aeoaa=balanced_accuracy_score(y_true=aetruel, y_pred=aepredl)
                aeacc=accuracy_score(y_true=aetruel, y_pred=aepredl)
                #######
                te=time.time()
                print('\nTotal time for all %d expressions: %fs'%(cn, (te-t0)))
                log='%s%d_M%d_D%d_N%d%s_linearVote_%s.txt'%(logprefix,GPU_Device_ID,Module,DataSet,NetworkType, tag[0:tag.rfind('_')],optimizerName)
                print('Summary file: %s'%log)
                print('P1hacc P1oaa P2hacc P2oaa: %s\tBAC: %f\tAC: %f'%(str(mafa), aeoaa, aeacc))
  
                logOVTFileKeras(log, runs, OAA=aeoaa, TA=aeacc, mafa=mafa, BN=batchNorm, CM=aecm, TC=(te-t0), LS=lrstep, ILR=cLR, FLR=clr, iBS=batchSize, input=sys.argv, T=time.localtime(te), df=dfile, params=paramsc, Module=Module)
                fusiondata={}
                #print(type(ffmms['trainffmms']), len(ffmms['trainffmms']))
                #print(ffmms['trainffmms'][0].shape, ffmms['trainffmms'][1].shape)
                fusiondata['trainX']=np.c_[ffmms['trainffmms'][0], ffmms['trainffmms'][1]]
                fusiondata['trainY']=labels['train']
                fusiondata['testX']=np.c_[ffmms['testffmms'][0], ffmms['testffmms'][1]]
                fusiondata['testY']=labels['test']
                with open(FeaturesPath,'wb') as fpkl:
                    pickle.dump(fusiondata, fpkl, 4)
                    print('Features saved in %s'%(FeaturesPath))
                with open(FeaturesPath.replace('.pkl','.info'),'w') as finfo:
                    for v in modelnamelist:
                        finfo.write('%s\n'%(v))
                fdata=Datasets(train=DataSetForAnonymousData(X=fusiondata['trainX'], Y=fusiondata['trainY']), 
                                test=DataSetForAnonymousData(X=fusiondata['testX'], Y=fusiondata['testY']))
            else:
                t1=time.time()
                timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))
                dfile=getSelectedDataFile(DataSet, NetworkType, TestID)
                fdata=loadAnonymousData(dfile)
                print('\n\n\nPretrain data loaded.\n\n')
                file_record=None

            ####### FUSION part
            #if runs%3==0:
            #    SaveModel=True
            #else:
            #    SaveModel=False
            SaveModel=False
            if Mfit:
                tag=('_keras_model_fit_%depochs_M23FUSIONHEAD%d_FLd%s_V2g'%(Mini_Epochs, FusionNType, str(FLdata)))
            else:
                lrstep=int(fdata.train.num_examples/batchSize*times)
                tag=('_nSCV3_%depochs_M23FUSIONHEAD%d_FLd%s_V2g'%(Mini_Epochs, FusionNType, str(FLdata)))
            if FusionMfit:
                model_save_path='%sOVT_M%d_D%d_N%d_FN%d_T%d_FMfit%s_R%d_FLdata%s%s_GPU%d%s.h5'%(model_save_path_prefix,Module,DataSet,NetworkType,FusionNType,TestID
                                        ,str(FusionMfit),runs,str(FLdata),timestamp,GPU_Device_ID,tag)
            else:
                model_save_path='%sOVT_M%d_D%d_N%d_FN%d_T%d_FMfit%sBS%d_R%d_FLdata%s%s_GPU%d%s.h5'%(model_save_path_prefix,Module,DataSet,NetworkType,FusionNType,TestID
                                        ,str(FusionMfit), batchSize,runs,str(FLdata),timestamp,GPU_Device_ID,tag)
            loss_f=LOSS_ANA()
            #fusionGrap=tf.Graph()
            #with fusionGrap.as_default():
            fdata_shape=fdata.train.X[0].shape
            #print(fdata_shape, len(fdata_shape), FusionNType)
            if BatchLearning:
                import BatchLearningModels as BLM
                t1=time.time()
                logprefix='./logs/BLM/D%d_'%(DataSet)
                log='%sM%d_D%d_N%d_BatchLearning_FLdata%s'%(logprefix, Module, DataSet, NetworkType, str(FLdata))
                bm=BLM.BatchLearning(fdata=fdata)

                
                #### Adaboosting #extremly bad performance
                ##print('\nAdaboosting')
                ##filelog = '%s_Adaboosting.txt'%(log)
                ##bm.train(generate_model = BLM.boosting_classfier, logfile = filelog, fold=TestID, argv=sys.argv)

                ## logistic regression
                print('\nLogisticRegression')
                filelog = '%s_LogisticRegression.txt'%(log)
                bm.train(generate_model = BLM.logistic_regression, logfile = filelog, fold=TestID, argv=sys.argv)

                
                ## SELECTFIRST LogisticRegression rbf
                print('\nSelectFeatureFirst LogisticRegression')
                filelog = '%s_SelectFeatFirst_LogisticRegression.txt'%(log)
                bm.trainWithFeatureSelectionFirst(generate_model = BLM.logistic_regression, logfile = filelog, fold=TestID, argv=sys.argv)
                
                ## decision tree
                print('\nDecision Tree')
                filelog = '%s_DecisionTree.txt'%(log)
                bm.train(generate_model = BLM.decision_tree, logfile = filelog, fold=TestID, argv=sys.argv)
                
                ## bagging
                print('\nBagging')
                filelog = '%s_Bagging.txt'%(log)
                bm.train(generate_model = BLM.bagging_classifier, logfile = filelog, fold=TestID, argv=sys.argv)

                ## SVM rbf
                print('\nSVM rbf kernel')
                filelog = '%s_SVMrbf.txt'%(log)
                bm.libsvmtrain(kernel='rbf', logfile = filelog, fold=TestID, argv=sys.argv)

                
                ## SELECTFIRST SVM rbf
                print('\nSelectFeatureFirst SVM rbf kernel')
                filelog = '%s_SelectFeatFirst_SVMrbf.txt'%(log)
                bm.libsvmtrainFeatureSelectionFirst(kernel='rbf', logfile = filelog, fold=TestID, argv=sys.argv)
            else:
                ferfm=FERN.GetFusionNetworkKV(fdata_shape, FusionNType, cn, DataSet)
                plot_model(ferfm, to_file='./modelfigures/M%dN%dFusionN%dD%dmodel_params-%d.png'%(Module,NetworkType,FusionNType,DataSet,ferfm.count_params()), show_shapes=True)
                if RADAM: 
                    op=OPT.RAdam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                else:
                    op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                if LOOKAHEAD:
                    op=OPT.Lookahead(op, sync_period=5, slow_step=0.5)#
                cLR=0.001
                ferfm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                optimizerName=str(type(op).__name__)
                mini_loss=np.inf
                #with tf.Session(graph=fusionGrap) as sess:
                #ferfm.session=sess
                #sess.run(tf.global_variables_initializer())
                if FusionMfit:
                    reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                    ccm=EvaCMSM(validation=(np.array(fdata.test.X), np.array(fdata.test.Y)), filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='FusionHead%d'%(FusionNType))
                    cbl=[reduce_lr, ccm]
                    if Summary:
                        history=ferfm.fit(x=np.array(fdata.train.X), y=np.array(fdata.train.Y), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    else:
                        history=ferfm.fit(x=np.array(fdata.train.X), y=np.array(fdata.train.Y), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    cm=history.history['cm'][-1]
                    mini_v_loss=history.history['b_v_loss'][-1]
                    be=history.history['epoch'][-1]
                    floss=history.history['loss'][be-1]
                    hacc=history.history['b_v_acc'][-1]
                    loss_f.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                    clr=history.history['lr'][-1]
                    tt=time.time()
                    afc=[]
                    cts=[]
                    oaa=overAllAccuracy(cm, afc, cts)
                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                else:
                    tag=tag.replace('_noPretrain_nSCV3_','_noPretrain_nSCV3_BS%d_'%(batchSize))
                    iters=int((fdata.train.num_examples*Mini_Epochs)/batchSize)+1
                    clr=K.get_value(ferfm.optimizer.lr)
                    for i in range(iters):
                        if (i+1)%lrstep==0 and ovt_drate<1:
                            clr=clr*ovt_drate
                            K.set_value(ferfm.optimizer.lr, clr)
                        batch=fdata.train.next_batch(batchSize, shuffle=False)
                        if Summary:
                            tloss, tac=ferfm.train_on_batch(x=np.array(batch[0]), y=np.array(batch[1]))
                        else:
                            tloss, tac=ferfm.train_on_batch(x=np.array(batch[0]), y=np.array(batch[1]))
                        #print(ferfm.metrics_names)
                        if tloss<mini_loss:
                            mini_loss=tloss
                        valid_loss, ta = ferfm.evaluate(x=np.array(fdata.test.X), y=np.array(fdata.test.Y), batch_size=None)
                        laflag = loss_f.analyzeLossVariation(valid_loss, i, fdata.train.epochs_completed, ta)
                        tt=time.time()
                        print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                (clr,i,batchSize,fdata.train.epochs_completed, tac, tloss, mini_loss, loss_f.minimun_loss, valid_loss, ta, loss_f.highestAcc, (tt-t1)))
                        if laflag:
                            predy=ferfm.predict(x=np.array(fdata.test.X))
                            predl=np.argmax(predy, axis=1)
                            truel=np.argmax(fdata.test.Y, axis=1)
                            cm=confusion_matrix(y_true=truel, y_pred=predl)
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            be=fdata.train.epochs_completed
                            hacc=ta
                            mini_v_loss=valid_loss
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_f.minimun_loss, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            if SaveModel:
                                ferfm.save(model_save_path)
                                print('Model saved!')

                log='%s%d_M%d_D%d_N%d_FN%d_FMfit%s_FLdata%s%s_%s.txt'%(logprefix, GPU_Device_ID, Module, DataSet, NetworkType, FusionNType, str(FusionMfit), str(FLdata), tag, optimizerName)
                tt=time.time()
                filelog=open(log,'a')
                filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\tParameters:%d\n'%(file_record, (tt-t1), optimizerName, log,ferfm.count_params()))
                print(log)
                filelog.close()

                print(log.split('.txt')[0])
                losslog=log.split('.txt')[0]+'_Runs%d_%d'%(runs, TestID)+'.validationlosslist'
                if losslog.find('/ovt/')>0:
                    losslog=losslog.replace('./logs/ovt/','./logs/ovt/VL/')
                else:
                    losslog=losslog.replace('./logs/KerasV/','./logs/KerasV/VL/')
                loss_f.outputlosslist(losslog)
                if Acc_Oriented:
                    newmodelname=model_save_path.replace('.h5','_Epoch%03d_ACC%f_MiniLoss%f_.h5'%(be, hacc, mini_v_loss))
                else:
                    newmodelname=model_save_path.replace('.h5','_Epoch%03d_MiniLoss%f_ACC%f_.h5'%(be, mini_v_loss, hacc))
                if os.path.exists(model_save_path):
                    os.rename((model_save_path),newmodelname)
        else:
            print('<<<Program Info: in this branch the pramater NewC is outcast.')
            logprefix='./logs/KerasV/D%d_kerasV_gpu'%(DataSet)
            if sf:
                logprefix='./logs/KerasV/D%d_kerasV_Shuffle_gpu'%(DataSet)
            else:
                logprefix='./logs/KerasV/D%d_kerasV_gpu'%(DataSet)
            if ForVisFea:
                logprefix=logprefix.replace('_kerasV_','_kerasV_ForVisFea_')
            mini_loss=10000
            loss_a=LOSS_ANA()
            file_record=None
            t1=time.time()
            timestamp=time.strftime('_%Y%m%d%H%M%S',time.localtime(t1))
            model_save_path=''
            '''Input Data-------------------------------------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #
            ##data set loading
            #
            dfile=Dataset_Dictionary.get(DataSet, None)
            if dfile is None:
                raise RuntimeError('\nERROR: check the DataSet again. It must be one of %s'%(str(Dataset_Dictionary.keys())))
                
            data = loaddata_v4(dfile, TestID, TestID, Module=Module, cn=cn)
            
            tt=time.time()
            if Mfit:
                tag='_noPretrain_keras_model_fit_%depochs.txt'%(Mini_Epochs)
            else:
                lrstep=int(data.train.num_examples/batchSize*times)
                print('\nlearning rate decay steps: %d'%lrstep)
                tag='_noPretrain_newSCV3_%depochs_upDLoLo.txt'%(Mini_Epochs)
            log='%s%d_M%d_D%d_N%d_AE%s'%(logprefix, GPU_Device_ID, Module, DataSet, NetworkType, tag)
            
            print('Time used for loading data: %fs'%(tt-t1))

            if os.path.exists('J:/Models/saves/'):
                model_save_path='J:/Models/saves/M%d/D%d/N%d/'%(Module, DataSet, NetworkType)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
            else:
                model_save_path='./saves/M%d/D%d/N%d/'%(Module, DataSet, NetworkType)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
            model_save_path='%sD%d_M%d_N%d_T%d_R%d%s_GPU%d%s.h5'%(model_save_path,DataSet,Module,NetworkType,TestID
                                    ,runs,timestamp,GPU_Device_ID,tag.replace('.txt',''))
            #print(log, model_save_path)#check the log file path and model save path
            '''Input Data Ends-----------------------------------------------------------------------------------------'''
            if Module==11:
                '''MODULE11---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('WCPCN: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                #if runs%2==0:
                #    sf=True##really bad and unconverge
                sf=False
                data_shape=data.train.wcs[0].shape
                print('data shape: %s'%(str(data_shape)))

                ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                ####For the KDEF tests, the best option is use default Adam without decay lr, the second best is default Adadelta.
                #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)#default lr 0.002 runs0-runs7, 0.001 for runs8 and runs9
                #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)# default lr 1.0 rho0.95 for runs10, lr 1.0 rho 0.8 for runs11
                #op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                if RADAM: 
                    op=OPT.RAdam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                else:
                    op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                if LOOKAHEAD:
                    op=OPT.Lookahead(op, sync_period=5, slow_step=0.5)#
                cLR=0.001
                ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])

                if Mfit:
                    ccm=EvaCMSM(validation=(np.array(data.test.wcs), np.array(data.test.labels)), filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                    reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                    cbl=[ccm, reduce_lr]
                    if DataSet%10==7 or DataSet%10==8 or DataSet%10==9:
                        batchSize=240
                    if Summary:
                        history=ferm.fit(x=np.array(data.train.wcs), y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    else:
                        history=ferm.fit(x=np.array(data.train.wcs), y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                        #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                    #exit()
                    cm=history.history['cm'][-1]
                    mini_v_loss=history.history['b_v_loss'][-1]
                    be=history.history['epoch'][-1]
                    floss=history.history['loss'][be-1]
                    hacc=history.history['b_v_acc'][-1]
                    loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                    clr=history.history['lr'][-1]
                    tt=time.time()
                    afc=[]
                    cts=[]
                    oaa=overAllAccuracy(cm, afc, cts)
                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                Epo=be, cBS=batchSize, iBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                else:
                    iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                    clr=K.get_value(ferm.optimizer.lr)
                    for i in range(iters):
                        if (i+1)%lrstep==0 and ovt_drate<1:
                            clr=clr*ovt_drate
                            K.set_value(ferm.optimizer.lr, clr)
                        batch=data.train.next_batch(batchSize, shuffle=sf)
                        if Summary:
                            tloss, tac=ferm.train_on_batch(x=np.array(batch[7]), y=np.array(batch[5]))
                        else:
                            tloss, tac=ferm.train_on_batch(x=np.array(batch[7]), y=np.array(batch[5]))
                        #print(ferm.metrics_names)
                        if tloss<mini_loss:
                            mini_loss=tloss
                        valid_loss, ta = ferm.evaluate(x=np.array(data.test.wcs), y=np.array(data.test.labels), batch_size=test_bat)
                        laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                        tt=time.time()
                        print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                        if laflag:
                            predy=ferm.predict(x=np.array(data.test.wcs))
                            predl=np.argmax(predy, axis=1)
                            truel=np.argmax(data.test.labels, axis=1)
                            cm=confusion_matrix(y_true=truel, y_pred=predl)
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            hacc=ta
                            mini_v_loss=valid_loss
                            be=data.train.epochs_completed
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            if SaveModel:
                                ferm.save(model_save_path)
                                print('Model saved!')
            elif Module==3:
                '''MODULE3---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('RCFN: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
            
                data_shape=[]
                data_shape.append(data.train.eyep[0].shape)
                data_shape.append(data.train.foreheadp[0].shape)
                data_shape.append(data.train.mouthp[0].shape)
                print('data shape: %s'%(str(data_shape)))
                
                g=tf.Graph()
                with g.as_default():
                    ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                    plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                    ####
                    #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                    #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                    op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                    cLR=0.0001
                    if NetworkType==2 or NetworkType==3:
                        ferm.compile(optimizer=op, loss=FERN.CPL, metrics=['accuracy'])
                    else:
                        ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    with tf.Session(graph=g) as sess:
                        ferm.session=sess
                        sess.run(tf.global_variables_initializer())
                        if NetworkType==2352 or NetworkType==2312:#without forehead patch
                            if Mfit:
                                reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                                ccm=EvaCMSM(validation=([np.array(data.test.eyep), np.array(data.test.mouthp)], np.array(data.test.labels)), 
                                            filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                                cbl=[reduce_lr, ccm]
                                if Summary:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                else:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                                #exit()
                                cm=history.history['cm'][-1]
                                mini_v_loss=history.history['b_v_loss'][-1]
                                be=history.history['epoch'][-1]
                                floss=history.history['loss'][be-1]
                                hacc=history.history['b_v_acc'][-1]
                                loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                                clr=history.history['lr'][-1]
                                tt=time.time()
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                            final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            else:
                                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                                clr=K.get_value(ferm.optimizer.lr)
                                for i in range(iters):
                                    if (i+1)%lrstep==0 and ovt_drate<1:
                                        clr=clr*ovt_drate
                                        K.set_value(ferm.optimizer.lr, clr)
                                    batch=data.train.next_batch(batchSize, shuffle=False)
                                    if Summary:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[4])], y=np.array(batch[5]))
                                    else:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[4])], y=np.array(batch[5]))
                                    #print(ferm.metrics_names)
                                    if tloss<mini_loss:
                                        mini_loss=tloss
                                    valid_loss, ta = ferm.evaluate(x=[np.array(data.test.eyep), np.array(data.test.mouthp)], y=np.array(data.test.labels), batch_size=None)
                                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                                    tt=time.time()
                                    print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                            (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                                    if laflag:
                                        predy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.mouthp)])
                                        predl=np.argmax(predy, axis=1)
                                        truel=np.argmax(data.test.labels, axis=1)
                                        cm=confusion_matrix(y_true=truel, y_pred=predl)
                                        afc=[]
                                        cts=[]
                                        oaa=overAllAccuracy(cm, afc, cts)
                                        hacc=ta
                                        mini_v_loss=valid_loss
                                        be=data.train.epochs_completed
                                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                            final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                        if SaveModel and hacc>getsaveMV(Module, DataSet):
                                            ferm.save(model_save_path)
                                            print('Model saved!')
                        else:
                            if Mfit:
                                reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                                ccm=EvaCMSM(validation=([np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)], np.array(data.test.labels)), 
                                            filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                                cbl=[reduce_lr, ccm]
                                if Summary:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                else:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                                #exit()
                                cm=history.history['cm'][-1]
                                mini_v_loss=history.history['b_v_loss'][-1]
                                be=history.history['epoch'][-1]
                                floss=history.history['loss'][be-1]
                                hacc=history.history['b_v_acc'][-1]
                                loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                                clr=history.history['lr'][-1]
                                tt=time.time()
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                            final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            else:
                                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                                clr=K.get_value(ferm.optimizer.lr)
                                for i in range(iters):
                                    if (i+1)%lrstep==0 and ovt_drate<1:
                                        clr=clr*ovt_drate
                                        K.set_value(ferm.optimizer.lr, clr)
                                    batch=data.train.next_batch(batchSize, shuffle=False)
                                    if Summary:
                                    
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[3]), np.array(batch[4])], y=np.array(batch[5]))
                                    else:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[3]), np.array(batch[4])], y=np.array(batch[5]))
                                    #print(ferm.metrics_names)
                                    if tloss<mini_loss:
                                        mini_loss=tloss
                                    valid_loss, ta = ferm.evaluate(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)], y=np.array(data.test.labels), batch_size=None)
                                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                                    tt=time.time()
                                    print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                            (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                                    if laflag:
                                        predy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)])
                                        predl=np.argmax(predy, axis=1)
                                        truel=np.argmax(data.test.labels, axis=1)
                                        cm=confusion_matrix(y_true=truel, y_pred=predl)
                                        afc=[]
                                        cts=[]
                                        oaa=overAllAccuracy(cm, afc, cts)
                                        hacc=ta
                                        mini_v_loss=valid_loss
                                        be=data.train.epochs_completed
                                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                            final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                        if SaveModel and hacc>getsaveMV(Module, DataSet):
                                            ferm.save(model_save_path)
                                            print('Model saved!')
            elif Module==1:
                '''MODULE1---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('OPNetwork: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
            
                data_shape=data.train.imgs[0].shape
                print('data shape: %s'%(str(data_shape)))
                
                g=tf.Graph()
                with g.as_default():
                    ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                    plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                    ####
                    #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                    #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                    op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                    cLR=0.0001
                    ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    with tf.Session(graph=g) as sess:
                        ferm.session=sess
                        sess.run(tf.global_variables_initializer())
                        if Mfit:
                            reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                            ccm=EvaCMSM(validation=(np.array(data.test.imgs), np.array(data.test.labels)), 
                                        filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                            cbl=[reduce_lr, ccm]
                            if Summary:
                                history=ferm.fit(x=np.array(data.train.imgs), y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                            else:
                                history=ferm.fit(x=np.array(data.train.imgs), y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                            #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                            #exit()
                            cm=history.history['cm'][-1]
                            mini_v_loss=history.history['b_v_loss'][-1]
                            be=history.history['epoch'][-1]
                            floss=history.history['loss'][be-1]
                            hacc=history.history['b_v_acc'][-1]
                            loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                            clr=history.history['lr'][-1]
                            tt=time.time()
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                        final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                        Epo=be, cBS=batchSize, BN=batchNorm,
                                        input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                        else:
                            iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                            clr=K.get_value(ferm.optimizer.lr)
                            for i in range(iters):
                                if (i+1)%lrstep==0 and ovt_drate<1:
                                    clr=clr*ovt_drate
                                    K.set_value(ferm.optimizer.lr, clr)
                                batch=data.train.next_batch(batchSize, shuffle=False)
                                if Summary:
                                    tloss, tac=ferm.train_on_batch(x=np.array(batch[0]), y=np.array(batch[5]))
                                else:
                                    tloss, tac=ferm.train_on_batch(x=np.array(batch[0]), y=np.array(batch[5]))
                                #print(ferm.metrics_names)
                                if tloss<mini_loss:
                                    mini_loss=tloss
                                valid_loss, ta = ferm.evaluate(x=np.array(data.test.imgs), y=np.array(data.test.labels), batch_size=None)
                                laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                                tt=time.time()
                                print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                        (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                                if laflag:
                                    predy=ferm.predict(x=np.array(data.test.imgs))
                                    predl=np.argmax(predy, axis=1)
                                    truel=np.argmax(data.test.labels, axis=1)
                                    cm=confusion_matrix(y_true=truel, y_pred=predl)
                                    afc=[]
                                    cts=[]
                                    oaa=overAllAccuracy(cm, afc, cts)
                                    hacc=ta
                                    mini_v_loss=valid_loss
                                    be=data.train.epochs_completed
                                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                        final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                        Epo=be, cBS=batchSize, BN=batchNorm,
                                        input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                    if SaveModel and hacc>getsaveMV(Module, DataSet):
                                        ferm.save(model_save_path)
                                        print('Model saved!')
            elif Module==2:
                '''MODULE2---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('GeoNetwork: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                #print(type(data.train.geo), type(data.train.geo[0]), len(data.train.geo[0]))
                data_shape=data.train.geo[0].shape
                print('data shape: %s'%(str(data_shape)))
                
                g=tf.Graph()
                with g.as_default():
                    ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                    plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                    ####
                    #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                    #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                    #op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                    if RADAM: 
                        op=OPT.RAdam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                    else:
                        op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                    if LOOKAHEAD:
                        op=OPT.Lookahead(op, sync_period=5, slow_step=0.5)#
                    cLR=0.001
                    ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    with tf.Session(graph=g) as sess:
                        ferm.session=sess
                        sess.run(tf.global_variables_initializer())
                        if Mfit:
                            reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                            ccm=EvaCMSM(validation=(np.array(data.test.geo), np.array(data.test.labels)), 
                                        filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                            cbl=[reduce_lr, ccm]
                            if Summary:
                                history=ferm.fit(x=np.array(data.train.geo), y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                            else:
                                history=ferm.fit(x=np.array(data.train.geo), y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                            #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                            #exit()
                            cm=history.history['cm'][-1]
                            mini_v_loss=history.history['b_v_loss'][-1]
                            be=history.history['epoch'][-1]
                            floss=history.history['loss'][be-1]
                            hacc=history.history['b_v_acc'][-1]
                            loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                            clr=history.history['lr'][-1]
                            tt=time.time()
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                        final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                        Epo=be, cBS=batchSize, BN=batchNorm,
                                        input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                        else:
                            iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                            clr=K.get_value(ferm.optimizer.lr)
                            for i in range(iters):
                                if (i+1)%lrstep==0 and ovt_drate<1:
                                    clr=clr*ovt_drate
                                    K.set_value(ferm.optimizer.lr, clr)
                                batch=data.train.next_batch(batchSize, shuffle=False)
                                if Summary:
                                    tloss, tac=ferm.train_on_batch(x=np.array(batch[1]), y=np.array(batch[5]))
                                else:
                                    tloss, tac=ferm.train_on_batch(x=np.array(batch[1]), y=np.array(batch[5]))
                                #print(ferm.metrics_names)
                                if tloss<mini_loss:
                                    mini_loss=tloss
                                valid_loss, ta = ferm.evaluate(x=np.array(data.test.geo), y=np.array(data.test.labels), batch_size=None)
                                laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                                tt=time.time()
                                print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                        (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                                if laflag:
                                    predy=ferm.predict(x=np.array(data.test.geo))
                                    predl=np.argmax(predy, axis=1)
                                    truel=np.argmax(data.test.labels, axis=1)
                                    cm=confusion_matrix(y_true=truel, y_pred=predl)
                                    afc=[]
                                    cts=[]
                                    oaa=overAllAccuracy(cm, afc, cts)
                                    hacc=ta
                                    mini_v_loss=valid_loss
                                    be=data.train.epochs_completed
                                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                        final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                        Epo=be, cBS=batchSize, BN=batchNorm,
                                        input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                    if SaveModel and hacc>getsaveMV(Module, DataSet):
                                        ferm.save(model_save_path)
                                        print('Model saved!')
            elif Module==4:
                '''MODULE4---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('OPNetwork: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
            
                data_shape=data.train.cropf[0].shape
                print('data shape: %s'%(str(data_shape)))
                
                ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                ####
                #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                ilr=0.0001
                if DataSet%10==0:
                    if NetworkType==999990 or NetworkType==999991 or NetworkType==999992:
                        ilr=0.001
                if RADAM:
                    op=OPT.RAdam(lr=ilr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                else:
                    op=OPT.Adam(lr=ilr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                cLR=ilr
                print('Initial learning rate: %f'%ilr)
                ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                
                if Mfit:
                    reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                    ccm=EvaCMSM(validation=(np.array(data.test.cropf), np.array(data.test.labels)), 
                                filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                    cbl=[reduce_lr, ccm]
                    if Summary:
                        history=ferm.fit(x=np.array(data.train.cropf), y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    else:
                        history=ferm.fit(x=np.array(data.train.cropf), y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                    #exit()
                    cm=history.history['cm'][-1]
                    mini_v_loss=history.history['b_v_loss'][-1]
                    be=history.history['epoch'][-1]
                    floss=history.history['loss'][be-1]
                    hacc=history.history['b_v_acc'][-1]
                    loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                    clr=history.history['lr'][-1]
                    tt=time.time()
                    afc=[]
                    cts=[]
                    oaa=overAllAccuracy(cm, afc, cts)
                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                else:
                    iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                    clr=K.get_value(ferm.optimizer.lr)
                    for i in range(iters):
                        if (i+1)%lrstep==0 and ovt_drate<1:
                            clr=clr*ovt_drate
                            K.set_value(ferm.optimizer.lr, clr)
                        batch=data.train.next_batch(batchSize, shuffle=False)
                        #print(len(batch[6]))
                        #print(type(batch[6][0]))
                        #print(batch[6][0].shape)
                        #print(np.array(batch[6]).shape)
                        #print(np.array(batch[5]).shape)
                        #print(type(data.train.cropf))
                        #print(type(data.train.cropf[0]))
                        #print(data.train.cropf[0].shape)
                        #print(type(np.array(data.train.cropf[0:20])))
                        #print((np.array(data.train.cropf[0:20])).shape)

                        if Summary:
                            tloss, tac=ferm.train_on_batch(x=np.array(batch[6]), y=np.array(batch[5]))
                        else:
                            tloss, tac=ferm.train_on_batch(x=np.array(batch[6]), y=np.array(batch[5]))
                        #print(ferm.metrics_names)
                        if tloss<mini_loss:
                            mini_loss=tloss
                        valid_loss, ta = ferm.evaluate(x=np.array(data.test.cropf), y=np.array(data.test.labels), batch_size=None)
                        laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                        tt=time.time()
                        print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                        if laflag:
                            predy=ferm.predict(x=np.array(data.test.cropf))
                            predl=np.argmax(predy, axis=1)
                            truel=np.argmax(data.test.labels, axis=1)
                            cm=confusion_matrix(y_true=truel, y_pred=predl)
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            hacc=ta
                            mini_v_loss=valid_loss
                            be=data.train.epochs_completed
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            if SaveModel and hacc>getsaveMV(Module, DataSet):
                                ferm.save(model_save_path)
                                print('Model saved!')
            elif Module==23:
                '''MODULE23---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('GEORCFNSlim: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
            
                data_shape=[]
                data_shape.append(data.train.eyep[0].shape)
                data_shape.append(data.train.foreheadp[0].shape)
                data_shape.append(data.train.mouthp[0].shape)
                data_shape.append(data.train.geo[0].shape)
                print('data shape: %s'%(str(data_shape)))
                
                g=tf.Graph()
                with g.as_default():
                    ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                    plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                    ####
                    #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                    #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                    op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                    cLR=0.0001
                    if M23CPL:
                        ferm.compile(optimizer=op, loss=FERN.CPL, metrics=['accuracy'])
                        log=log.replace('.txt','_CPL.txt')
                        tag=tag.replace('.txt','_CPL.txt')
                        print('\n\n\nUsing CPL in training.\n\n')
                    else:
                        ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    with tf.Session(graph=g) as sess:
                        ferm.session=sess
                        sess.run(tf.global_variables_initializer())
                        if NetworkType==2312201 or NetworkType==2352202:
                            if Mfit:
                                reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                                ccm=EvaCMSM(validation=([np.array(data.test.eyep), np.array(data.test.mouthp), np.array(data.test.geo)], np.array(data.test.labels)), 
                                            filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                                cbl=[reduce_lr, ccm]
                                if Summary:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.mouthp), np.array(data.train.geo)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                else:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.mouthp), np.array(data.train.geo)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                                #exit()
                                cm=history.history['cm'][-1]
                                mini_v_loss=history.history['b_v_loss'][-1]
                                be=history.history['epoch'][-1]
                                floss=history.history['loss'][be-1]
                                hacc=history.history['b_v_acc'][-1]
                                loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                                clr=history.history['lr'][-1]
                                tt=time.time()
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                            final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            else:
                                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                                clr=K.get_value(ferm.optimizer.lr)
                                for i in range(iters):
                                    if (i+1)%lrstep==0 and ovt_drate<1:
                                        clr=clr*ovt_drate
                                        K.set_value(ferm.optimizer.lr, clr)
                                    batch=data.train.next_batch(batchSize, shuffle=False)
                                    if Summary:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[4]), np.array(batch[1])], y=np.array(batch[5]))
                                    else:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[4]), np.array(batch[1])], y=np.array(batch[5]))
                                    #print(ferm.metrics_names)
                                    if tloss<mini_loss:
                                        mini_loss=tloss
                                    valid_loss, ta = ferm.evaluate(x=[np.array(data.test.eyep), np.array(data.test.mouthp), np.array(data.test.geo)], y=np.array(data.test.labels), batch_size=None)
                                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                                    tt=time.time()
                                    print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                            (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                                    if laflag:
                                        predy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.mouthp),  np.array(data.test.geo)])
                                        predl=np.argmax(predy, axis=1)
                                        truel=np.argmax(data.test.labels, axis=1)
                                        cm=confusion_matrix(y_true=truel, y_pred=predl)
                                        afc=[]
                                        cts=[]
                                        oaa=overAllAccuracy(cm, afc, cts)
                                        hacc=ta
                                        mini_v_loss=valid_loss
                                        be=data.train.epochs_completed
                                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                            final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                        if SaveModel and hacc>getsaveMV(Module, DataSet):
                                            ferm.save(model_save_path)
                                            print('Model saved!')
                        else:
                            if Mfit:
                                reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                                ccm=EvaCMSM(validation=([np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp), np.array(data.test.geo)], np.array(data.test.labels)), 
                                            filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                                cbl=[reduce_lr, ccm]
                                if Summary:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp), np.array(data.train.geo)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                else:
                                    history=ferm.fit(x=[np.array(data.train.eyep), np.array(data.train.foreheadp), np.array(data.train.mouthp), np.array(data.train.geo)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                                #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                                #exit()
                                cm=history.history['cm'][-1]
                                mini_v_loss=history.history['b_v_loss'][-1]
                                be=history.history['epoch'][-1]
                                floss=history.history['loss'][be-1]
                                hacc=history.history['b_v_acc'][-1]
                                loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                                clr=history.history['lr'][-1]
                                tt=time.time()
                                afc=[]
                                cts=[]
                                oaa=overAllAccuracy(cm, afc, cts)
                                file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                            final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            else:
                                iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                                clr=K.get_value(ferm.optimizer.lr)
                                for i in range(iters):
                                    if (i+1)%lrstep==0 and ovt_drate<1:
                                        clr=clr*ovt_drate
                                        K.set_value(ferm.optimizer.lr, clr)
                                    batch=data.train.next_batch(batchSize, shuffle=False)
                                    if Summary:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.array(batch[1])], y=np.array(batch[5]))
                                    else:
                                        tloss, tac=ferm.train_on_batch(x=[np.array(batch[2]), np.array(batch[3]), np.array(batch[4]), np.array(batch[1])], y=np.array(batch[5]))
                                    #print(ferm.metrics_names)
                                    if tloss<mini_loss:
                                        mini_loss=tloss
                                    valid_loss, ta = ferm.evaluate(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp), np.array(data.test.geo)], y=np.array(data.test.labels), batch_size=None)
                                    laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                                    tt=time.time()
                                    print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                            (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                                    if laflag:
                                        predy=ferm.predict(x=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp),  np.array(data.test.geo)])
                                        predl=np.argmax(predy, axis=1)
                                        truel=np.argmax(data.test.labels, axis=1)
                                        cm=confusion_matrix(y_true=truel, y_pred=predl)
                                        afc=[]
                                        cts=[]
                                        oaa=overAllAccuracy(cm, afc, cts)
                                        hacc=ta
                                        mini_v_loss=valid_loss
                                        be=data.train.epochs_completed
                                        file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                            final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                            Epo=be, cBS=batchSize, BN=batchNorm,
                                            input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                                        if SaveModel and hacc>getsaveMV(Module, DataSet):
                                            ferm.save(model_save_path)
                                            print('Model saved!')
            elif Module==211:
                '''MODULE211---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('GEOWCPC: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                data_shape=[]
                data_shape.append(data.train.geo[0].shape)
                data_shape.append(data.train.wcs[0].shape)
                print('data shape: %s'%(str(data_shape)))
                
                ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                ####
                #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                #op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                if RADAM: 
                    op=OPT.RAdam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                else:
                    op=OPT.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                if LOOKAHEAD:
                    op=OPT.Lookahead(op, sync_period=5, slow_step=0.5)#
                cLR=0.001
                ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                
                if Mfit:
                    reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                    ccm=EvaCMSM(validation=([np.array(data.test.geo), np.array(data.test.wcs)], np.array(data.test.labels)), 
                                filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                    cbl=[reduce_lr, ccm]
                    if Summary:
                        history=ferm.fit(x=[np.array(data.train.geo), np.array(data.train.wcs)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    else:
                        history=ferm.fit(x=[np.array(data.train.geo), np.array(data.train.wcs)], y=np.array(data.train.labels), batch_size=None, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)
                    #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                    #exit()
                    cm=history.history['cm'][-1]
                    mini_v_loss=history.history['b_v_loss'][-1]
                    be=history.history['epoch'][-1]
                    floss=history.history['loss'][be-1]
                    hacc=history.history['b_v_acc'][-1]
                    loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                    clr=history.history['lr'][-1]
                    tt=time.time()
                    afc=[]
                    cts=[]
                    oaa=overAllAccuracy(cm, afc, cts)
                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                else:
                    iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                    clr=K.get_value(ferm.optimizer.lr)
                    for i in range(iters):
                        if (i+1)%lrstep==0 and ovt_drate<1:
                            clr=clr*ovt_drate
                            K.set_value(ferm.optimizer.lr, clr)
                        batch=data.train.next_batch(batchSize, shuffle=False)
                        if Summary:
                            tloss, tac=ferm.train_on_batch(x=[np.array(batch[1]), np.array(batch[7])], y=np.array(batch[5]))
                        else:
                            tloss, tac=ferm.train_on_batch(x=[np.array(batch[1]), np.array(batch[7])], y=np.array(batch[5]))
                        #print(ferm.metrics_names)
                        if tloss<mini_loss:
                            mini_loss=tloss
                        valid_loss, ta = ferm.evaluate(x=[np.array(data.test.geo), np.array(data.test.wcs)], y=np.array(data.test.labels), batch_size=None)
                        laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                        tt=time.time()
                        print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                        if laflag:
                            predy=ferm.predict(x=[np.array(data.test.geo), np.array(data.test.wcs)])
                            predl=np.argmax(predy, axis=1)
                            truel=np.argmax(data.test.labels, axis=1)
                            cm=confusion_matrix(y_true=truel, y_pred=predl)
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            hacc=ta
                            mini_v_loss=valid_loss
                            be=data.train.epochs_completed
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            if SaveModel and hacc>getsaveMV(Module, DataSet):
                                ferm.save(model_save_path)
                                print('\n\n\n\n\n\n\nModel saved!\n\n\n\n\n\n\n')
            elif Module==411:
                '''MODULE411---------------------------------------------------------------------------------------------------- 
                Options for the 
                -------------------------------------------------------------------------------------------------------------'''
                print('CFNBCN: %s'%(NetworkType))
                '''Here begins the implementation logic-------------------------------------------------------------------
                -------------------------------------------------------------------------------------------------------------'''
                if DataSet%10==9 or DataSet==690 or DataSet==4090 or DataSet==4190:
                    test_bat=3570
                elif DataSet%10==7 or DataSet%10==8:
                    test_bat=500
                else:
                    test_bat=300
                data_shape=[]
                data_shape.append(data.train.cropf[0].shape)
                data_shape.append(data.train.wcs[0].shape)
                print('data shape: %s'%(str(data_shape)))
                
                ferm=FERN.GetNetworkKV(data_shape, NetworkType, 'AE', cn, batchNorm, DataSet)
                plot_model(ferm, to_file='./modelfigures/M%dN%dD%dmodel_params-%d.png'%(Module,NetworkType,DataSet,ferm.count_params()), show_shapes=True)
                ####
                #op=OPT.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
                #op=OPT.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
                #op=OPT.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
                if DataSet%10<6 and DataSet%10>0:
                    ilr=0.0005
                elif DataSet%10==9:
                    ilr=0.0001
                    if NetworkType==4111:
                        ilr=0.00005
                elif DataSet==690 or DataSet==4090 or DataSet==4190:
                    ilr=0.00003
                    if NetworkType==4111:
                        ilr=0.00003
                else:
                    ilr=0.001
                print('Initial learning rate %f'%ilr)
                if RADAM: 
                    op=OPT.RAdam(lr=ilr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                else:
                    op=OPT.Adam(lr=ilr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)# default lr0.001 for runs12 and runs14, lr0.0001 for runs15 and runs16
                if LOOKAHEAD:
                    op=OPT.Lookahead(op, sync_period=5, slow_step=0.5)#
                cLR=ilr
                if DataSet%10==9:
                    ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                else:
                    ferm.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                
                if Mfit:
                    reduce_lr=ReduceLROnPlateau(moniter='loss', factor=0.5, patience=5, mode='auto', min_delta=0.0001)#Available metrics are: loss,acc,lr
                    ccm=EvaCMSM(validation=([np.array(data.test.cropf), np.array(data.test.wcs)], np.array(data.test.labels)), 
                                filepath=model_save_path, monitor='c_val_acc', verbose=1, save_model=SaveModel, save_best_only=True, Acc_Oriented=Acc_Oriented, tag='AE')
                    cbl=[reduce_lr, ccm]
                    if Summary:
                        history=ferm.fit(x=[np.array(data.train.cropf), np.array(data.train.wcs)], y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)#modified in 20191007, before modification batch_size=None
                    else:
                        history=ferm.fit(x=[np.array(data.train.cropf), np.array(data.train.wcs)], y=np.array(data.train.labels), batch_size=batchSize, epochs=Mini_Epochs, callbacks=cbl, shuffle=sf)#modified in 20191007, before modification batch_size=None
                    #print(history.history, history.history.keys()) #dict_keys(['loss', 'acc', 'lr', 'cm', 'epoch', 'b_v_loss', 'b_v_acc']) (in list type), the last three are added by EvaCMSM above.
                    #exit()
                    cm=history.history['cm'][-1]
                    mini_v_loss=history.history['b_v_loss'][-1]
                    be=history.history['epoch'][-1]
                    floss=history.history['loss'][be-1]
                    hacc=history.history['b_v_acc'][-1]
                    loss_a.preSetOutputList(lossl=history.history['c_val_loss'], accl=history.history['c_val_acc'], lr=history.history['lr'])
                    clr=history.history['lr'][-1]
                    tt=time.time()
                    afc=[]
                    cts=[]
                    oaa=overAllAccuracy(cm, afc, cts)
                    file_record = logfileKeras(file_record, runs=runs, OAA=oaa, TA=hacc, afc=afc, valid_loss=mini_v_loss,
                                final_train_loss=floss, TC=(tt-t1), ILR=cLR, FLR=clr, LS='ReduceLROnPlateau',
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                else:
                    iters=int((data.train.num_examples*Mini_Epochs)/batchSize)+1
                    clr=K.get_value(ferm.optimizer.lr)
                    for i in range(iters):
                        if (i+1)%lrstep==0 and ovt_drate<1:
                            clr=clr*ovt_drate
                            K.set_value(ferm.optimizer.lr, clr)
                        batch=data.train.next_batch(batchSize, shuffle=sf)
                        if Summary:
                            tloss, tac=ferm.train_on_batch(x=[np.array(batch[6]), np.array(batch[7])], y=np.array(batch[5]))
                        else:
                            tloss, tac=ferm.train_on_batch(x=[np.array(batch[6]), np.array(batch[7])], y=np.array(batch[5]))
                        #print(ferm.metrics_names)
                        if tloss<mini_loss:
                            mini_loss=tloss
                        valid_loss, ta = ferm.evaluate(x=[np.array(data.test.cropf), np.array(data.test.wcs)], y=np.array(data.test.labels), batch_size=test_bat)
                        laflag = loss_a.analyzeLossVariation(valid_loss, i, data.train.epochs_completed, ta)
                        tt=time.time()
                        print('LR:%.6f Ite:%05d Bs:%03d Epo:%03d Tac:%.3f Los:%.5f mLo:%.6f >>mVL: %.6f VL: %.6f VA: %f Best: %f T: %.1fs'%
                                (clr,i,batchSize,data.train.epochs_completed, tac, tloss, mini_loss, loss_a.minimun_loss, valid_loss, ta, loss_a.highestAcc, (tt-t1)))
                        if laflag:
                            predy=ferm.predict(x=[np.array(data.test.cropf), np.array(data.test.wcs)], batch_size=test_bat)
                            predl=np.argmax(predy, axis=1)
                            truel=np.argmax(data.test.labels, axis=1)
                            cm=confusion_matrix(y_true=truel, y_pred=predl)
                            afc=[]
                            cts=[]
                            oaa=overAllAccuracy(cm, afc, cts)
                            hacc=ta
                            mini_v_loss=valid_loss
                            be=data.train.epochs_completed
                            file_record = logfileKeras(file_record, runs=runs, OAA=oaa, afc=afc, TA=ta, valid_loss=mini_v_loss, valid_min_loss=loss_a.minimun_loss, 
                                final_train_loss=tloss, train_min_loss=mini_loss, TC=(tt-t1), LS=lrstep, ILR=cLR, FLR=clr, ites=i,
                                Epo=be, cBS=batchSize, BN=batchNorm,
                                input=sys.argv, CM=cm, T=time.localtime(tt), df=dfile)
                            if SaveModel and hacc>getsaveMV(Module, DataSet):
                                ferm.save(model_save_path)
                                print('\n\n\n\n\n\n\nModel saved!\n\n\n\n\n\n\n')
            if Acc_Oriented:
                newmodelname=model_save_path.replace('.h5','_Epoch%03d_ACC%f_MiniLoss%f_.h5'%(be, hacc, mini_v_loss))
            else:
                newmodelname=model_save_path.replace('.h5','_Epoch%03d_MiniLoss%f_ACC%f_.h5'%(be, mini_v_loss, hacc))
            if os.path.exists(model_save_path):
                os.rename((model_save_path),newmodelname)
            te=time.time()
            print('\n\nTotal time for all %d expressions: %fs\nOAA: %f\tTA: %f\n'%(cn, (te-t0), oaa, hacc))
            optimizerName=str(type(op).__name__)
            tt=time.time()
            log=log.replace('.txt',('_'+optimizerName+'.txt'))
            filelog=open(log,'a')

            print(log.split('.txt')[0])
            losslog=log.split('.txt')[0]+'_Runs%d_%d_%s'%(runs, TestID, timestamp)+'.validationlosslist'
            if losslog.find('/ovt/')>0:
                losslog=losslog.replace('./logs/ovt/','./logs/ovt/VL/')
            else:
                losslog=losslog.replace('./logs/KerasV/','./logs/KerasV/VL/')
            loss_a.outputlosslist(losslog)

            filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\tParameters: %d\n'%(file_record, (tt-t1), optimizerName, losslog, ferm.count_params()))
            print(log)
            filelog.close()
            '''ENDS---------------------------------------------------------------------------------------------'''
    except:
        try:
            tt=time.time()
            log=log.replace('.txt',('_'+optimizerName+'.txt'))
            filelog=open(log,'a')

            print(log.split('.txt')[0])
            losslog=log.split('.txt')[0]+'_Runs%d_%d_%d'%(runs, TestID)+'.validationlosslist'
            losslog=losslog.replace('./logs/ovt','./logs/ovt/VL')
            loss_a.outputlosslist(losslog)

            filelog.write('%s\t\t TotalTimeConsumed: %f\tOptimizer: %s\t%s\n'%(file_record, (tt-t1), optimizerName, losslog))
            print('\n\n>>>>>> Saving current run info after it crrupted or interrupted.\n\n')
            print(log)
            filelog.close()
            print('>>>>>> Current run info has been saved after it crrupted or interrupted.\n\n')
        except:
            print('ERROR: Fail to save current run info. after it crrupted')
        ferror=open(errorlog,'w')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()
    return
