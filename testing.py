####
# Platform windows 10 with Tensorflow 1.13.1, CUDA 10.0.130, python 3.7.1 64 bit, MSC v.1915 64bit,
###################
import collections
import os
import pickle
import sys, cv2
import time
import traceback
import warnings
import numpy as np
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import win_unicode_console
win_unicode_console.enable()

scale_factor=1.0/255.0
DPI=150
SaveImage=False
#FUSION=True#for module 11 with NewC as 2 or 3

##
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
def logfileKerasTEST(file_record, OAA=None, TA=None, valid_loss=None, 
                 TC=None, input=None, CM=None, T=None, df=None):
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
    file_record='OverAllACC: %s\tTA: %s\tvloss: %s\tTimeComsumed:%s\tInput:%s\t%s\tTime:%s\tDataFile:%s'%(
                                        str(OAA), str(TA), str(valid_loss), str(TC), str(input),str(CMS),T,df)
    return file_record

#############


##for data loading
Label_Dictionary={0:'angry', 1:'surprise', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'contempt'}
TESTDataset_Dictionary={
    3:'H:/Datasets/CK+107TEST_M1_M2_M3_M4_vs28.pkl',#from different group
    4:'H:/Datasets/KDEF6TEST_M1_M2_M3_M4_vs24.pkl',
    13:'H:/Datasets/CK+107TEST_M1_M2_M3Wavelet_M4_vs28.pkl',#from different group
    14:'H:/Datasets/KDEF6TEST_M1_M2_M3Wavelet_M4_vs24.pkl',
    23:'H:/Datasets/CK+107TEST2_M1_M2_M3_M4_vs12.pkl',#from different group
    24:'H:/Datasets/KDEF6TEST2_M1_M2_M3_M4_vs12.pkl',
    33:'H:/Datasets/CK+107TEST2_M2_M3WaL1C3_vs12.pkl',#for M23 #from different group
    34:'H:/Datasets/KDEF6TEST2_M2_M3WaL1C3_vs12.pkl',#for M23
    43:'H:/Datasets/CK+107TEST2_M2_M3WaL1C3_vs12.pkl',#for M211 #from different group
    44:'H:/Datasets/KDEF6TEST2_pair_M2_M11withoutWaveletPAIRDiff_vs12.pkl',#for M211
    }

def getModelName(Module, DataSet, NetworkType):
    modelname=None
    if Module==3:
        if DataSet==3:
            if NetworkType==1:
                modelname='J:/Models/saves/M3/D3/N1/D3_M3_N1_T0_R1_20190830123859_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch008_ACC1.000000_MiniLoss0.097159_.h5';
            elif NetworkType==2:
                modelname='J:/Models/saves/M3/D3/N2/D3_M3_N2_T0_R0_20190830130738_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch019_ACC1.000000_MiniLoss1.176664_.h5';
            elif NetworkType==3:
                modelname='J:/Models/saves/M3/D3/N3/D3_M3_N3_T0_R1_20190830134607_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch020_ACC1.000000_MiniLoss1.176708_.h5';
        elif DataSet==4:
            if NetworkType==1:
                modelname='J:/Models/saves/M3/D4/N1/D4_M3_N1_T0_R2_20190830130024_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch046_ACC0.958333_MiniLoss0.145303_.h5';
            elif NetworkType==2:
                modelname='J:/Models/saves/M3/D4/N2/D4_M3_N2_T0_R0_20190830132624_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch038_ACC0.944444_MiniLoss1.127547_.h5';
            elif NetworkType==3:
                modelname='J:/Models/saves/M3/D4/N3/D4_M3_N3_T0_R2_20190830140759_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch029_ACC0.944444_MiniLoss1.129744_.h5';
    elif Module==1:
        if DataSet==3:
            if NetworkType==0:
                modelname='J:/Models/saves/M1/D3/N0/D3_M1_N0_T0_R2_20190830143101_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch013_ACC0.969697_MiniLoss0.266888_.h5';
        elif DataSet==4:
            if NetworkType==0:
                modelname='J:/Models/saves/M1/D4/N0/D4_M1_N0_T0_R3_20190830150929_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch021_ACC0.916667_MiniLoss0.344600_.h5';
    elif Module==4:
        if DataSet==3:
            if NetworkType==0:
                modelname='J:/Models/saves/M4/D3/N0/D3_M4_N0_T0_R0_20190830151634_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch009_ACC1.000000_MiniLoss0.071121_.h5';
        elif DataSet==4:
            if NetworkType==0:
                modelname='J:/Models/saves/M4/D4/N0/D4_M4_N0_T0_R0_20190830153507_GPU1_noPretrain_newStopCriteriaV3_140epochs_updatedLoadingLogic_Epoch015_ACC0.944444_MiniLoss0.333165_.h5';
    elif Module==23:
        if DataSet==4:
            if NetworkType==2352202:
                modelname='J:/Models/saves/M23/D64/N2352202/D64_M23_N2352202_T0_R6_20190929192653_GPU0_noPretrain_newSCV3_120epochs_upDLoLo_Epoch046_ACC0.958333_MiniLoss0.166162_.h5'
    elif Module==211:
        if DataSet==4:
            if NetworkType==20138:
                modelname='J:/Models/saves/M211/D94/N20138/D94_M211_N20138_T0_R8_20190929193158_GPU1_noPretrain_newSCV3_120epochs_upDLoLo_Epoch020_ACC0.958333_MiniLoss0.300010_.h5'
    return modelname

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
def listShuffle(listV, ind):
    tm=listV[:]
    for i, v in enumerate(ind):
        tm[i]=listV[v]
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

                else:#modified in 20190303, 20190516
                    print('Original Data Type: %s, Transfering into Float64'%(str(self._cropf[0].dtype)))
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
                if loadImg or loadPat or loadInnerF:
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
                else:#modified in 20190516
                    print('Original Data Type: %s, Transfering into Float64'%(str(self._wcs[0].dtype)))
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

def loadtestdata_v4(datafilepath, validation_no=0, test_no=0, Module=0, Df=False, one_hot=True, reshape=False, cn=7):
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
        test_no=0
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

    elif Module==1:
        tl_rescaleimgs=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(imgs=ckplus10g[i]['imgs'], labels=ckplus10g[i]['labels'],
                                            num_Classes=cn,Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=False, loadImg=True, loadPat=False, loadInnerF=False)
           
    elif Module==2:
        tl_geo=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(geometry=ckplus10g[i]['geo'], labels=ckplus10g[i]['labels'],
                                            num_Classes=cn,Df=Df, one_hot=one_hot, reshape=reshape,
                                            loadGeo=True, loadImg=False, loadPat=False, loadInnerF=False)
            
    elif Module==4:
        tl_innerf=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],cropf=ckplus10g[i]['inner_face'], 
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=reshape,
                                            loadGeo=False, loadImg=False, loadPat=False, loadInnerF=True)
            
    elif Module==11:
        tl_wc=[]
        tl_labels=[]

        for i in range(nL):
            if i==test_no:
                test=DataSetFor3KindsDataV4(labels=ckplus10g[i]['labels'],wcs=ckplus10g[i]['wcs'], 
                                            one_hot=one_hot, num_Classes=cn,Df=Df,reshape=False,
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
        raise RuntimeError('ERROR: Unexpected Module in loadPKLData_V4')
  
    #return Datasets(train=train, test = test, validation = valid)
    return Datasets(train=None, test = test)
def loadAnonymousData(datafilepath):
    if(os.path.exists(datafilepath)):
        print('Loading data from file: %s'%datafilepath)
    else:
        raise FileNotFoundError('Cannot find the data file: %s'%datafilepath)
    with open(datafilepath,'rb') as datafile:
        fusiondata=pickle.load(datafile)
    return Datasets(train=DataSetForAnonymousData(X=fusiondata['trainX'], Y=fusiondata['trainY']), 
                                test=DataSetForAnonymousData(X=fusiondata['testX'], Y=fusiondata['testY']))
##########
#from https://github.com/xiaochus/VisualizationCNN
def normalize(x):
    """utility function to normalize a tensor by its L2 norm
    Args:
           x: gradient.
    Returns:
           x: gradient.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
def vis_conv(images, n, name, t):
    """visualize conv output and conv filter.
    Args:
           img: original image.
           n: number of col and row.
           t: vis type.
           name: save name.
    """
    size = 64
    margin = 5

    if t == 'filter':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin, 3))
    if t == 'conv':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin))

    for i in range(n):
        for j in range(n):
            if t == 'filter':
                filter_img = images[i + (j * n)]
            if t == 'conv':
                filter_img = images[..., i + (j * n)]
            filter_img = cv2.resize(filter_img, (size, size))

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            if t == 'filter':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
            if t == 'conv':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img

    # Display the results grid
    plt.imshow(results)
    plt.savefig('images\{}_{}.jpg'.format(t, name), dpi=600)
    plt.show()
def vis_heatmap(img, size, heatmap, heatmappath):
    """visualize heatmap.
    Args:
           img: original image.
           heatmap：heatmap.
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(221)
    plt.imshow(cv2.resize(img, size))
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(heatmap)
    plt.axis('off')

    plt.subplot(212)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(heatmappath, dpi=600)
    plt.show()
def output_heatmap(model, last_conv_layer, inputs):
    """Get the heatmap for image.
    Args:
           model: keras model.
           last_conv_layer: name of last conv layer in the model.
           inputs: processed input image.
    Returns:
           heatmap: heatmap.
    """
    # predict the image class
    preds = model.predict(inputs)
    # find the class index
    index = np.argmax(preds[0])
    # This is the entry in the prediction vector
    target_output = model.output[:, index]

    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

    # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads 
    # given the input picture
    iterate = K.function(inputs, [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(inputs)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap
##########
def getLayersKF(ferm):
    FL={}
    FL['name']=[]
    FL['function']=[]
    for i in range((len(ferm.layers)-1)):
        if ferm.layers[i].name.find('atten')>-1:
            continue
        if ferm.layers[i].name.find('input')>-1:
            continue
        if ferm.layers[i].name.find('ctiva')>-1:
            continue
        if ferm.layers[i].name.find('_BN_')>-1:
            continue
        if ferm.layers[i].name.find('cropping')>-1:
            continue
        if ferm.layers[i].name.find('batch')>-1:
            continue
        if ferm.layers[i].name.find('norm')>-1:
            continue
        if ferm.layers[i].name.find('Shape')>-1:
            continue
        if ferm.layers[i].name.find('rop')>-1:
            continue
        if ferm.layers[i].name.find('ncat')>-1:
            if ferm.layers[i].name=='concatenate_2':
                print('accept')
            else:
                continue
        if ferm.layers[i].name.find('DR')>-1:
            continue
        if ferm.layers[i].name.find('normal')>-1:
            continue
        if ferm.layers[i].name.find('fusion')>-1:
            if ferm.layers[i].name=='fusion_2':
                print('accept')
            else:
                continue
        FL['function'].append(K.function(ferm.inputs,[ferm.layers[i].output]))
        FL['name'].append(ferm.layers[i].name)
    return FL, len(FL['name'])
def unifyDataShape3D(src, s_shape):
    out=np.asarray(src).copy()
    channel=s_shape[-1]
    if s_shape[0]==1 and s_shape[1]==1:
        if channel==8:
            out=np.reshape(out, (4,2))
        elif channel==4:
            out=np.reshape(out, (2,2))
        elif channel==24:
            out=np.reshape(out, (6,4))
        elif channel==12:
            out=np.reshape(out, (4,3))
        elif channel==16:
            out=np.reshape(out, (4,4))
        elif channel==32:
            out=np.reshape(out, (8,4))
        elif channel==64:
            out=np.reshape(out, (8,8))
        elif channel==72:
            out=np.reshape(out, (8,9))
        elif channel==128:
            out=np.reshape(out, (16,8))
        elif channel==256:
            out=np.reshape(out, (16,16))
        elif channel==512:
            out=np.reshape(out, (32,16))
        else:
            out=out
    return out
def unifyDataShape1D(src, s_shape):
    out=np.asarray(src).copy()
    channel=s_shape[-1]
    if channel==1024:
        out=np.reshape(out, (32,32))
    elif channel==2048:
        out=np.reshape(out, (64,32))
    elif channel==8:
        out=np.reshape(out, (4,2))
    elif channel==4:
        out=np.reshape(out, (2,2))
    elif channel==24:
        out=np.reshape(out, (6,4))
    elif channel==64:
        out=np.reshape(out, (8,8))
    elif channel==128:
        out=np.reshape(out, (16,8))
    elif channel==256:
        out=np.reshape(out, (16,16))
    elif channel==420:
        out=np.reshape(out, (21,20))
    elif channel==512:
        out=np.reshape(out, (32,16))
    elif channel==4096:
        out=np.reshape(out, (64,64))
    elif channel==9216:
        out=np.reshape(out, (128,64))
    elif channel==1536:
        out=np.reshape(out, (48,32))
    elif channel==2176:
        out=np.reshape(out, (64,34))
    elif channel==288:
        out=np.reshape(out, (24,12))
    elif channel==216:
        out=np.reshape(out, (18,12))
    elif channel==384:
        out=np.reshape(out, (24,16))
    elif channel==12:
        out=np.reshape(out, (4,3))
    elif channel==300:
        out=np.reshape(out, (20,15))
    elif channel==310:
        out=np.reshape(out, (31,10))
    elif channel==124:
        out=np.reshape(out, (31,4))
    return out
def save_img(imgsrc, savename, color=False):
    plt.figure(1,(5,5),DPI)
    if color:
        plt.imshow(imgsrc)
    else:
        plt.imshow(imgsrc,cmap=plt.cm.gray)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
    plt.margins(0,0)
    plt.savefig(savename,dpi=DPI)

    img=Image.open(savename)
    imgV=ImageOps.invert(img)
    print(img.size, imgV.getbbox())
    img_c=img.crop(imgV.getbbox())
    img_c.save(savename)
    print('image saved in %s'%savename)
    plt.close()
####


def runKerasTest(GPU_Device_ID, 
        DataSet, NetworkType, Module, COLOR=False):

    try:
        '''GPU Option---------------------------------------------------------------------------------------------
        Determine which GPU is going to be used
        ------------------------------------------------------------------------------------------------------------'''
        print('GPU Option: %s'%(GPU_Device_ID))
        if (0==GPU_Device_ID) or (1==GPU_Device_ID):
            os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_Device_ID)
            errorlog='./logs/KerasTEST/ovt_errors_gpu'+str(GPU_Device_ID)+'.txt'
            templog='./logs/KerasTEST/ovt_templogs_newSC_gpu'+str(GPU_Device_ID)+'_M'+str(Module)+'_D'+str(DataSet)+'.txt'
        else:
            raise RuntimeError('Usage: python')
        
        '''GPU Option ENDS---------------------------------------------------------------------------------------'''
        
        t0=time.time()

        file_record=None
        t1=time.time()
        pic_save_path=''
        '''Input Data-------------------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------------------------------'''
        #
        ##data set loading
        #
        dfile=TESTDataset_Dictionary.get(DataSet, None)
        if dfile is None:
            raise RuntimeError('\nERROR: check the DataSet again. It must be one of %s'%(str(TESTDataset_Dictionary.keys())))
        
        cn=6
        if DataSet%10==3:
            cn=7
        data = loadtestdata_v4(dfile, 0, 0, Module=Module, cn=cn)
            
        tt=time.time()
        timestamp=time.strftime('%Y%m%d%H%M%S',time.localtime(t1))
        if COLOR:
            tag='Color'
        else:
            tag='Gray'
        if os.path.exists('J:/ModelTEST/'):
            pic_save_path='J:/ModelTEST/M%dD%dN%d/%s%s/'%(Module, DataSet, NetworkType,timestamp,tag)
            if not os.path.exists(pic_save_path):
                os.makedirs(pic_save_path)
        else:
            pic_save_path='./ModelTEST/M%dD%dN%d/%s%s/'%(Module, DataSet, NetworkType,timestamp,tag)
            if not os.path.exists(pic_save_path):
                os.makedirs(pic_save_path)
        log='%sGPU%d_M%d_D%d_N%d.txt'%(pic_save_path.replace('/%s%s/'%(timestamp,tag),'/'), GPU_Device_ID, Module, DataSet, NetworkType)
        print('Time used for loading data: %fs'%(tt-t1))

        modelname=getModelName(Module, DataSet%10, NetworkType)
        if modelname is None:
            print('\n\n\n\n\n\nERROR>>>>>Unexpected Setting.')
            exit(333)

        ferm=load_model(modelname)
        FL,fc=getLayersKF(ferm)
        print('Model loaded %s'%(modelname))
        '''Input Data Ends-----------------------------------------------------------------------------------------'''
        if Module==3:
            '''MODULE3---------------------------------------------------------------------------------------------------- 
            Options for the 
            -------------------------------------------------------------------------------------------------------------'''
            print('RCFN: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            finput=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp)]
        elif Module==1:
            '''MODULE1---------------------------------------------------------------------------------------------------- 
            Options for the 
            -------------------------------------------------------------------------------------------------------------'''
            print('OPNetwork: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            finput=[np.array(data.test.imgs)]
        elif Module==2:
            '''MODULE2---------------------------------------------------------------------------------------------------- 
            Options for the 
            -------------------------------------------------------------------------------------------------------------'''
            print('GeoNetwork: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #print(type(data.train.geo), type(data.train.geo[0]), len(data.train.geo[0]))
            finput=[np.array(data.test.geo)]
        elif Module==4:
            '''MODULE4---------------------------------------------------------------------------------------------------- 
            Options for the 
            -------------------------------------------------------------------------------------------------------------'''
            print('OPNetwork: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            finput=[np.array(data.test.cropf)]
               
        elif Module==23:
            '''MODULE23---------------------------------------------------------------------------------------------------- 
            Options for the 
            -------------------------------------------------------------------------------------------------------------'''
            print('LDGN-SRCFN Network: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            if NetworkType==2312201 or NetworkType==2352202:
                finput=[np.array(data.test.eyep), np.array(data.test.mouthp), np.array(data.test.geo)]
            else:
                finput=[np.array(data.test.eyep), np.array(data.test.foreheadp), np.array(data.test.mouthp), np.array(data.test.geo)]
            
        elif Module==211:
            '''MODULE211---------------------------------------------------------------------------------------------------- 
            Options for the 
            -------------------------------------------------------------------------------------------------------------'''
            print('BCN-LDGN Network: %s'%(NetworkType))
            '''Here begins the implementation logic-------------------------------------------------------------------
            -------------------------------------------------------------------------------------------------------------'''
            #print(type(data.train.geo), type(data.train.geo[0]), len(data.train.geo[0]))
            finput=[np.array(data.test.geo), np.array(data.test.wcs)]
        if SaveImage:
            for i in range(fc):
                print('\n\nProcessing layer: %s'%(FL['name'][i]))
                temout=FL['function'][i](finput)[0]
                sc=temout.shape[0]
                for j in range(sc):
                    psptem='%s%d/'%(pic_save_path,j+1)
                    if not os.path.exists(psptem):
                        os.makedirs(psptem)
                    s_shape=temout[j].shape
                    #print(s_shape)
                    if len(s_shape)==1:
                        try:
                            name='%d_%s_%d'%(i,FL['name'][i],nc)
                            savename='%s%s.jpg'%(psptem,name)
                            temImg=unifyDataShape1D(temout[j], s_shape)
                            save_img(temImg, savename, COLOR)
                        except:
                            einfo=open(savename.replace('.jpg','.txt'),'w')
                            info='Error in 1D saving %s\n\n'%(savename)
                            einfo.write(info)
                            print(info)
                            traceback.print_exc()
                            traceback.print_exc(file=einfo)
                            einfo.close()
                    elif len(s_shape)==3:
                        try:
                            if s_shape[0]==1 and s_shape[1]==1:
                                name='%d_%s_0'%(i,FL['name'][i])
                                savename='%s%s.jpg'%(psptem,name)
                                temImg=unifyDataShape3D(temout[j], s_shape)
                                save_img(temImg, savename, COLOR)
                            elif s_shape[0]==1 or s_shape[1]==1:
                                name='%d_%s_0'%(i,FL['name'][i])
                                savename='%s%s.jpg'%(psptem,name)
                                temoutimg=np.reshape(temout[j], (1,1,s_shape[0]*s_shape[1]*s_shape[2]))
                                temImg=unifyDataShape3D(temoutimg, (1,1,s_shape[0]*s_shape[1]*s_shape[2]))
                                save_img(temImg, savename, COLOR)
                            else:
                                for nc in range(s_shape[-1]):
                                    name='%d_%s_%d'%(i,FL['name'][i],nc)
                                    savename='%s%s.jpg'%(psptem,name)
                                    print(s_shape)
                                    temImg=unifyDataShape3D(temout[j,:,:,nc], s_shape)
                                    save_img(temImg, savename, COLOR)

                        except:
                            einfo=open(savename.replace('.jpg','.txt'),'w')
                            info='Error in 3D saving %s\n\n'%(savename)
                            einfo.write(info)
                            print(info)
                            traceback.print_exc()
                            traceback.print_exc(file=einfo)
                            einfo.close()

        plot_model(ferm, to_file='%s0_test_%s.png'%(pic_save_path,os.path.basename(modelname)), show_shapes=False)
        t1=time.time()
        valid_loss, ta = ferm.evaluate(x=finput, y=np.array(data.test.labels), batch_size=None)
        tt=time.time()
        print('VL: %.6f VA: %f T: %.1fs'%(valid_loss, ta, (tt-t1)))
        predy=ferm.predict(x=finput)
        predl=np.argmax(predy, axis=1)
        truel=np.argmax(data.test.labels, axis=1)
        cm=confusion_matrix(y_true=truel, y_pred=predl)
        afc=[]
        cts=[]
        oaa=overAllAccuracy(cm, afc, cts)
        hacc=ta
        mini_v_loss=valid_loss
        file_record = logfileKerasTEST(file_record, OAA=oaa, TA=ta, valid_loss=mini_v_loss, TC=(tt-t1),
            input=sys.argv, CM=cm, T=timestamp, df='%s  ModelName:%s'%(dfile, modelname))
        
        te=time.time()
        print('\n\nTotal time for all %d expressions: %fs\nOAA: %f\tTA: %f\n'%(cn, (te-t0), oaa, hacc))
        tt=time.time()
        filelog=open(log,'a')
        filelog.write('%s\tTotalTimeConsumed: %f\tParameters: %d\n'%(file_record, (tt-t1), ferm.count_params()))
        print(log)
        filelog.close()
        '''ENDS---------------------------------------------------------------------------------------------'''
    except:
        try:
            print('>>>>>> WArning.\n\n')
        except:
            print('ERROR: Fail to save current run info. after it crrupted')
        ferror=open(errorlog,'w')
        traceback.print_exc()
        traceback.print_exc(file=ferror)
        ferror.close()
    return
