####
#Source codes of the paper 'Facial Expression Recognition via Region-based Convolutional Fusion Network'
# submitted to Neurocomputing for the purpose of review.
# Platform windows 10 with Tensorflow 1.2.1, CUDA 8.0, python 3.5.2 64 bit, MSC v.1900 64bit
# Trimmed source file with testing.
###################
import ntpath
import pickle
import time
import numpy as np

import dlib
import FaceProcessUtil as fpu

#### Initiate ################################
tm=time.time()
SavePKL=True
LOG=False##inadaptive for this Module
M1=False##inadaptive for this Module
M2=True
M2Pair=False
M3=False##inadaptive for this Module
M3DT=1# data type, 0: original, 1: wavelet, 2: wavelet channel3(L1_C3)
M4=False##inadaptive for this Module
M11=True
M11DT=1#M11 data type, 0: _M11s, 1: _M11withoutWavelet, 3: _M11withLogarithm(bad performance)
RO=False#M11, whether to rotate the image 90 degree.

if LOG:
    import os, cv2

def initiate(labeltxt):
    datapath=''
    if labeltxt=='OuluCASIAVN6_pair':
        datapath = 'I:\Data\OuluCasIA\OriginalImg/VL'
        logdir='I:\Data\OuluCasIA\OriginalImg/VL/LogPatches'
    elif labeltxt.find('OuluCASIAVN6_FLIP')==0 or labeltxt.find('OuluCASIAVN6_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\OuluCASIAVN_FLIPSN'
        logdir='I:\Data\EnlargeData\OuluCASIAVN_FLIPSN/LogPatches'
    elif labeltxt=='OuluCASIANIR6_pair':
        datapath = 'I:\Data\OuluCasIA\OriginalImg/NI'
        logdir = 'I:\Data\OuluCasIA\OriginalImg/NI/LogPatches'
    elif labeltxt.find('OuluCASIANIR6_FLIP')==0 or labeltxt.find('OuluCASIANIR6_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\OuluCASIANIR_FLIPSN'
        logdir = 'I:\Data\EnlargeData\OuluCASIANIR_FLIPSN/LogPatches'
    elif labeltxt=='KDEF6_pair' or labeltxt=='KDEF6TEST2_pair':
        datapath = 'I:\Data\KDEF_G'
        logdir = 'I:\Data\KDEF_G/LogPatches'
    elif labeltxt=='KDEF6TEST_pair':
        datapath = 'I:/Python/Learning/findtestingsamples/KDEF6TEST'
        logdir = 'I:/Python/Learning/findtestingsamples/KDEF6TEST/LogPatches'
    elif labeltxt.find('KDEF6_FLIP')==0 or labeltxt.find('KDEF6_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\KDEF_G_FLIPSN'
        logdir = 'I:\Data\EnlargeData\KDEF_G_FLIPSN/LogPatches'
    elif labeltxt=='CK+106_pair':
        datapath = 'I:\Data\CK+\CKplus10G'
        logdir = 'I:\Data\CK+\CKplus10G/LogPatches'
    elif labeltxt.find('CK+106_FLIP')==0 or labeltxt.find('CK+106_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\CKplus10G_FLIPSN'
        logdir = 'I:\Data\EnlargeData\CKplus10G_FLIPSN/LogPatches'
    elif labeltxt=='CK+107_pair' or labeltxt=='CK+107TEST2_pair':
        datapath = 'I:/Data/CK+/CK+10GwithContempt'
        logdir = 'I:/Data/CK+/CK+10GwithContempt/LogPatches'
    elif labeltxt=='CK+107TEST_pair':
        datapath = 'I:/Python/Learning/findtestingsamples/CK+107TEST'
        logdir = 'I:/Python/Learning/findtestingsamples/CK+107TEST/LogPatches'
    elif labeltxt.find('CK+107_FLIP')==0 or labeltxt.find('CK+107_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\CK+10GwithContempt_FLIPSN'
        logdir = 'I:\Data\EnlargeData\CK+10GwithContempt_FLIPSN/LogPatches'
    elif labeltxt=='CK+86_pair':
        datapath = 'I:\Data\CK+\CKplus8G'
        logdir = 'I:\Data\CK+\CKplus8G/LogPatches'
    elif labeltxt.find('CK+86_FLIP')==0 or labeltxt.find('CK+86_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\CKplus8G_FLIPSN'
        logdir = 'I:\Data\EnlargeData\CKplus8G_FLIPSN/LogPatches'
    else:
        raise RuntimeError('Unexpected case while initialize datapath')

    print('Data from %s'%(datapath))
    return (labeltxt+'.txt'), datapath, logdir
def getTag():
    tag=''
    if M1:
        tag=tag+'_M1'
    if M2:
        if M2Pair:
            tag=tag+'_M2PAIRDiff'
        else:
            tag=tag+'_M2'
    if M3:
        tag=tag+'_M3'
        if M3DT==1:
            tag=tag+'Wavelet'
        elif M3DT==2:
            tag=tag+'WaL1C3'
    if M4:
        tag=tag+'_M4'
    if M11:
        if M11DT==0:
            tag=tag+'_M11sPAIRDiff'
        elif M11DT==1:
            tag=tag+'_M11withoutWaveletPAIRDiff'
        elif M11DT==3:
            tag=tag+'_M11withLogarithmPAIRDiff'
        if RO:
            tag=tag+'Rotate90'
        
    return tag

def preprocess(labeltxt):
    global tm
    errorcount=0
    ec=0
    labelfile, datapath, logdir=initiate(labeltxt)
    if not ntpath.exists(logdir) and LOG:
        os.makedirs(logdir)
    tag=getTag()
    fi = open(labelfile)
    flist = fi.readlines()
    print('Total records: %d'%len(flist))
    ipl=[]
    lal=[]
    for fn in flist:
        n, l=fn.replace('\n','').split(' ')
        ipl.append(n)
        lal.append(int(l))
    print(len(ipl))
    #################
    ##### Prepare images ##############################
    cums=[]
    imglist=[]
    labellist=[]

    if labeltxt.find('CASIA')>0:
        groupcount=len(ipl)//10
        gc=0
        for i in range(10):
            cums.append(groupcount)
            imglist.append(ipl[i*groupcount:i*groupcount+groupcount])
            labellist.append(lal[i*groupcount:i*groupcount+groupcount])
    else:
        afl=[]
        lab=[]
        groups=None
        gc=0
        for i, v in enumerate(ipl):
            cg=v.split('/')[0]
            if groups==None:
                groups=cg
                afl.append(v)
                lab.append(lal[i])
                gc=gc+1
            elif groups==cg:
                afl.append(v)
                lab.append(lal[i])
                gc=gc+1
            else:
                #print('\n\nGroup %s\n'%(groups))
                #print(afl)
                imglist.append(afl)
                labellist.append(lab)
                cums.append(gc)
                gc=1
                afl=[]
                lab=[]
                groups=cg
                afl.append(v)
                lab.append(lal[i])
        #print('\n\n\nGroup %s\n'%(groups))
        #print(afl)
        imglist.append(afl)
        labellist.append(lab)
        cums.append(gc)

        
    #print(imglist)
    print(cums)
    print(len(imglist))
    print("Total training images:%d"%(sum(cums)))

    count=0
    vc=0
    feature_group_of_subject=[]
    for id in range(len(imglist)):
        #if id < 2:
        #    continue
        print("\nProcess Group %d>>>>>>>>\n"%(id+1))
        imagelist=imglist[id]
        lablist=labellist[id]
        imgs=[]
        cropfaces=[]
        eyepath=[]
        foreheadpath=[]
        mouthpatch=[]
        ckplus_label=[]
        geo=[]
        wcsl=[]
        gc=0
        for i, v in enumerate(imagelist):
            image_path1, image_path2=v.split('\t')
            tm1=time.time()
            print("\n>%d Prepare image "%(i)+image_path1 + ":")
            imname = ntpath.basename(image_path1)
            image_path1=datapath+'/'+image_path1
            image_path2=datapath+'/'+image_path2
            label=lablist[i]
            print("Name: %s            Label: %s"%(imname, label))
            count=count+1

            flag1, img1=fpu.calibrateImge(image_path1)
            flag2, img2=fpu.calibrateImge(image_path2)
            if flag1 and flag2:
                wl=False
                wp=False
                if M2:
                    wl=True
                if M3 or M4 or M11:
                    wp=True
                imgr1 = fpu.getLandMarkFeatures_and_ImgPatches(img1, wl, wp, True)
                imgr2 = fpu.getLandMarkFeatures_and_ImgPatches(img2, wl, wp, True)

            else:
                errorcount=errorcount+1
                print('Unexpected case while calibrating for:'+str(image_path1))
                continue
                #raise RuntimeError('Unexpected case while calibrating for:'+str(image_path1))
            if M2 and not imgr1[1]:
                raise RuntimeError('Unexpected case while calibrating for:'+str(image_path1))
            if M3 and not imgr1[3]:
                continue

            if imgr1[3] and M3 and imgr2[3]:
                if M3DT==0:
                    print("Get Patches>>>>>>>>>>>>>>")
                    eyepath.append(imgr1[4])
                    foreheadpath.append(imgr1[5])
                    mouthpatch.append(imgr1[6])
                elif M3DT==1:
                    print("Get wavelet Patches>>>>>>>>>>>>>>")
                    eyepath.append(fpu.getWaveletData(imgr1[4]/255.0))
                    foreheadpath.append(fpu.getWaveletData(imgr1[5]/255.0))
                    mouthpatch.append(fpu.getWaveletData(imgr1[6]/255.0))
                elif M3DT==2:
                    mo='periodization'
                    if labeltxt.find('CASIA')>0:
                        mo='constant'
                    print("Get wavelet Patches>>>>>>>>>>>>>>")
                    eyepath.append(fpu.get_WaveletDataC3(imgr1[4]/255.0, mo=mo))
                    foreheadpath.append(fpu.get_WaveletDataC3(imgr1[5]/255.0, mo=mo))
                    mouthpatch.append(fpu.get_WaveletDataC3(imgr1[6]/255.0, mo=mo))
                else:
                    raise RuntimeWarning('Unexperted M3 data type. Please check the M3DT again with correct setting.')
            if M1:
                imgs.append(imgr1[0])
                print('Get recale image>>>>>>>>>>>')
            if M2:
                
                if imgr1[1] and imgr2[1] and M2Pair:
                    #print(type(np.array(imgr1[2])), np.array(imgr1[2]).shape)
                    #exit()
                    geo.append((np.array(imgr1[2])-np.array(imgr2[2])))
                    print('Get geometry features difference>>>>>>>>>>>')
                elif not M2Pair and imgr1[2]:
                    geo.append(np.array(imgr1[2]))
                    print('Get geometry features>>>>>>>>>>>')
                else:
                    #fw=open('M2_exception_log.txt','a')
                    #fw.write('%s\n'%v)
                    #fw.close()
                    continue
            if M4:
                cropfaces.append(imgr1[7])
                print('Get cropped face>>>>>>>>>>>')
            if M11:
                if imgr1[7] is None:
                    ec=ec+1
                    #raise RuntimeError('Unexpected error in data preprocesing.')
                    print('Unexpected error in data preprocesing.')
                    continue
                #else:
                #    continue
                if RO:
                    r,c=imgr1[7].shape
                    rm=cv2.getRotationMatrix2D((c//2, r//2), 90, 1.0)
                    imgr1[7]=cv2.warpAffine(imgr1[7], rm, (c, r))
                twcs1, wcs=fpu.getWaveletComplexShiftData(imgr1[7]/255.0, log=LOG, Type=M11DT)
                twcs2, wcs=fpu.getWaveletComplexShiftData(imgr2[7]/255.0, log=LOG, Type=M11DT)
                wcsl.append((twcs1-twcs2))
                print('Get WCS data>>>>>>>>>>>>>')
            if LOG:
                newname=logdir+'/'+imname
                if M1:
                    cv2.imwrite(newname+'_rescale.jpg', imgr1[0])
                if M4:
                    cv2.imwrite(newname+'_crop.jpg',imgr1[7])
                if M3:
                    cv2.imwrite(newname+'_eye.jpg',imgr1[4])
                    cv2.imwrite(newname+'_forehead.jpg',imgr1[5])
                    cv2.imwrite(newname+'_mouth.jpg', imgr1[6])
                if M11:
                    newname=logdir+'/'+imname
                    cv2.imwrite(newname+'_cropwcs.jpg', wcs)

            ckplus_label.append(label)
            gc=gc+1

            tm2=time.time()
            dtm=tm2-tm1
            print("Time comsuming: %f"%(dtm))
        print("Group %d has %d samples"%(id+1, gc))
        vc=vc+gc
        if SavePKL:
            ckplus={}
            ckplus['labels']=ckplus_label
            ckplus['imgs']=imgs
            ckplus['geo']=geo
            ckplus['eye_patch']=eyepath
            ckplus['middle_patch']=foreheadpath
            ckplus['mouth_patch']=mouthpatch
            ckplus['inner_face']=cropfaces
            ckplus['wcs']=wcsl

            feature_group_of_subject.append(ckplus)
    if SavePKL:
        if labeltxt.find('/')>0:
            labeltxt=labeltxt.split('/')[-1]
        filenametosave='H:/Datasets/%s%s_vs%d.pkl'%(labeltxt,tag,vc)
        with open(filenametosave,'wb') as fin:
            pickle.dump(feature_group_of_subject,fin,4)
        print('File saved: %s'%(filenametosave))
    tm2=time.time()
    dtm=tm2-tm
    print("Total time comsuming: %fs for %d images"%(dtm, count))
    print(ec, errorcount)

datasetlist=[]
#datasetlist.append('CK+107TEST_pair')
#datasetlist.append('KDEF6TEST_pair')
#datasetlist.append('CK+107TEST2_pair')
#datasetlist.append('KDEF6TEST2_pair')

datasetlist.append('CK+86_pair')
datasetlist.append('CK+106_pair')
datasetlist.append('CK+107_pair')
datasetlist.append('KDEF6_pair')
datasetlist.append('OuluCASIAVN6_pair')
#datasetlist.append('OuluCASIANIR6_pair')

#datasetlist.append('CK+86_FLIP_enlarge_pair')#2 times
#datasetlist.append('CK+106_FLIP_enlarge_pair')#2 times
#datasetlist.append('CK+107_FLIP_enlarge_pair')#2 times
#datasetlist.append('KDEF6_FLIP_enlarge_pair')#2 times
#datasetlist.append('OuluCASIAVN6_FLIP_enlarge_pair')#2 times
#datasetlist.append('OuluCASIANIR6_FLIP_enlarge_pair')#2 times

#datasetlist.append('CK+86_FLIPNS_pair')#22 times
#datasetlist.append('CK+106_FLIPNS_pair')#22 times
#datasetlist.append('CK+107_FLIPNS_pair')#22 times
#datasetlist.append('KDEF6_FLIPNS_pair')#22 times
#datasetlist.append('OuluCASIAVN6_FLIPNS_pair')#22 times
#datasetlist.append('OuluCASIANIR6_FLIPNS_pair')#22 times

#datasetlist.append('CK+86_FLIPNST10_pair')#22 times
#datasetlist.append('CK+106_FLIPNST10_pair')#22 times
#datasetlist.append('CK+107_FLIPNST10_pair')#22 times
#datasetlist.append('KDEF6_FLIPNST10_pair')#22 times
#datasetlist.append('OuluCASIAVN6_FLIPNST10_pair')#22 times
#datasetlist.append('OuluCASIANIR6_FLIPNST10_pair')#22 times

#datasetlist.append('I:/Data/EnlargeData/CK+10GwithContempt_FLIPSN/CK+107_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_pair')
#datasetlist.append('I:/Data/EnlargeData/CKplus8G_FLIPSN/CK+86_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_pair')
#datasetlist.append('I:/Data/EnlargeData/CKplus10G_FLIPSN/CK+106_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_pair')
#datasetlist.append('I:/Data/EnlargeData/KDEF_G_FLIPSN/KDEF6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_pair')
#datasetlist.append('I:/Data/EnlargeData/OuluCASIAVN_FLIPSN/OuluCASIAVN6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_pair')
#datasetlist.append('I:/Data/EnlargeData/OuluCASIANIR_FLIPSN/OuluCASIANIR6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10_pair')

for i in range(len(datasetlist)):
    preprocess(datasetlist[i])
