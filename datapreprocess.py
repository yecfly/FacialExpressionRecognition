####
#Source codes of the paper 'Facial Expression Recognition via Region-based Convolutional Fusion Network'
# submitted to Neurocomputing for the purpose of review.
# Platform windows 10 with Tensorflow 1.13.1, CUDA 10.0.130, python 3.7.1 64 bit, MSC v.1915 64bit,
# Trimmed source file with testing.
###################
import ntpath
import pickle
import time, cv2, traceback
import numpy as np

import dlib
import FaceProcessUtil as fpu

#### Initiate ################################
tm=time.time()
SavePKL=True
LOG=False
M1=True
M2=True
M3=True
M3DT=0# data type, 0: original, 1: wavelet, 2: wavelet channel3(L1_C3)
M4=True
M11=True
M11DT=1#M11 data type, 0: _M11s, 1: _M11withoutWavelet, 3: _M11withLogarithm
RO=False#M11, whether to rotate the image 90 degree.

if LOG:
    import os


H_e_ratio=0.29316#0.26316
ehs=0.05#0.1
H_m_ratio=0.3859
W_m_ratio=0.74
H_f_ratio=0.27
W_f_ratio=0.347
def defaultPatchForSFEWv1(img):
    h,w=img.shape

    rhe=int(h*H_e_ratio)
    rwe=w
    xe=0
    ye=int(h*ehs)
    eyeP=np.zeros((rhe, rwe), dtype = 'uint8')
    eyeP[0:rhe,0:rwe]=img[ye:ye+rhe,xe:xe+rwe]
    eyeP=cv2.resize(eyeP,fpu.eye_patch_size)

    rhm=int(h*H_m_ratio)
    rwm=int(w*W_m_ratio)
    xm=int((w-rwm)/2)
    ym=int(h*0.51)
    mouP=np.zeros((rhm, rwm), dtype = 'uint8')
    mouP[0:rhm,0:rwm]=img[ym:ym+rhm, xm:xm+rwm]
    mouP=cv2.resize(mouP, fpu.mouth_patch_size)

    rhf=int(h*H_f_ratio)
    rwf=int(w*W_f_ratio)
    xf=int((w-rwf)/2)
    yf=0
    foreP=np.zeros((rhf,rwf), dtype = 'uint8')
    foreP[0:rhf,0:rwf]=img[yf:yf+rhf, xf:xf+rwf]
    foreP=cv2.resize(foreP, fpu.middle_patch_size)
    return eyeP, foreP, mouP

def initiate(labeltxt):
    datapath=''
    if labeltxt=='OuluCASIAVN6':
        datapath = 'I:\Data\OuluCasIA\OriginalImg/VL'
        logdir='I:\Data\OuluCasIA\OriginalImg/VL/LogPatches'
    elif labeltxt.find('OuluCASIAVN6_FLIP')==0 or labeltxt.find('OuluCASIAVN6_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\OuluCASIAVN_FLIPSN'
        logdir='I:\Data\EnlargeData\OuluCASIAVN_FLIPSN/LogPatches'
    elif labeltxt=='OuluCASIANIR6':
        datapath = 'I:\Data\OuluCasIA\OriginalImg/NI'
        logdir = 'I:\Data\OuluCasIA\OriginalImg/NI/LogPatches'
    elif labeltxt.find('OuluCASIANIR6_FLIP')==0 or labeltxt.find('OuluCASIANIR6_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\OuluCASIANIR_FLIPSN'
        logdir = 'I:\Data\EnlargeData\OuluCASIANIR_FLIPSN/LogPatches'
    elif labeltxt=='KDEF6':
        datapath = 'I:\Data\KDEF_G'
        logdir = 'I:\Data\KDEF_G/LogPatches'
    elif labeltxt=='KDEF6TEST' or labeltxt=='KDEF6TEST2':
        datapath = 'I:/Python/Learning/findtestingsamples/KDEF6TEST'
        logdir = 'I:/Python/Learning/findtestingsamples/KDEF6TEST/LogPatches'
    elif labeltxt.find('KDEF6_FLIP')==0 or labeltxt.find('KDEF6_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\KDEF_G_FLIPSN'
        logdir = 'I:\Data\EnlargeData\KDEF_G_FLIPSN/LogPatches'
    elif labeltxt=='CK+106':
        datapath = 'I:\Data\CK+\CKplus10G'
        logdir = 'I:\Data\CK+\CKplus10G/LogPatches'
    elif labeltxt.find('CK+106_FLIP')==0 or labeltxt.find('CK+106_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\CKplus10G_FLIPSN'
        logdir = 'I:\Data\EnlargeData\CKplus10G_FLIPSN/LogPatches'
    elif labeltxt=='CK+107':
        datapath = 'I:/Data/CK+/CK+10GwithContempt'
        logdir = 'I:/Data/CK+/CK+10GwithContempt/LogPatches'
    elif labeltxt=='CK+107TEST' or labeltxt=='CK+107TEST2':
        datapath = 'I:/Python/Learning/findtestingsamples/CK+107TEST'
        logdir = 'I:/Python/Learning/findtestingsamples/CK+107TEST/LogPatches'
    elif labeltxt.find('CK+107_FLIP')==0 or labeltxt.find('CK+107_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\CK+10GwithContempt_FLIPSN'
        logdir = 'I:\Data\EnlargeData\CK+10GwithContempt_FLIPSN/LogPatches'
    elif labeltxt=='CK+86':
        datapath = 'I:\Data\CK+\CKplus8G'
        logdir = 'I:\Data\CK+\CKplus8G/LogPatches'
    elif labeltxt.find('CK+86_FLIP')==0 or labeltxt.find('CK+86_FLIP')>0:
        datapath = 'I:\Data\EnlargeData\CKplus8G_FLIPSN'
        logdir = 'I:\Data\EnlargeData\CKplus8G_FLIPSN/LogPatches'
    elif labeltxt=='SFEW2.0_7' or labeltxt=='SFEW2_7v2':
        datapath = 'I:/Data/SFEW 2.0'
        logdir='I:/Data/SFEW 2.0/logPreImg'
        fn=[]
        fn.append((labeltxt.replace('_','_Val')+'.txt'))
        fn.append((labeltxt.replace('_','_Train')+'.txt'))
        return fn, datapath, logdir
    else:
        raise RuntimeError('Unexpected case while initialize datapath')

    print('Data from %s'%(datapath))
    return (labeltxt+'.txt'), datapath, logdir
def getTag(SFEW=False):
    tag=''
    if M1:
        tag=tag+'_M1'
        if M3DT==1:
            tag=tag+'Wavelet'
        elif M3DT==2:
            tag=tag+'WaL1C3'
    if M2 and not SFEW:
        tag=tag+'_M2'
    if M3:
        tag=tag+'_M3'
        if M3DT==1:
            tag=tag+'Wavelet'
        elif M3DT==2:
            tag=tag+'WaL1C3'
    if M4:
        tag=tag+'_M4'
        if M3DT==1:
            tag=tag+'Wavelet'
        elif M3DT==2:
            tag=tag+'WaL1C3'
    if M11:
        if M11DT==0:
            tag=tag+'_M11s'
        elif M11DT==1:
            tag=tag+'_M11withoutWavelet'
        elif M11DT==3:
            tag=tag+'_M11withLogarithm'
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
    
    if type(labelfile) is list:
        cums=[]
        imglist=[]
        labellist=[]
        for fvn in labelfile:
            fv=open(fvn)
            fvl=fv.readlines()
            print('\n\n\n######Total samples %d for %s'%(len(fvl), fvn))
            ipl=[]
            lal=[]
            for fr in fvl:
                n, l=fr.replace('\n','').split(' ')
                ipl.append(n)
                lal.append(int(l))
            cums.append(len(lal))
            imglist.append(ipl)
            labellist.append(lal)
        if labeltxt=='SFEW2.0_7':
            tag=getTag(SFEW=True)
            tag=tag+'_TestID_0_TrainID_1'
        else:
            tag=getTag()
    else:
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
        #if labeltxt=='OuluCASIAVN6' or labeltxt=='OuluCASIANIR6':
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
            image_path=v
            tm1=time.time()
            print("\n>%d Prepare image "%(i)+image_path + ":")
            imname = ntpath.basename(image_path)
            image_path=datapath+'/'+image_path
            label=lablist[i]
            count=count+1
            print("Name: %s            Label: %s"%(imname, label))
            if labeltxt=='SFEW2.0_7' or labeltxt=='SFEW2_7v2':
                tmimg=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                tr,tc=tmimg.shape
                print(tr,tc)
                if tr>450:
                    flag, img=fpu.calibrateImge(image_path)
                    if flag:
                        wl=False
                        wp=False
                        if M3 or M4 or M11:
                            wp=True
                        imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, wl, wp, True)
                    else:
                        errorcount=errorcount+1
                        print('Unexpected case while calibrating for:'+str(image_path))
                        raise RuntimeWarning(image_path)

                    if M3 and not imgr[3]:
                        raise RuntimeWarning('Cannot extract patches after calibrating for %s'%image_path)

                    if imgr[3] and M3:
                        if M3DT==0:
                            print("Get Patches>>>>>>>>>>>>>>")
                            eyepath.append(imgr[4])
                            foreheadpath.append(imgr[5])
                            mouthpatch.append(imgr[6])
                        elif M3DT==1:
                            print("Get wavelet Patches>>>>>>>>>>>>>>")
                            eyepath.append(fpu.getWaveletData(imgr[4]/255.0))
                            foreheadpath.append(fpu.getWaveletData(imgr[5]/255.0))
                            mouthpatch.append(fpu.getWaveletData(imgr[6]/255.0))
                        elif M3DT==2:
                            mo='periodization'
                            if labeltxt.find('CASIA')>0:
                                mo='constant'
                            print("Get wavelet Patches>>>>>>>>>>>>>>")
                            eyepath.append(fpu.get_WaveletDataC3(imgr[4]/255.0, mo=mo))
                            foreheadpath.append(fpu.get_WaveletDataC3(imgr[5]/255.0, mo=mo))
                            mouthpatch.append(fpu.get_WaveletDataC3(imgr[6]/255.0, mo=mo))
                        else:
                            raise RuntimeWarning('Unexperted in M3 data type. Please check the M3DT again with correct setting.')
                    if M1:
                        if M3DT==0:
                            imgs.append(imgr[7])
                        elif M3DT==1:
                            imgs.append(fpu.getWaveletData(imgr[7]/255.0))
                        elif M3DT==2:
                            mo='periodization'
                            if labeltxt.find('CASIA')>0:
                                mo='constant'
                            imgs.append(fpu.get_WaveletDataC3(imgr[7]/255.0, mo=mo))
                        else:
                            raise RuntimeWarning('Unexperted in M1 data type. Please check the M3DT again with correct setting.')
                        
                        print('Get recale image>>>>>>>>>>>')
                    if M4:
                        if M3DT==0:
                            cropfaces.append(imgr[7])
                        elif M3DT==1:
                            cropfaces.append(fpu.getWaveletData(imgr[7]/255.0))
                        elif M3DT==2:
                            mo='periodization'
                            if labeltxt.find('CASIA')>0:
                                mo='constant'
                            cropfaces.append(fpu.get_WaveletDataC3(imgr[7]/255.0, mo=mo))
                        else:
                            raise RuntimeWarning('Unexperted in M4 data type. Please check the M3DT again with correct setting.')
                        print('Get cropped face>>>>>>>>>>>')
                    if M11:
                        if imgr[7] is None:
                            ec=ec+1
                            raise RuntimeWarning('Unexpected error in saving M11 data.')
                        twcs, wcs=fpu.getWaveletComplexShiftData(imgr[7]/255.0, log=LOG, Type=M11DT)
                        wcsl.append(twcs)
                        print('Get WCS data>>>>>>>>>>>>>')
                    if LOG:
                        newname=logdir+'/'+imname
                        try:
                            if M1:
                                cv2.imwrite(newname+'_rescale.jpg', imgr[0])
                            if M4:
                                cv2.imwrite(newname+'_crop.jpg',imgr[7])
                            if M3:
                                cv2.imwrite(newname+'_eye.jpg',imgr[4])
                                cv2.imwrite(newname+'_forehead.jpg',imgr[5])
                                cv2.imwrite(newname+'_mouth.jpg', imgr[6])
                            if M11:
                                newname=logdir+'/'+imname
                                cv2.imwrite(newname+'_cropwcs.jpg', wcs)
                        except:
                            traceback.print_exc()
                else:
                    if M1:
                        if M3DT==0:
                            imgs.append(cv2.resize(tmimg,fpu.inner_face_size))
                        elif M3DT==1:
                            imgs.append(fpu.getWaveletData(cv2.resize(tmimg,fpu.inner_face_size)/255.0))
                        elif M3DT==2:
                            mo='periodization'
                            if labeltxt.find('CASIA')>0:
                                mo='constant'
                            imgs.append(fpu.get_WaveletDataC3(cv2.resize(tmimg,fpu.inner_face_size)/255.0, mo=mo))
                        else:
                            raise RuntimeWarning('Unexperted in M1 data type. Please check the M3DT again with correct setting.')
                        print('Get recale image>>>>>>>>>>>')
                    if M4:
                        if M3DT==0:
                            cropfaces.append(cv2.resize(tmimg,fpu.inner_face_size))
                        elif M3DT==1:
                            cropfaces.append(fpu.getWaveletData(cv2.resize(tmimg,fpu.inner_face_size)/255.0))
                        elif M3DT==2:
                            mo='periodization'
                            if labeltxt.find('CASIA')>0:
                                mo='constant'
                            cropfaces.append(fpu.get_WaveletDataC3(cv2.resize(tmimg,fpu.inner_face_size)/255.0, mo=mo))
                        else:
                            raise RuntimeWarning('Unexperted in M4 data type. Please check the M3DT again with correct setting.')
                        print('Get cropped face>>>>>>>>>>>')
                    if M3:
                        epimg, fpimg, mpimg=defaultPatchForSFEWv1(tmimg)
                        if M3DT==0:
                            print("Get Patches>>>>>>>>>>>>>>")
                            eyepath.append(epimg)
                            foreheadpath.append(fpimg)
                            mouthpatch.append(mpimg)
                        elif M3DT==1:
                            print("Get wavelet Patches>>>>>>>>>>>>>>")
                            eyepath.append(fpu.getWaveletData(epimg/255.0))
                            foreheadpath.append(fpu.getWaveletData(fpimg/255.0))
                            mouthpatch.append(fpu.getWaveletData(mpimg/255.0))
                        elif M3DT==2:
                            mo='periodization'
                            if labeltxt.find('CASIA')>0:
                                mo='constant'
                            print("Get wavelet Patches>>>>>>>>>>>>>>")
                            eyepath.append(fpu.get_WaveletDataC3(epimg/255.0, mo=mo))
                            foreheadpath.append(fpu.get_WaveletDataC3(fpimg/255.0, mo=mo))
                            mouthpatch.append(fpu.get_WaveletDataC3(mpimg/255.0, mo=mo))
                        else:
                            raise RuntimeWarning('Unexperted M3 data type. Please check the M3DT again with correct setting.')
                    if M11:
                        twcs, wcs=fpu.getWaveletComplexShiftData(cv2.resize(tmimg,fpu.inner_face_size)/255.0, log=LOG, Type=M11DT)
                        wcsl.append(twcs)
                        print('Get WCS data>>>>>>>>>>>>>')
            else:
                flag, img=fpu.calibrateImge(image_path)
                if flag:
                    wl=False
                    wp=False
                    if M2:
                        wl=True
                    if M3 or M4 or M11:
                        wp=True
                    imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, wl, wp, True)
                else:
                    errorcount=errorcount+1
                    print('Unexpected case while calibrating for:'+str(image_path))
                    continue
                    #raise RuntimeError('Unexpected case while calibrating for:'+str(image_path))
                if M2 and not imgr[1]:
                    continue
                if M3 and not imgr[3]:
                    continue

                if imgr[3] and M3:
                    if M3DT==0:
                        print("Get Patches>>>>>>>>>>>>>>")
                        eyepath.append(imgr[4])
                        foreheadpath.append(imgr[5])
                        mouthpatch.append(imgr[6])
                    elif M3DT==1:
                        print("Get wavelet Patches>>>>>>>>>>>>>>")
                        eyepath.append(fpu.getWaveletData(imgr[4]/255.0))
                        foreheadpath.append(fpu.getWaveletData(imgr[5]/255.0))
                        mouthpatch.append(fpu.getWaveletData(imgr[6]/255.0))
                    elif M3DT==2:
                        mo='periodization'
                        if labeltxt.find('CASIA')>0:
                            mo='constant'
                        print("Get wavelet Patches>>>>>>>>>>>>>>")
                        eyepath.append(fpu.get_WaveletDataC3(imgr[4]/255.0, mo=mo))
                        foreheadpath.append(fpu.get_WaveletDataC3(imgr[5]/255.0, mo=mo))
                        mouthpatch.append(fpu.get_WaveletDataC3(imgr[6]/255.0, mo=mo))
                    else:
                        raise RuntimeWarning('Unexperted M3 data type. Please check the M3DT again with correct setting.')
                if M1:
                    if M3DT==0:
                        imgs.append(imgr[0])
                    elif M3DT==1:
                        imgs.append(fpu.getWaveletData(imgr[0]/255.0))
                    elif M3DT==2:
                        mo='periodization'
                        if labeltxt.find('CASIA')>0:
                            mo='constant'
                        imgs.append(fpu.get_WaveletDataC3(imgr[0]/255.0, mo=mo))
                    else:
                        raise RuntimeWarning('Unexperted M1 data type. Please check the M3DT again with correct setting.')
                    print('Get recale image>>>>>>>>>>>')
                if M2:
                    if imgr[1]:
                        #print(type(np.array(imgr[2])), np.array(imgr[2]).shape)
                        #exit()
                        geo.append(np.array(imgr[2]))
                        print('Get geometry features>>>>>>>>>>>')
                    else:
                        #fw=open('M2_exception_log.txt','a')
                        #fw.write('%s\n'%v)
                        #fw.close()
                        continue
                if M4:
                    if M3DT==0:
                        cropfaces.append(imgr[7])
                    elif M3DT==1:
                        cropfaces.append(fpu.getWaveletData(imgr[7]/255.0))
                    elif M3DT==2:
                        mo='periodization'
                        if labeltxt.find('CASIA')>0:
                            mo='constant'
                        cropfaces.append(fpu.get_WaveletDataC3(imgr[7]/255.0, mo=mo))
                    else:
                        raise RuntimeWarning('Unexperted M4 data type. Please check the M3DT again with correct setting.')
                    
                    print('Get cropped face>>>>>>>>>>>')
                if M11:
                    if imgr[7] is None:
                        ec=ec+1
                        #raise RuntimeError('Unexpected error in data preprocesing.')
                        print('Unexpected error in data preprocesing.')
                        continue
                    #else:
                    #    continue
                    if RO:
                        r,c=imgr[7].shape
                        rm=cv2.getRotationMatrix2D((c//2, r//2), 90, 1.0)
                        imgr[7]=cv2.warpAffine(imgr[7], rm, (c, r))
                    twcs, wcs=fpu.getWaveletComplexShiftData(imgr[7]/255.0, log=LOG, Type=M11DT)
                    wcsl.append(twcs)
                    print('Get WCS data>>>>>>>>>>>>>')
                if LOG:
                    newname=logdir+'/'+imname
                    if M1:
                        cv2.imwrite(newname+'_rescale.jpg', imgr[0])
                    if M4:
                        cv2.imwrite(newname+'_crop.jpg',imgr[7])
                    if M3:
                        cv2.imwrite(newname+'_eye.jpg',imgr[4])
                        cv2.imwrite(newname+'_forehead.jpg',imgr[5])
                        cv2.imwrite(newname+'_mouth.jpg', imgr[6])
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
#datasetlist.append('CK+107TEST')
#datasetlist.append('KDEF6TEST')
#datasetlist.append('CK+107TEST2')
#datasetlist.append('KDEF6TEST2')

#datasetlist.append('CK+86')
#datasetlist.append('CK+106')
#datasetlist.append('CK+107')
#datasetlist.append('KDEF6')
#datasetlist.append('OuluCASIAVN6')
##datasetlist.append('OuluCASIANIR6')
#datasetlist.append('SFEW2.0_7')
datasetlist.append('SFEW2_7v2')

#datasetlist.append('CK+86_FLIP_enlarge')#2 times
#datasetlist.append('CK+106_FLIP_enlarge')#2 times
#datasetlist.append('CK+107_FLIP_enlarge')#2 times
#datasetlist.append('KDEF6_FLIP_enlarge')#2 times
#datasetlist.append('OuluCASIAVN6_FLIP_enlarge')#2 times
#datasetlist.append('OuluCASIANIR6_FLIP_enlarge')#2 times

#datasetlist.append('CK+86_FLIPNS')#22 times
#datasetlist.append('CK+106_FLIPNS')#22 times
#datasetlist.append('CK+107_FLIPNS')#22 times
#datasetlist.append('KDEF6_FLIPNS')#22 times
#datasetlist.append('OuluCASIAVN6_FLIPNS')#22 times
#datasetlist.append('OuluCASIANIR6_FLIPNS')#22 times

#datasetlist.append('CK+86_FLIPNST10')#22 times
#datasetlist.append('CK+106_FLIPNST10')#22 times
#datasetlist.append('CK+107_FLIPNST10')#22 times
#datasetlist.append('KDEF6_FLIPNST10')#22 times
#datasetlist.append('OuluCASIAVN6_FLIPNST10')#22 times
#datasetlist.append('OuluCASIANIR6_FLIPNST10')#22 times

#datasetlist.append('I:/Data/EnlargeData/CK+10GwithContempt_FLIPSN/CK+107_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10')
#datasetlist.append('I:/Data/EnlargeData/CKplus8G_FLIPSN/CK+86_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10')
#datasetlist.append('I:/Data/EnlargeData/CKplus10G_FLIPSN/CK+106_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10')
#datasetlist.append('I:/Data/EnlargeData/KDEF_G_FLIPSN/KDEF6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10')
#datasetlist.append('I:/Data/EnlargeData/OuluCASIAVN_FLIPSN/OuluCASIAVN6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10')
#datasetlist.append('I:/Data/EnlargeData/OuluCASIANIR_FLIPSN/OuluCASIANIR6_FLIP_enlarge_SignalNoise_enlarge_reorderT6_ACIT10')

for i in range(len(datasetlist)):
    preprocess(datasetlist[i])
