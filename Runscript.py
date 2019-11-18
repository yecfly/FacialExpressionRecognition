####
#Source codes of the paper 'Facial Expression Recognition via Region-based Convolutional Fusion Network'
# submitted to Neurocomputing for the purpose of review.
# Platform windows 10 with Tensorflow 1.2.1, CUDA 8.0, python 3.5.2 64 bit, MSC v.1900 64bit
###################
import training, sys

def boolArgv(a):
    if a=='1' or a=='True' or a=='true':
        return True
    elif a=='0' or a=='False' or a=='false':
        return False
    else:
        raise RuntimeError('Unexpected argument with %s in %s'%(a, str(sys.argv)))

if sys.argv[-1]=='keras':
    training.runKerasV(GPU_Device_ID=int(sys.argv[1]), DataSet=int(sys.argv[2]), TestID=int(sys.argv[3]),
                    NewC=int(sys.argv[4]), NetworkType=int(sys.argv[5]), runs=int(sys.argv[6]), 
                    Module=int(sys.argv[7]), Mfit=boolArgv(sys.argv[8]))
elif sys.argv[-1]=='Fusion':
    training.runKerasV(GPU_Device_ID=int(sys.argv[1]), DataSet=int(sys.argv[2]), TestID=int(sys.argv[3]),
                    NewC=int(sys.argv[4]), NetworkType=int(sys.argv[5]), runs=int(sys.argv[6]), 
                    Module=int(sys.argv[7]), Mfit=boolArgv(sys.argv[8]), FusionNType=int(sys.argv[9]), 
                    FusionMfit=boolArgv(sys.argv[10]), FLdata=boolArgv(sys.argv[11]), FUSION=True)
elif len(sys.argv)==8:
    training.run(GPU_Device_ID=int(sys.argv[1]), 
                    DataSet=int(sys.argv[2]), 
                    ValidID=int(sys.argv[3]), TestID=int(sys.argv[4]), 
                    NetworkType=int(sys.argv[5]), 
                    runs=int(sys.argv[6]), Module=int(sys.argv[7]))
else:
    print("argument errors, try\npython runfile.py <GPU_Device_ID> <DataSet> <ValidID> <TestID> <NetworkType> <runs> <Module>")
'''
Usage:
 python Runscript.py 0 1 0 0 1 0
 python Runscript.py 0 2 3 3 2 5
 python Runscript.py 1 4 3 3 2 8
 python Runscript.py 1 6 5 5 3 3
'''
