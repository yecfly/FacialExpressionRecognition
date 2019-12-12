####
#Source codes of the 'YYStex' project
# submitted to Neurocomputing for the purpose of review.
# Platform windows 10 with Tensorflow 1.13.1, CUDA 10.0, python 3.7.1 64 bit
###################
import testing, sys

def boolArgv(a):
    if a=='1' or a=='True' or a=='true':
        return True
    elif a=='0' or a=='False' or a=='false':
        return False
    else:
        raise RuntimeError('Unexpected argument with %s in %s'%(a, str(sys.argv)))

if len(sys.argv)==5:
    testing.runKerasTest(GPU_Device_ID=int(sys.argv[1]), 
                    DataSet=int(sys.argv[2]), 
                    NetworkType=int(sys.argv[3]), 
                    Module=int(sys.argv[4]))
elif len(sys.argv)==6:
    testing.runKerasTest(GPU_Device_ID=int(sys.argv[1]), 
                    DataSet=int(sys.argv[2]), 
                    NetworkType=int(sys.argv[3]), 
                    Module=int(sys.argv[4]),
                    COLOR=boolArgv(sys.argv[5]))
else:
    print("argument errors, try\npython Testscript.py <GPU_Device_ID> <DataSet> <NetworkType> <Module>")
'''
python Testscript.py 1 44 20138 211 1
python Testscript.py 1 44 20138 211 0

python Testscript.py 0 34 2352202 23 1
python Testscript.py 0 34 2352202 23 0


'''
