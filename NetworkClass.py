"""Functions for building the facial expression recognition network class.
"""
# 
# part of the codes are based on David Sandberg 2016 
# 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time,os
import FERNetworks
import tensorflow as tf

def networkCall(NetworkType, holders, Category):
    if NetworkType==0:
        return FERNetworks.OPNetwork(holders[0], classNo=Category)
    elif NetworkType==1:
        return FERNetworks.ARCFN(holders[0], holders[1], holders[2], classNo=Category)
    elif NetworkType==2:
        return FERNetworks.BRCFN(holders[0], holders[1], holders[2], classNo=Category)
    elif NetworkType==3:
        return FERNetworks.CRCFN(holders[0], holders[1], holders[2], classNo=Category)
    else:
        raise RuntimeError('Unexpected NetworkType, please check it.')
        
def formFeedDict(holders, inputs):
    assert len(holders)==len(inputs), 'Unexpected length for the inputs'
    inputDict={}
    for i, v in enumerate(holders):
        inputDict[holders[i]]=inputs[i]
    return inputDict
def getOptimizer(learningRate, optimizerType):
    if optimizerType==1:
        return tf.train.AdamOptimizer(learningRate)
    else:
        raise RuntimeError('Unexpected optimizer')
#### using the commom class
class FERFN():
    '''rethinking the tanh activations in the network'''
    def __init__(self, holders, NetworkType, Category, graph, modelPath=None, phase_train=True, Runs=0, lossType=1,
                 initialLearningRate=0.0001, learningDecayStep=2000, learningDecayRate=0.8, staircase=True, optimizerType=1, Summary=False):
        '''Set the parameters while creating the class.'''
        self.graph=graph
        with self.graph.as_default():
            #self.__is_training=tf.placeholder(tf.bool)
            self.__NetwortType=NetworkType
            self.__logits = networkCall(NetworkType, holders, Category)

            self.__Label=tf.placeholder(tf.float32, [None, Category])
            self.__initialLoss(lossType)
            
            self.__globalStep=tf.Variable(0, trainable=False)
            self.__learningRate=tf.train.exponential_decay(initialLearningRate, self.__globalStep, 
                                                            learningDecayStep, learningDecayRate, staircase=staircase)
            self.__optm=tf.train.AdamOptimizer(self.__learningRate)
            self.__train_op=self.__optm.minimize(self.__loss, self.__globalStep)
            self.__correcta_prediction=tf.equal(tf.argmax(self.__logits, 1), tf.argmax(self.__Label, 1))####important to understand the self.__logits above
            self.__test_cast=tf.cast(self.__correcta_prediction, 'float')
            self.__sum_test=tf.reduce_sum(self.__test_cast)
            self.__accuracy=tf.reduce_mean(self.__test_cast)

            self.saver=tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=3,pad_step_number=True,filename='auto-checkpoint')
            if phase_train:
                SummaryDir='./Summary/FERFN%d/train/runs%d'%(NetworkType,Runs)
                self.__summary_writer_train=tf.summary.FileWriter(SummaryDir,self.graph)
            SummaryDir='./Summary/FERFN%d/test/runs%d'%(NetworkType, Runs)
            self.__summary_writer_test=tf.summary.FileWriter(SummaryDir,self.graph)
            if Summary: 
                tf.summary.histogram('metric/__logits',self.__logits)
                tf.summary.scalar('metric/batch_accuracy', self.__accuracy)
                tf.summary.scalar('metric/loss', self.__loss)
                self.__summary_ops=tf.summary.merge_all()
        if modelPath is None:
            self.checked=False
            self.session=None
        else:
            try:
                self.session=tf.InteractiveSession(graph=self.graph)
                self.session.run(tf.variables_initializer(var_list=self.graph.get_collection(name='variables')))
                self.saver.restore(sess=self.session, save_path=modelPath)
                self.checked=True
            except:
                raise RuntimeError('Unable to restore the model from %s. \nPlease check the error info bellow.'%(modelPath))

        return super().__init__()


    def __initialLoss(self, LossType):
        if LossType==1:
            self.__loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.__Label, logits=self.__logits), 0)
        else:
            raise RuntimeError('Unexpected lossType, please check it.')

    @property
    def _T_loss(self):
        return self.__loss

    @property
    def _T_SummaryOPS(self):
        return self.__summary_ops

    @property
    def _T_train_op(self):
        return self.__train_op

    @property
    def _T_learningRate(self):
        return self.__learningRate

    @property
    def _T_label(self):
        return self.__Label

    @property
    def _T_accuracy(self):
        return self.__accuracy

    @property
    def _T_sum_test(self):
        return self.__sum_test

    @property
    def _T_logits(self):
        return self.__logits

    @property
    def _T_globalStep(self):
        return self.__globalStep

    @property
    def _T_correcta_prediction(self):
        return self.__correcta_prediction

    @property
    def _T_test_cast(self):
        return self.__test_cast

    @property
    def _T_optm(self):
        return self.__optm

    def Train_addSummary(self, valueList, step):
        self.__summary_writer_train.add_summary(valueList, global_step=step)

    def Test_addSummary(self, valueList, step):
        self.__summary_writer_test.add_summary(valueList, global_step=step)

    def train(self, session, targetList, holders, inputs):
        '''past the session, and the targetList(tensor you want to )'''
        outPutTuple = session.run(targetList, feed_dict=formFeedDict(holders, inputs))
        return outPutTuple

    def restore(self, modelpath, session):
        if session is None:
            if self.session is None:
                self.session=tf.InteractiveSession(graph=self.graph)
                self.session.run(tf.variables_initializer(var_list=self.graph.get_collection(name='variables')))
            self.saver.restore(sess=self.session, save_path=modelpath)
            self.checked=True
        else:
            self.saver.restore(sess=session, save_path=modelpath)
            
    def resetGlobalStep(self):
        self.session.run(tf.assign(self.__globalStep, 0))

    def save(self, session, modelname):
        self.saver.save(sess=session, save_path=modelname)
        self.checked=True

    def pridict(self, holders, inputs, session):
        prob=session.run([self.__logits], feed_dict=formFeedDict(holders, inputs))
        return prob