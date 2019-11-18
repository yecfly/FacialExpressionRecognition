# FacialExpressionRecognition
a FER training framework for Keras API

The FaceProcessUtil.py and FERNetworks.py are missing for current training framework. And the Radam and Lookahead optimizer are not available in Keras2.2.5. You can acquire it from the internet or modify the optimizer.py of Keras just like 'Keras optimizers Added radam lookahead.py' did.

The labels employed in the CK+, KDEF, and Oulu txt files are as follows:
0=anger, 
1=surprise, 
2=disgust, 
3=fear, 
4=happy, 
5=sadness,
6=contempt.
------------------sample labels and groups, corresponding to the six splits in the Experiment section.
***	CK+86.txt
***	CK+106.txt
***	CK+107.txt
***	KDEF6.txt
***	OuluCASIANIR6.txt
***	OuluCASIAVN6.txt
