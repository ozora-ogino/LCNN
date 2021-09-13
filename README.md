# Light CNN implemented by Tensorflow-Keras

## Description
Light CNN (LCNN) is CNN based model which was propsed in Interspeech 2019 by STC teams and state of the art of ASVspoof2019.
This LCNN is specificaly designed for spoofing detection for ASV system, but I believe LCNN can be used for other situations.

LCNN is fearued by max feature mapping function (MFM).
MFM is an alternative of ReLU to suppress low-activation neurons in each layer.
MFM contribute to make LCNN lighter and more efficient than CNN with ReLU.

If you'd like to know more detail, see the references below.

## Experiment setup
In this project, I used ASVspoof2019 database for testing my LCNN.
As a speech feature, I used spectrograms that extracted by using STFT or CQT. 
EER is used for evaluating.


## Description
feature.py : In this file, you can see some functions that extracts speech features from raw audio data. 

model/lcnn.py :LCNN model which implemented on Keras.

model/layers.py : Maxout function for MFM.

metrics.py : Calculating EER

train.py : Train and evaluation.


## Reference
"A Light CNN for Deep Face Representation with Noisy Labels" [https://arxiv.org/pdf/1511.02683.pdf]

"STC Antispoofing Systems for the ASVspoof2019 Challenge" [https://arxiv.org/abs/1904.05576]

ASVspoof2019 [https://www.asvspoof.org]
