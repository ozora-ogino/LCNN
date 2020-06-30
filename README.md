# LCNN
LCNN is a kind of CNN that is fearued by max Ffeature mapping function (MFM).

MFM is an alternative of ReLU to suppress low-activation neurons in each layer.

If you'd like to know more detail, see the references below.
## Experiment setup
In this code, I used ASVspoof2019 database.

I used spectrograms that extracted by using FFT or CQT as speech feature.

EER is used for evaluating.


## Description
feature.py : In this file, you can see some functions that extracts speech features from raw audio data. 

model/lcnn.py : Helper function for building LCNN.

model/layers.py : Maxout function for MFM.

metrics.py : Calculating EER.

train.py : In this file, you can train your LCNN model and evaluate.


## Reference
"A Light CNN for Deep Face Representation with Noisy Labels" [https://arxiv.org/pdf/1511.02683.pdf]

"STC Antispoofing Systems for the ASVspoof2019 Challenge" [https://arxiv.org/abs/1904.05576]

ASVspoof2019 [https://www.asvspoof.org]
