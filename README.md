# Light CNN for ASVSpoof (Tensorflow-Keras)

[![Test](https://github.com/ozora-ogino/LCNN/actions/workflows/test.yaml/badge.svg)](https://github.com/ozora-ogino/LCNN/actions/workflows/test.yaml)

## Description
Light CNN (LCNN) is CNN based model which was proposed in Interspeech 2019 by STC teams and state of the art of ASVspoof2019.

LCNN is featured by max feature mapping function (MFM).
MFM is an alternative of ReLU to suppress low-activation neurons in each layer.
MFM contribute to make LCNN lighter and more efficient than CNN with ReLU.

If you'd like to know more detail, see the references below.

## Experiment setup
In this project, LCNN is trained with ASVspoof2019 PA dataset.
As a speech feature, I used spectrograms that extracted by using STFT or CQT.

## Reference
["A Light CNN for Deep Face Representation with Noisy Labels"](https://arxiv.org/pdf/1511.02683.pdf)

["STC Antispoofing Systems for the ASVspoof2019 Challenge"](https://arxiv.org/abs/1904.05576)

[ASVspoof2019](https://www.asvspoof.org)


## Contributing
Interested in contributing? Awesome!
Fork and create PR! Or you can post Issue for bug reports or your requests (e.g. pytorch support).