<!-- # INTERSPEECH' 22 -->
# [SyncNet: Using Causal Convolutions and Correlating Objective for Time Delay Estimation in Audio Signals](https://arxiv.org/abs/2203.14639)
### Authors: [Akshay Raina](https://raina-akshay.github.io) and [Vipul Arora](https://vipular.github.io)

<details>
<summary>Abstract</summary>
This paper addresses the task of performing robust and reliable time-delay estimation in audio-signals in noisy and reverberating environments. In contrast to the popular signal processing based methods, this paper proposes machine learning based method, i.e., a semi-causal convolutional neural network consisting of a set of causal and anti-causal layers with a novel correlation-based objective function. The causality in the network ensures non-leakage of representations from future time-intervals and the proposed loss function makes the network generate sequences with high correlation at the actual time delay. The proposed approach is also intrinsically interpretable as it does not lose time information. Even a shallow convolution network is able to capture local patterns in sequences, while also correlating them globally. SyncNet outperforms other classical approaches in estimating mutual time delays for different types of audio signals including pulse, speech and musical beats.
</details>

This repository consists of the supplementary material and implementation (using Python 3+) of __SyncNet__, a deep neural network for time-synchronization of signals.

<!--For the demo of the proposed methodology, please check the [jupyter notebook](https://google.com).-->

### Citation
```BibTeX

@misc{https://doi.org/10.48550/arxiv.2203.14639,
  doi = {10.48550/ARXIV.2203.14639},
  url = {https://arxiv.org/abs/2203.14639},
  author = {Raina, Akshay and Arora, Vipul},
  keywords = {Audio and Speech Processing (eess.AS), Signal Processing (eess.SP), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {SyncNet: Using Causal Convolutions and Correlating Objective for Time Delay Estimation in Audio Signals},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

