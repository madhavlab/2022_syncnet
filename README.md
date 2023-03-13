<!-- # INTERSPEECH' 22 -->
# [SyncNet: Correlating Objective for Time Delay Estimation in Audio Signals](https://arxiv.org/abs/2203.14639)
### Authors: [Akshay Raina](https://raina-akshay.github.io) and [Vipul Arora](https://vipular.github.io)

<details>
<summary>Abstract</summary>
This study addresses the task of performing robust and reliable time-delay estimation in signals in noisy and reverberating environments. In contrast to the popular signal processing based methods, this paper proposes to transform the input signals using a deep neural network into another pair of sequences which show high cross correlation at the actual time delay. This is achieved with the help of a novel correlation function based objective function for training the network. The proposed approach is also intrinsically interpretable as it does not lose temporal information. Experimental evaluations are performed for estimating mutual time delays for different types of audio signals such as pulse, speech and musical beats. SyncNet outperforms other classical approaches, such as GCC-PHAT, and some other learning based approaches.
</details>

This repository consists of the supplementary material and implementation (using Python 3+) of __SyncNet__, a deep neural network for time-synchronization of signals.

<!--For the demo of the proposed methodology, please check the [jupyter notebook](https://google.com).-->

### Citation
```BibTeX
@article{raina2022syncnet,
  title={SyncNet: Using Causal Convolutions and Correlating Objective for Time Delay Estimation in Audio Signals},
  author={Raina, Akshay and Arora, Vipul},
  journal={arXiv preprint arXiv:2203.14639},
  year={2022}
}
```

