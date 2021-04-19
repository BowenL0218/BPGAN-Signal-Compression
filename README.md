# Unified Signal Compression Using Generative Adversarial Networks
Codes for [Unified Signal Compression Using Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053233) (ICASSP 2020), a generative adverdarial networks (GAN) based signal (Image/Speech) compression algorithm.

# Introduction
The proposed unified compression framework (BPGAN) uses a GAN to compress heterogeneous signals. The compressed signal is represented as a latent vector and fed into a generator network that is trained to produce high quality realistic signals that minimize a target objective function. To efficiently quantize the compressed signal, non-uniformly quantized optimal latent vectors are identified by iterative back-propagation with alternating direction method of multipliers (ADMM) optimization performed for each iteration. 

## Framework
![Flow chart](https://github.com/BowenL0218/BPGAN-Signal-Compression/blob/main/Images/flowchart.png)
The target signal is first encoded as an initialization to the latent representation by an encoder. Then the latent vector z is optimized according to a reconstruction criterion through interative ADMM back-propagation. The latent variable is also discretized in this iterative back-propagation (BP) process.The quantized latent signal is entropy encoded for further code size reduction. In the proposed framework, the generator parameters are shared between the transmitter and receiver. At the receiver, the generator decodes the latent signal, which is converted back to its original format via post-processing.

## Architecture
The detailed neural network structure of our generator model.
![Generator architecture](https://github.com/BowenL0218/BPGAN-Signal-Compression/blob/main/Images/arc.png)

## Datasets
In order to use the datasets used in the paper, please download the [Open Image dataset](https://storage.googleapis.com/openimages/web/index.html), the [Kodak dataset](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html), and the [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1).

- The Open Image dataset are used for training the image compression network. 
- The Kodak datasets is implemented for testing the performance.
- The TIMIT dataset is used for training the speech compression network. 
- The compressed signal is the input for the generator network. 

## ADMM quantization
To further reduce the bitrate of the compressed signal, we applied [ADMM quantization](https://arxiv.org/abs/1812.11677) for the compressed signal by searching the optimal quantized latent through back-propagation iteratively.

## Huffman Coding
To further reduce the size of the compressed signal, we implemented the Huffman coding as the final compression step in our framework.  

# Code

## Installation (For Linux)
- Install Pytorch and Torchvision [website](https://pytorch.org/)
- Install requirement package libfftw3-dev, liblapack-dev, ltfatpy, librosa
```sh
$ sudo apt install libfftw3-dev
$ sudo apt install liblapack-dev
$ pip install ltfatpy, librosa 
```

## Train the model
With all the package and data downloaded, train the compression model with commend:
```sh
$ python train.py
```

## Test pretrained model
To tested the result without ADMM quantization,
```sh
$ python test.py
```

To test the result with ADMM quantization
```sh
$ python Compression_ADMM.py
```

# Citation
Please cite our paper if you find our paper useful for your research.
```bibtex
@INPROCEEDINGS{9053233,
  author={B. {Liu} and A. {Cao} and H. -S. {Kim}},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Unified Signal Compression Using Generative Adversarial Networks}, 
  year={2020},
  volume={},
  number={},
  pages={3177-3181},
  doi={10.1109/ICASSP40776.2020.9053233}}
```
