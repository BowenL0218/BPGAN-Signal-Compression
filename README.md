# Unified Signal Compression Using Generative Adversarial Networks
Codes for [Unified Signal Compression Using Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053233)(ICASSP 2020), a generative adverdarial networks (GAN) based signal (Image/Speech) compression algorithm.

# Introduction and Framework
The proposed unified compression framework uses a generative adversarial network (GAN) to compress heterogeneous signals. The compressed signal is represented as a latent vector and fed into a generator network that is trained to produce high quality realistic signals that minimize a target objective function. To efficiently quantize the compressed signal, non-uniformly quantized optimal latent vectors are identified by iterative back-propagation with alternating direction method of multipliers (ADMM) optimization performed for each iteration. 
![Flow chart](https://github.com/BowenL0218/BPGAN-Signal-Compression/blob/main/Images/flowchart.png)
The target signal is first encoded as an initialization to the latent representation by an encoder. Then the latent vector $\vx$ is optimized according to a reconstruction criterion throughinterative ADMM back-propagation. The latent variable is also discretized in this iterative back-propagation (BP) process.The quantized latent signal is entropy encoded for further code size reduction. In the proposed framework, the generatorparameters are shared between the transmitter and receiver. At the receiver, the generator decodes the latent signal, whichis converted back to its original format via post-processing.

## Architecture
The detailed neural network structure of our predictor model.
![Predictor architecture](https://github.com/BowenL0218/Video-compression/blob/main/Images/predictor.png)
The detailed neural network structure of our decoder model.
![Decoder architecture](https://github.com/BowenL0218/Video-compression/blob/main/Images/decoder.png)

## Datasets
In order to use the datasets used in the paper, please download the [UVG dataset](https://media.withyoutube.com/), the [Kinetics dataset](https://deepmind.com/research/open-source/kinetics), the [VTL dataset](http://trace.eas.asu.edu/index.html), and the [UVG dataset](http://ultravideo.fi/).

- The UVG and Kinetics dataset are used for training the prediction network. 
- The VTL and UVG datasets are implemented for testing the performance.
- Note that we use the learning based image compression algorithm ([Liu et al](https://arxiv.org/pdf/1912.03734.pdf)) as the intra compression for one single frame. 
- The compressed latents are the input for the prediction network. 

## ADMM quantization
To further reduce the bitrate of the compressed video, we applied ADMM quantization for the residual from latent prediction incorporated in the proposed video compression framework. 

## Arithmetic Coding
To use the entropy coding method in this paper, download the general code library in python with [arithmetic coding](https://github.com/ahmedfgad/ArithmeticEncodingPython). 

## Test pretrained model
To tested the result without ADMM quantization,
```sh
$ python test.py
```

To test the result with ADMM quantization
```sh
$ python Compression_ADMM.py
```

## Citation
Please cite our paper if you find our paper useful for your research. 
