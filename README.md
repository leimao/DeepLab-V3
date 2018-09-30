# DeepLab v3

Lei Mao, Shengjie Lin

University of Chicago

Toyota Technological Institute at Chicago

## Introduction


DeepLab is the state-of-art image semantic segmentation model developed by Google. Its latest version is v3+ which employes an encoder-decoder architecture with atrous spatial pyramid prooling (ASPP). While the model works extremely well for semantic segmentation, its open sourced code is extremely hard to read (at least from my personal perspective). Here we reimplemented the DeepLab v3, earlier version of DeepLab v3+ which employs encoder architecture with ASPP, in a much simpler and understandable way.


## Dependencies

* Python 3.5
* TensorFlow 1.8
* Tqdm
* Numpy 1.14
* OpenCV

## Files

This project uses the `nets` module contained in `tensorflow/models`. In order to achieve this, first run the following command in `SOME_DIR` (preferably outside the `PROJECT_ROOT`) to make available the entire repo:
> git clone git@github.com:tensorflow/models.git tf-models

Then run the following command to make the soft link:
> ln -s $SOME_DIR/tf-models/research/slim/nets $PROJECT_ROOT/

## Usages

### Download Dataset



### Preprocess Dataset



### Train Model

### Test Model





## References

L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. [Deeplab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915). TPAMI, 2017.

L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam. [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587). arXiv:1706.05587, 2017.

L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611). arXiv:1802.02611, 2018.




