# MusiCNN
MusiCNN reimplementation in PyTorch - so that it's easier for me to train with dataset subsets for testing limited-data scenarios. Mel spectrogram calculation is done online in GPU (so no dataset preprocessing needed).

Pieced together from:
[Original inference](https://github.com/jordipons/musicnn) | [Original training](https://github.com/jordipons/musicnn-training) | [PyTorch implementation](https://github.com/ilaria-manco/music-audio-tagging-pytorch) | [`mir_ref`](https://github.com/chrispla/mir_ref)

Paper: 
```
@inproceedings{pons2019musicnn,
  title={musicnn: pre-trained convolutional neural networks for music audio tagging},
  author={Pons, Jordi and Serra, Xavier},
  booktitle={Late-breaking/demo session in 20th International Society for Music Information Retrieval Conference (LBD-ISMIR2019)},
  year={2019},
}
```

