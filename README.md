# BloomCoder
A scalable fine-tuning implementation utilizing the Bloom model for code generation. Huggingface provides a section in their transformers documentation for [training models in native pytorch](https://huggingface.co/docs/transformers/training).

### Usage
```bash
$ git clone https://github.com/conceptofmind/BloomCoder.git
$ cd BloomCoder
$ colossalai run --nproc_per_node 1 train.py --use_trainer
```

### Additional Information:
You can find more information, about Bloom, on the main website at https://bigscience.huggingface.co. You can also follow BigScience on Twitter at https://twitter.com/BigScienceW.

### Citations:

```bibtex
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```