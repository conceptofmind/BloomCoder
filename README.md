## BloomCoder
A scalable fine-tuning implementation utilizing the Bloom model for code generation.

### Usage
```bash
$ git clone https://github.com/conceptofmind/BloomCoder.git
$ cd BloomCoder
$ colossalai run --nproc_per_node 1 train.py --use_trainer
```

### Developer Updates
Developer updates can be found on: 
- https://twitter.com/EnricoShippole
- https://www.linkedin.com/in/enrico-shippole-495521b8/

### TODO:
- [ ] Add logging with Weights and Biases
- [x] Build data loaders
- [ ] Setup ColossalAI engine
- [ ] Implement ZeRO

### Author:
- Enrico Shippole

### Additional Information:
You can find more information, about Bloom, on the main website at https://bigscience.huggingface.co. You can also follow BigScience on Twitter at https://twitter.com/BigScienceW. Huggingface provides a section in their transformers documentation for [training models in native pytorch](https://huggingface.co/docs/transformers/training).

### Citations:

```bibtex
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```