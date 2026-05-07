# Third-party code

This directory contains adapted third-party code from TabDiff, which is used for tabular diffusion modeling in this repository.

Original repository: [MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff)

TabDiff is used in this repository in two places:

- as the diffusion component in the OncoSynth generation pipeline;
- as the TabDiff baseline in the paper experiments.

The original TabDiff README is preserved in [`TabDiff/README_original.md`](TabDiff/README_original.md).

## License and attribution

TabDiff is distributed under the MIT License. The original license text is included in [`TabDiff/LICENSE`](TabDiff/LICENSE).


If you use the TabDiff components included in this repository, please also refer to the original TabDiff repository for the appropriate citation and license information.


If you use this code, please also cite the original TabDiff paper:

```bibtex
@inproceedings{
shi2025tabdiff,
title={TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation},
author={Juntong Shi and Minkai Xu and Harper Hua and Hengrui Zhang and Stefano Ermon and Jure Leskovec},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=swvURjrt8z}
}