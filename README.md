## Earthformer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earthformer-exploring-space-time-transformers/weather-forecasting-on-sevir)](https://paperswithcode.com/sota/weather-forecasting-on-sevir?p=earthformer-exploring-space-time-transformers)

By [Zhihan Gao](https://scholar.google.com/citations?user=P6ACUAUAAAAJ&hl=zh-CN), [Xingjian Shi](https://github.com/sxjscience), [Hao Wang](http://www.wanghao.in/), [Yi Zhu](https://bryanyzhu.github.io/), [Yuyang Wang](https://scholar.google.com/citations?user=IKUm624AAAAJ&hl=en), [Mu Li](https://github.com/mli), [Dit-Yan Yeung](https://scholar.google.com/citations?user=nEsOOx8AAAAJ&hl=en).

This repo is the official implementation of ["Earthformer: Exploring Space-Time Transformers for Earth System Forecasting"](https://arxiv.org/pdf/2207.05833v1.pdf) that will appear in NeurIPS 2022.
We will soon attach the source code.

## Introduction

Conventionally, Earth system (e.g., weather and climate) forecasting relies on numerical simulation with complex physical models and are hence both 
expensive in computation and demanding on domain expertise. With the explosive growth of the spatiotemporal Earth observation data in the past decade, 
data-driven models that apply Deep Learning (DL) are demonstrating impressive potential for various Earth system forecasting tasks. 
The Transformer as an emerging DL architecture, despite its broad success in other domains, has limited adoption in this area. 
In this paper, we propose **Earthformer**, a space-time Transformer for Earth system forecasting. 
Earthformer is based on a generic, flexible and efficient space-time attention block, named **Cuboid Attention**. 
The idea is to decompose the data into cuboids and apply cuboid-level self-attention in parallel. These cuboids are further connected with a collection of global vectors.

Earthformer achieves strong results in synthetic datasets like MovingMNIST and N-body MNIST dataset, and also outperforms non-Transformer models (like ConvLSTM, CNN-U-Net) in SEVIR (precipitation nowcasting) and ICAR-ENSO2021 (El Nino/Southern Oscillation forecasting).


![teaser](figures/teaser.png)


## Citing Earthformer

```
@inproceedings{gao2022earthformer,
  title={Earthformer: Exploring Space-Time Transformers for Earth System Forecasting},
  author={Gao, Zhihan and Shi, Xingjian and Wang, Hao and Zhu, Yi and Wang, Yuyang and Li, Mu and Yeung, Dit-Yan},
  booktitle={NeurIPS},
  year={2022}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

