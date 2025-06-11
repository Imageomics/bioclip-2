# BioCLIP 2

This repository contains the code for [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) training and evaluation (testing and visualizing embeddings). We developed this repository based on [BioCLIP](https://github.com/imageomics/BioCLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip).
BioCLIP 2 is trained on the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) and achieves state-of-the-art performance on both species classification and other biological visual tasks. The BioCLIP 2 website is hosted from the `gh-pages` branch of this repository.

[Paper](https://doi.org/10.48550/arXiv.2505.23883) | [Model](https://huggingface.co/imageomics/bioclip-2) | [Data](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) | [Demo](https://huggingface.co/spaces/imageomics/bioclip-2-demo)
---

BioCLIP 2 is a CLIP model trained on a new 200M-image dataset of biological organisms with fine-grained taxonomic labels.
BioCLIP 2 outperforms general domain baselines on a wide spread of biology-related tasks, including zero-shot and few-shot classification.

## Table of Contents

1. [Model](#model)
2. [Training and Evaluation Commands](#commands)
3. [Paper, website, and data](#paper)
4. [Citation](#citation)

## Model

The main differences in the training implementation between BioCLIP 2 and BioCLIP are the adopted model architecture and the introduction of experience replay. BioCLIP 2 employs a ViT-L/14 CLIP architecture pre-trained with LAION-2B data. Along with the contrastive optimization of biological organism data, we also include part of the LAION-2B data for experience replay. In order to reduce the influence of the domain gap between hierarchical labels and image captions, we use two separate visual projectors on top of the visual encoder. This part of the code is in [transformer.py](src/open_clip/transformer.py).
We provide the weight of BioCLIP 2 in the [BioCLIP 2 model repo](https://huggingface.co/imageomics/bioclip-2).

## Commands

### Training
The [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) images can be downloaded from their original sources with [distributed-downloader](https://github.com/Imageomics/distributed-downloader). [TreeOfLife-toolbox/docs](https://github.com/Imageomics/TreeOfLife-toolbox/tree/main/docs#treeoflife200m-dataset-download-guide) contains instructions for full download into the proper format, and the code to construct the webdataset for training. These repositories are included in the supplementary material.
[img2dataset](https://github.com/rom1504/img2dataset) can be used to download data from the first three metadata parquet files of LAION-2B-en; we use the first downloaded 4,000 tar files for experience replay. Finally, download the validation set from [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) ([download instructions](https://github.com/Imageomics/bioclip/blob/main/docs/imageomics/treeoflife10m.md)), as we use that for evaluation during training.

Clone this repository, then install the requirements:
```
conda env create -f requirements-training.yml
```

To train the model, run:
```
sbatch slurm/train.sh
```

### Evaluation
**Species classification**

We evaluated [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) on the same test sets as used for [BioCLIP](https://huggingface.co/imageomics/bioclip), as well as a newly curated camera trap test set:

- [NABirds](https://dl.allaboutbirds.org/nabirds): In place of [Birds525](https://www.kaggle.com/datasets/gpiosenka/100-bird-species), since it is no longer available.
- [Meta-Album](https://meta-album.github.io/): For comparison to [BioCLIP](https://huggingface.co/imageomics/bioclip), we used the Plankton, Insects, Insects 2, PlantNet, Fungi, PlantVillage, and Medicinal Leaf datasets.
- [Rare Species](https://huggingface.co/datasets/imageomics/rare-species): Nearly 12K images representing 400 species labeled Near Threatened through Extinct in the Wild by the [IUCN Red List](https://www.iucnredlist.org/).
- [IDLE-OO Camera Traps](https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps): A new dataset we curated to evaluate performance on camera trap images. It is constructed from five [Labeled Information Library of Alexandria: Biology and Conservation (LILA BC)](https://lila.science) datasets labeled to the image-level, which we then balanced. See the [IDLE-OO Camera Traps dataset](https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps) for more details.

The metadata used in evaluation is provided in [`data/annotation`](data/annotation/), including [NABirds](data/annotation/nabirds), [Rare Species](data/annotation/rare_species/), and other benchmarks from [Meta Album](data/annotation/meta-album/). All evaluation parameters are described in [src/evaluation/README.md](src/evaluation/README.md).
Please be sure to update the directories accordingly to reflect the locations of these data and metadata in `slurm/eval.sh` and run:
```
sbatch slurm/eval.sh
```

**Other biological visual tasks**

We also evaluated on biological tasks that go beyond species classification with the following datasets:
- [NeWT](https://github.com/visipedia/newt)
- [FishNet](https://fishnet-2023.github.io/)
- [AwA2](https://cvml.ista.ac.at/AwA2/)
- [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6/data)
- [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)

Please be sure to update the directories accordingly to reflect the locations of these data in `slurm/eval_other.sh` and run:
```
sbatch slurm/eval_other.sh
```

<h2 id="paper">Paper, Website, and Data</h2>

We have a preprint on [arXiv](https://doi.org/10.48550/arXiv.2505.23883) and a [project website](https://imageomics.github.io/bioclip-2/).

Our data is published on Hugging Face: [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) and [IDLE-OO Camera Traps](https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps). Step-by-step download instructions for TreeOfLife-200M are available in [TreeOfLife-toolbox](https://github.com/Imageomics/TreeOfLife-toolbox/tree/main/docs#treeoflife200m-dataset-download-guide).

## Citation

Please cite our papers and the associated repositories if you use our code or results.

```
@article{gu2025bioclip,
  title = {{B}io{CLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning}, 
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  eprint={2505.23883},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.23883}, 
}
 ```

Our code (this repository):
```
@software{bioclip2code,
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G. Campolongo and Matthew J. Thompson and Net Zhang and Jiaman Wu and Zheda Mai},
  doi = {TBA},
  title = {{B}io{CLIP} 2},
  version = {1.0.0},
  year = {2025}
}
```

Also consider citing OpenCLIP and BioCLIP:

```
@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title={OpenCLIP},
  year={2021},
  doi={10.5281/zenodo.5143773},
}
```

Original BioCLIP Paper:
 ```
@inproceedings{stevens2024bioclip,
  title = {{B}io{CLIP}: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
  pages = {19412-19424}
}
```
Original Code:
```
@software{bioclip2023code,
  author = {Samuel Stevens and Jiaman Wu and Matthew J. Thompson and Elizabeth G. Campolongo and Chan Hee Song and David Edward Carlyn},
  doi = {10.5281/zenodo.10895871},
  title = {BioCLIP},
  version = {v1.0.0},
  year = {2024}
}
```

## License

BioCLIP 2 is released under the MIT License. Some elements of the code are copyright by others (see [`LICENSE`](LICENSE)); detailed provenance information is provided in [`HISTORY.md`](HISTORY.md).
