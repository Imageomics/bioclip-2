## BioCLIP 2

* Based on [openCLIP 2.29.0](https://github.com/mlfoundations/open_clip/tree/82d7496799688ea11772576b73c6b71fb7dcb986).
* Trained on [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M).
* Includes part of the LAION-2B data for experience replay.
* [commit ec92236](https://github.com/Imageomics/bioclip-2/commit/ec92236a74fc41eb90e5968e871191e0e6d76802) from [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).
* [commit 730e08c](https://github.com/Imageomics/bioclip-2/commit/730e08c08f93ba0e7ab9d0643c27a9ac95cb4d1b):
  * [src/evaluation/faster_mix_k_means_pytorch.py](src/evaluation/faster_mix_k_means_pytorch.py), [src/evaluation/herbarium19.py](src/evaluation/herbarium19.py), and [src/evaluation/herbarium_19_class_splits.pkl](src/evaluation/herbarium_19_class_splits.pkl) from [sgvaze/generalized-category-discovery](https://github.com/sgvaze/generalized-category-discovery).
  * [src/evaluation/simpleshot.py](src/evaluation/simpleshot.py) from [samuelstevens/biobench](https://github.com/samuelstevens/biobench).
  * [src/evaluation/newt.py](src/evaluation/newt.py) from [samuelstevens/biobench](https://github.com/samuelstevens/biobench), modified from Visipedia.

## BioCLIP

* Based on [openCLIP 2.14.0](https://github.com/mlfoundations/open_clip/tree/7ae3e7a9853b1aa2fe7825e4272f3b169f8e65af).
* Trained on [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M).
* [commit 37b729b](https://github.com/Imageomics/bioclip-2/commit/37b729bc69068daa7e860fb7dbcf1ef1d03a4185) and prior from [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).
