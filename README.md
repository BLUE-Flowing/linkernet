# linkernet

### Dependency

The code has been tested in the following environment:


| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8       |
| PyTorch           | 1.13.1    |
| CUDA              | 11.6      |
| PyTorch Geometric | 2.2.0     |
| RDKit             | 2022.03.2 |

### Install via Conda and Pip
```bash
conda create -n targetdiff python=3.8
conda activate targetdiff
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
```

### Pretrained checkpoint
The [pretrained checkpoints](https://drive.google.com/drive/folders/1C1srELCCNJLk8v1smjvmbE-xYvnog5jU?usp=sharing) on ZINC / PROTAC. 
The ZINC checkpoint is right for this task

### Sampling
  python scripts/sample_exists.py --config 'configs/sampling/zinc.yml' --outdir 'outputs/zinc' --device 'cuda:0' --fragment_1_path 'path/to/fragment_1.sdf' --fragment_2_path 'path/to/fragment_2.sdf'

You can change the outdir and the outdir/folder_name. I used str(str(fragment_1_name) + '_' + str(fragment_2_name)) for reference. (on line 62)
