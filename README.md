# Graph Emotion Decoding (GED)

This repository contains the official code for the paper
**Graph Emotion Decoding from Visually Evoked Neural Responses (MICCAI 2022)**.

## 1&nbsp; Installation

Follow the steps below to prepare the virtual environment.

Create and activate the environment:
```shell
conda create -n ged python=3.6
conda activate ged
```

Install dependencies:
```shell
pip install -r requirements.txt
```

## 2&nbsp; Experiments

### 2.1&nbsp; Download Datasets

The preprocessed fMRI data for five subjects and emotion scores are available at [figshare](https://doi.org/10.6084/m9.figshare.11988351).
We have downloaded the file `feature.tar.gz` of emotion category scores and unzipped it into the sub-folder `data/feature/` in this repository.
However, we do not provide the fMRI data since the size of a single file is at least 1GB.
Please follow the instructions below to download the required fMRI data.
You can refer to the [official website](https://github.com/KamitaniLab/EmotionVideoNeuralRepresentation) of Horikawa et al. for more details about the data.

Download the preprocessed fMRI data from [here](https://doi.org/10.6084/m9.figshare.11988351) and unzip the downloaded file `Subject?_preprocessed_fmri.tar.gz` into the corresponding sub-folder `data/fmri/Subject?/preprocessed/`.
Take "Subject 1" for example:
```shell
tar -zxvf Subject1_preprocessed_fmri.tar.gz
```
If downloaded and unzipped correctly, the files for Subject 1 can be found at `data/fmri/Subject1/preprocessed/fmri_Subject1_Session[1-5].h5`.

After getting all data for five subjects following the above steps, the folder tree of `data` should look like:
```
data/
  ├───feature/
  │     └───category.mat
  │         categcontinuous.mat
  └───fmri/
        ├───Subject1/
        │     └───preprocessed/
        │           └───fmri_Subject1_Session1.h5
        │               fmri_Subject1_Session2.h5
        │               ...
        │               fmri_Subject1_Session5.h5
        ├───Subject2/
        │     └───preprocessed/
        │           └───fmri_Subject2_Session1.h5
        │               fmri_Subject2_Session2.h5
        │               ...
        │               fmri_Subject2_Session7.h5
        ├───Subject3/
        │     └───preprocessed/
        │           └───fmri_Subject3_Session1.h5
        │               fmri_Subject3_Session2.h5
        │               ...
        │               fmri_Subject3_Session6.h5
        ├───Subject4/
        │     └───preprocessed/
        │           └───fmri_Subject4_Session1.h5
        │               fmri_Subject4_Session2.h5
        │               ...
        │               fmri_Subject4_Session5.h5
        └───Subject5/
              └───preprocessed/
                    └───fmri_Subject5_Session1.h5
                        fmri_Subject5_Session2.h5
                        ...
                        fmri_Subject5_Session5.h5
```

### 2.2&nbsp; Run and Reproduce

Execute the following command to run and reproduce the experiments:
```shell
python main.py --subject_id <subject> --num_sessions <session> --fold_idx <fold>
```
where `<fold>` takes from 0 to 9, and `<subject>` takes from 1 to 5.
Different `<subject>` corresponds to different `<session>`, and their relationships are listed as follows:

|         | sub 1 | sub 2 | sub 3 | sub 4 | sub 5 |
| :------ | :---: | :---: | :---: | :---: | :---: |
| Session | 5     | 7     | 6     | 5     | 5     |

## 3&nbsp; Cite

If you find this code or our GED paper helpful for your research, please cite our paper:

```bibtex
@inproceedings{huang2022graph,
  title     = {Graph Emotion Decoding from Visually Evoked Neural Responses},
  author    = {Huang, Zhongyu and Du, Changde and Wang, Yingheng and He, Huiguang},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention},
  year      = {2022}
}
```
