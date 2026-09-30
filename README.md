# SegClass-CD
Segmentation-Guided Region-Based Classification of Crohn’s Disease Imaging Findings from Multi-Contrast MRE.
This paper was accepted as an oral presentation in the MICCAI adjacent workshop CLIP (Clinical image-based procedures).

Overview: 
This is a classification model for 8 ileal findings for Crohn's disease, based on automatic segmentation, eliminating the need for manual ROI extraction. The model uses Coronal T1 and T2 images, together with coronal T2 segmentations for the ileum (T1 segmentations are lacking in original dataset).

Data:
The dataset used in this study contains sensitive patient information and cannot be shared publicly
due to privacy and confidentiality regulations. 
Depending on your project structure, you may want to change the "make_subjects" function in utilities to fit your needs.
Depending on the number of masks you have in your original data, you  might want to change the  "ChangeLabels" function in utilities to keep the segments you  want (in our dataset, the ileum was label 2 out of a total of 8 labels).

Requirements:
python >= 3.12
nnunetv2 >= 2.6

Citetation:
@inproceedings{
STeitz2026segclasscd,
title={SegClass-{CD}: Segmentation-Guided Region-Based Classification of Crohn's Disease Imaging Findings from Multi-Contrast {MRE}},
author={Sarah Teitz, Naama Gavrielov, Leah Gitelman, Gili Focht, Ruth Cytter-Kuint, Talar Hagopian, Elena Vainberg, Israel Cohen, Dan Turner, and Moti Freiman},
booktitle={15th MICCAI Workshop on Clinical Image-based: Towards Holistic Patient Models for Personalised Healthcare},
year={2026},
url={https://openreview.net/forum?id=7HGk6Fsz8d}
}



