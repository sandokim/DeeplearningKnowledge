## Segmantation 관련 논문 입니다

---

# Head and Neck Tumor Segmentation Challenge 2020

[Overview of the HECKTOR Chanllenge at MICCAI 2020: Automatic Head and Neck Tumor Segmentation in PET/CT](https://www.researchgate.net/publication/348453198_Overview_of_the_HECKTOR_Challenge_at_MICCAI_2020_Automatic_Head_and_Neck_Tumor_Segmentation_in_PETCT)

[Automatic segmentation of head and neck tumors and nodal metastases in PET-CT scans](https://openreview.net/pdf?id=1Ql71nEERx)

[Squeeze-and-excitation normalization for automated delineation of head and neck primary tumors in combined PET and CT images](https://arxiv.org/pdf/2102.10446.pdf)

[Combining CNN and hybrid active contours for head and neck tumor segmentation](https://arxiv.org/pdf/2012.14207.pdf)

[Two-stage approach for segmenting gross tumor volume in head and neck cancer with CT and PET imaging](https://www.programmersought.com/article/36287421048/)

---

[Automatic Head and Neck Tumor Segmentation in PET/CT with Scale Attention Network](https://www.medrxiv.org/content/10.1101/2020.11.11.20230185v1.full.pdf)

### Encoding Path Way

The encoding pathway is built upon ResNet [16] blocks, where each block consists of two Convolution-Normalization-ReLU layers followed by additive identity
skip connection.

***We keep the batch size to 1 in our study to allocate more GPU
memory resource to the depth and width of the model, therefore, we use instance normalization, i.e., group normalization [21] with one feature channel in
each group, which has been demonstrated with better performance than batch
normalization when batch size is small.***

In order to further improve the representative capability of the model, we add a squeeze-and-excitation module [14] into
each residual block with reduction ratio r = 4 to form a ResSE block. The initial
scale includes one ResSE block with the initial number of features (width) of 24.
We then progressively halve the feature map dimension while doubling the feature width using a strided (stride=2) convolution at the first convolution layer of
the first ResSE block in the adjacent scale level. All the remaining scales include two ResSE blocks and the endpoint of the encoding pathway has a dimension of 384 × 8 × 8 × 8.

### Decoding Path Way

The decoding pathway follows the reverse pattern of the encoding one, but with
a single ResSE block in each spatial scale. At the beginning of each scale, we use
a transpose convolution with stride of 2 to double the feature map dimension
and reduce the feature width by 2. ***The upsampled feature maps are then added
to the output of SA-block. Here we use summation instead of concatenation for
information fusion between the encoding and decoding pathways to reduce GPU
memory consumption and facilitate the information flowing.*** 

The endpoint of the decoding pathway has the same spatial dimension as the original input tensor and its feature width is reduced to 1 after a 1 × 1 × 1 convolution and a sigmoid function. In order to regularize the model training and enforce the low- and middle level blocks to learn discriminative features, we introduce deep supervision at
each intermediate scale level of the decoding pathway. Each deep supervision subnet employs a 1 × 1 × 1 convolution for feature width reduction, followed by
a trilinear upsampling layer such that they have the same spatial dimension as the output, then applies a sigmoid function to obtain extra dense predictions.
These deep supervision subnets are directly connected to the loss function in order to further improve gradient flow propagation

---

[Iteratively refine the segmentation of head andneck tumor in FDG-PET and CT images]

[Patch-based 3D UNet for head and neck tumor segmentation with an ensemble of conventional and dilated convolutions]

[GAN-based bi-modal segmentation using mumfordshah loss: Application to head and neck tumors in PET-CT images]

[The head and neck tumor segmentation using nnU-Net with spatial and channel squeeze & excitation blocks]

[Tumor segmentation in patients with head and neck cancers using deep learning based-on multi-modality PET/CT images]

[Oropharyngeal Tumour Segmentation using Ensemble 3D PET-CT Fusion Networks for the HECKTOR Challenge]

### Hecktor Challenge 2020 review 논문에서 시사하는 바

According to these criteria, the task is partially solved. The first criterion, evaluating the segmentation at the pixel level, is fulfilled. At the occurrence level
(criteria 2 and 3), however, even the algorithms with the highest DSC output FP and FN regions. These errors are generally made in very difficult cases and
we should further evaluate their source, e.g. Figure 2c and 2d. Besides, there is still a lot of work to do on highly related tasks, including the segmentation of
lymph nodes, the development of super-annotator ground truth as well as the agreement of multiple annotators, and, finally, the prediction of patient outcome
following the tumor segmentation.

Following the analysis of poorly segmented cases, we identified several key elements that cause the algorithms to fail. ***These elements are as follows; low
FDG uptake on PET, primary tumor that looks like a lymph node, abnormal uptake in the tongue and tumor present at the border of the oropharynx region.***
Some examples are illustrated in Figure 1. ***Understanding these errors will lead to better methods and to a more targeted task for the next iteration of this challenge.***

---

# U-Net

[Modality specific U-Net variants for biomedical image segmentation: A survey](https://arxiv.org/pdf/2107.04537.pdf)

---

U-Net의 한계점

[Integrating global spatial features in CNN based Hyperspectral/SAR imagery classification](https://arxiv.org/pdf/2107.04537.pdf)

Considering the present survey it is also observed that each modality requires a different approach to address the
corresponding challenges. Though there are segmentation approaches that are validated on multiple modalities to form generic architectures like nn-UNet, U-Net++, MRUnet, etc. but it is difficult to achieve optimal performance in all segmentation tasks. The main reason is due to the diverse variation in the features corresponding to the target
structures involving lungs nodule, brain tumor, skin lesions, retina blood vessels, nuclei cells, etc. and hence require different mechanism (dense, residual, inception, attention, fusion, etc.) to integrate with U-Net model to effectively learn the complex target patterns. Moreover, the presence of noise or artefacts in different modalities adds another factor to propose different segmentation methods.

---

# Biomedical Segmentation

[Introduction to 3D medical imaging for machine learning: preprocessing and augmentations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060809/pdf/nihms-590656.pdf)

[Multimodal Spatial Attention Module for Targeting Multimodal PET-CT Lung Tumor Segmentation](https://ieeexplore.ieee.org/document/9354983)

[A review on segmentation of positron emission tomography images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4060809/pdf/nihms-590656.pdf)

[Transparent reporting of biomedical image analysis challenges](https://arxiv.org/ftp/arxiv/papers/1910/1910.04071.pdf)

[Why rankings of
biomedical image analysis competitions should be interpreted with care](https://arxiv.org/ftp/arxiv/papers/1806/1806.02051.pdf)

[3D Deeply Supervised Network for Automatic Liver Segmentation from CT Volumes](https://arxiv.org/pdf/1607.00582.pdf)

[3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650.pdf)

[CNN-based Segmentation of Medical Imaging Data](https://arxiv.org/pdf/1701.03056.pdf)

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

[DeepOrgan Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation](https://arxiv.org/pdf/1506.06448.pdf)

[Efficient multi-scale 3d cnn with fully connected crf for accurate brain lesion segmentation](https://reader.elsevier.com/reader/sd/pii/S1361841516301839?token=36242A43B0FFCCEEE7986351EB960AF4DCF5707C08A2DE362B3B9C2A7C94EE7EE269998E928BB10443E47B0ACA0961D4&originRegion=us-east-1&originCreation=20210730103538)

[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)

[Improved Inception-Residual Convolutional Neural Network for Object Recognition](https://arxiv.org/ftp/arxiv/papers/1712/1712.09888.pdf)

[Inception Recurrent Convolutional Neural Network for Object Recognition](https://arxiv.org/pdf/1704.07709.pdf)

[Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf)

[On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task](https://arxiv.org/pdf/1707.01992.pdf)

[Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)

[SegNet A Deep Convolutional Encoder-Decoder Architecture fo Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

[V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/pdf/1606.04797.pdf)

[VoxResNet: Deep Voxelwise Residual Networks for Volumetric Brain Segmentation](https://arxiv.org/pdf/1608.05895.pdf)

[Discriminative unsupervised feature learning with convolutional neural networks](https://arxiv.org/pdf/1406.6909.pdf)

[Three-Dimensional Visualization of Medical Image using Image Segmentation Algorithm based on Deep Learning](http://koreascience.or.kr/article/JAKO202010163509916.page)
