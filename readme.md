# Data Augmentation
The transformations used by [35, 43, 32] include shifting, rotating and

scaling images, as well as augmenting grey values. One transformation cited in each of the three

papers is the application of a random deformation field. This random deformation field is applied to

the image at the beginning of each training iteration. Hence, in every epoch, the network is training

on a different version of the original data set.

# Max pooling vs Stride

Strided convolution is used in

place of max-pooling as it was found to yield slightly better results in the preliminary experiments.

---

# Dice Simliarity Coefficient and Jaccard index

Another limitation stems from the fact that both

the dice coefficient and the Jaccard index are only defined for binary maps.

---

# Segmentation Loss Function

DSC, Jaccard Loss, categorical cross-entropy, Focal Loss

---

# U-net Skip connection

U-net에서 Contracting path를 skip connection으로 Expanding path에 concatenate하는 이유는 image가 Downsampling하면서 잃어버리는 local details를 보전하기 위해서입니다.

An important part of the network described in U-Net are long skip connections forwarding feature

maps from the contracting stage to the expanding stage. These allow the network to recover

local details which are lost due to the usage of downsampling operations. In order to justify the

inclusion of long skip connections, the network was trained and tested on the hand MRI data set

described in Section 3 with and without long skip connections. The results of this experiment are

summarized in Fig. 8. It was revealed that removing long skip connections unambiguously worsens

the performance of the network.
