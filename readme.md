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

# categorical cross-entropy

a loss function that corresponds to the maximum likelihood solution of a multiclass classification problem. 

It is defined as: H(t; p) = 􀀀
P
t(x) log(p(x)), p and t corresponding to “prediction” and “target”.

Turning to the least frequent two classes

“middle” and “distal”, it becomes clear that the network trained on categorical cross-entropy is far

less capable of learning to detect highly infrequent classes, which can be explained by the fact that

categorical cross-entropy corresponds to a maximum likelihood solution: the network gets biased

towards more frequent classes, as this increases the likelihood of the training data.

It was shown before in Table 1 that the classes “middle” and “distal” are the least common of all four foreground

labels due to their smaller size. While the class imbalance could be countered by using weight maps

wherein less frequent classes are associated with higher weights, this would introduce an additional

hyper-parameter optimization problem. In order to avoid this, the Jaccard distance was used in all

of the ensuing experiments.

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

Next, the network was trained and tested using two different types of long skip connections: concatenation

and element-wise summation. Six cross-validation folds were used in order to get a sound

comparison. The training and validation curves of all of the folds are collected in Fig. 11. The results

suggest that the summation network is outperformed in all cross-validation folds. In order to confirm

this, the test set dice scores obtained by the networks were averaged over all six cross-validation

folds. These values, along with the curve of the validation loss achieved by each network averaged

over all folds are visualized in Fig. 12. Like the single-fold learning curves, these plots suggest that

concatenation works better than element-wise summation when used in long skip connections.

### Performance Comparison : Concatenation > Summation > Cross Validation

[CNN-based Segmentation of Medical Imaging Data](https://arxiv.org/pdf/1701.03056.pdf)
