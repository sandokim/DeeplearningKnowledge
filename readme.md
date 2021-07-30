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
