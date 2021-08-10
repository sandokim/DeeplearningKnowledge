[Two-Stage Approach for Segmenting Gross Tumor Volume in Head and Neck Cancer with CT and PET Imaging](https://www.programmersought.com/article/36287421048)

[Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images](https://arxiv.org/pdf/1808.08114.pdf)

[Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)

We propose a novel attention gate (AG) model for medical image analysis that automatically learns to focus on target structures of varying shapes and sizes.
Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while highlighting salient features useful for a specific task. This enables
us to eliminate the necessity of using explicit external tissue/organ localisation modules when using convolutional neural networks (CNNs). AGs can be easily
integrated into standard CNN models such as VGG or U-Net architectures with
minimal computational overhead while increasing the model sensitivity and prediction accuracy.

We show that the proposed attention mechanism can provide efficientobject localisation while ***improving the overall prediction performance by reducing false positives.***

### Segmentation 관건 => False Positives ↓, True Negatives ↑
                         
    (FP : 병변이 아닌 부분을 병변이라 예측, TN : Background인 부분을 Background라고 예측)
    

### How Attention Mechanisms work

![image](https://user-images.githubusercontent.com/74639652/128440435-11c37b5f-a13e-45a1-8ae4-49cd2f0730a9.png)

The idea of attention mechanisms is to generate a context vector which assigns weights on the input sequence. Thus, the signal highlights the salient feature of the sequence conditioned on the current word while suppressing the irrelevant counter-parts, making the prediction more contextualised.

Here, we demonstrate that the same objective can be achieved by integrating attention gates (AGs) in a standard CNN model. This does not require the training of multiple models and a large number of extra model parameters. In contrast to the localisation model in multi-stage CNNs, AGs progressively suppress feature responses in irrelevant background regions without the requirement to crop a ROI between networks.

### Attnetion in Medical Imaging (Pooling => Spatial Context Disadvantage => Let's Use Attention Mechanism) 

In the context of medical imaging, however, since most objects of interest are highly localised, flattening may have the disadvantage of losing important spatial context. In fact, in many cases a few max-pooling operations are sufficient to infer the global context without explicitly using the global pooling. Therefore, we propose a grid attention mechanism. The idea is to use the coarse scale feature map before any flattening is done.

![image](https://user-images.githubusercontent.com/74639652/128439684-66e9775d-7dc3-436f-bb8a-5aadd52681c3.png)

Information extracted from coarse scale is used in gating to disambiguate irrelevant and noisy responses in input feature-maps. For instance, ***in the U-Net architecture, gating is performed on skip connections right before the concatenation to merge only relevant activations.***
***Additionally, AGs filter the neuron activations during the forward pass as well as during the backward pass. Gradients originating from background regions are down weighted during the backward pass. This allows model parameters in shallower layers to be updated mostly based on spatial regions that are relevant to a given task.***

***Coarse feature-maps capture contextual information and highlight the category and location of foreground objects. Feature-maps extracted at multiple scales are later merged through skip connections to combine coarseand fine-level dense predictions as shown in Figure 3***

![image](https://user-images.githubusercontent.com/74639652/128440312-318f07f3-f937-4b98-a086-dce697248af4.png)

 The proposed AGs ar incorporated into the standard U-Net architecture ***to highlight salient features that are passed through the skip connections.***
 
For AGs, we chose sigmoid activation function for normalisation: σ2(x) = 1/(1+exp(−x)). While in image captioning (Anderson et al., 2017) and classification (Jetley et al., 2018) tasks, the softmax activation function is used to normalise the attention coefficients σ2, however, sequential use of softmax yields sparser activations at the output. For dense prediction task, ***we empirically observed that sigmoid resulted in better training convergence for the AG parameters.***

We empirically found that attention gates were less effective if applied to the earliest layer. We speculate that this is because first few layers only represent low-level features, which is not discriminative yet to be attended. The proposed architecture is shown in Figure 4.

### 3D-CT Model => Batch Size 2 or 4

The batch size for the Sononet models was set to 64. However, for the 3D-CT segmentation models, gradient updates are computed using small batch sizes of 2 to 4 samples. For larger segmentation networks, gradient averaging is used over multiple forward and backward passes. This is mainly because we propose a 3D-model to capture sufficient semantic context in contrast to the state-of-the-art CNN segmentation frameworks (Cai et al., 2017; Roth et al., 2018).

### Further Study

Similarly, ***residual and dense connections can be used as in (Gibson et al., 2017) in conjunction with AGs to improve the segmentation results.*** In that regard, our 3D Attention U-Net model performs similar to the state-of-the-art, despite the input images are downsampled to lower resolution. More importantly, our approach significantly improves the results compared to single-model based segmentation frameworks (see Table 4). We do not require multiple CNN models to localise and segment object boundaries. Lastly, we performed 5-fold cross-validation on the CT-82 dataset using the Attention U-Net for a better comparison, which achieved 81.48 ± 6.23 DSC for pancreas labels

## AG-Sononet Reduces false positive examples because the gating mechanism suppresses background noise and forces the network to make the prediction based on class-specific features.

In general, AG-Sononet improves the results over Sononet at all capacity levels. In particular, AG-Sononet achieves higher precision.

