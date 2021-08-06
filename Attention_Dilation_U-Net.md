[Two-Stage Approach for Segmenting Gross Tumor Volume in Head and Neck Cancer with CT and PET Imaging](https://www.programmersought.com/article/36287421048)

[Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images](https://arxiv.org/pdf/1808.08114.pdf)

We propose a novel attention gate (AG) model for medical image analysis that automatically learns to focus on target structures of varying shapes and sizes.
Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while highlighting salient features useful for a specific task. This enables
us to eliminate the necessity of using explicit external tissue/organ localisation modules when using convolutional neural networks (CNNs). AGs can be easily
integrated into standard CNN models such as VGG or U-Net architectures with
minimal computational overhead while increasing the model sensitivity and prediction accuracy.

We show that the proposed attention mechanism can provide efficientobject localisation while ***improving the overall prediction performance by reducing false positives.***

### Segmentation 관건 => False Positives ↓, True Negatives ↑
                         
    (FP : 병변이 아닌 부분을 병변이라 예측, TN : Background인 부분을 Background라고 예측)
    

### How Attention Mechanisms work

The idea of attention mechanisms is to generate a context vector which assigns weights on the input sequence. Thus, the signal highlights the salient feature of the sequence conditioned on the current word while suppressing the irrelevant counter-parts, making the prediction more contextualised.

Here, we demonstrate that the same objective can be achieved by integrating attention gates (AGs) in a standard CNN model. This does not require the training of multiple models and a large number of extra model parameters. In contrast to the localisation model in multi-stage CNNs, AGs progressively suppress feature responses in irrelevant background regions without the requirement to crop a ROI between networks.

### Attnetion in Medical Imaging (Pooling => Spatial Context Disadvantage => Let's Use Attention Mechanism) 

In the context of medical imaging, however, since most objects of interest are highly localised, flattening may have the disadvantage of losing important spatial context. In fact, in many cases a few max-pooling operations are sufficient to infer the global context without explicitly using the global pooling. Therefore, we propose a grid attention mechanism. The idea is to use the coarse scale feature map before any flattening is done.
