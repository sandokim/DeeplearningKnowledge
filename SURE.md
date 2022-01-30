Thus, it is important for deep-learning-based denoisers to use high quality noiseless ground truth data for
high performance. However, it is often challenging or even infeasible to obtain noiseless images in some applications. Here, we
propose a method based on Stein’s unbiased risk estimator (SURE) for training DNN denoisers based only on the use of noisy images
in the training data with Gaussian noise. We demonstrate that our SURE-based method, without the use of ground truth data, is able to
train DNN denoisers to yield performances close to those networks trained with ground truth for both grayscale and color images. We
also propose a SURE-based refining method with a noisy test image for further performance improvement. Our quick refining method
outperformed conventional BM3D, deep image prior, and often the networks trained with ground truth. Potential extension of our
SURE-based methods to Poisson noise model was also investigated

### K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Image denoising by sparse 3-D transform-domain collaborative filtering,” 

Deep learning based image denoisers have yielded performances that are equivalent to or better than those of
conventional state-of-the-art denoising techniques such as BM3D
