# Project HRI 1516 Object_Recognition status #

##Current Pipeline##

**Training Procedure**

1. Segmentation using *Mean shift* algorithm
2. *Adaptive Thresholding* to convert to binary.
3. Find *contours* on the images
4. Draw bounding boxes if the *contour area is > 500 pixels*
5. Compute *ORB features* on the Regions of interest
6. Create a *codebook* of these features
7. *Compress the codebook* using *kmeans clustering* to reduce redundancy in features
8. For each training image create a *histogram* of these features
9. Use the histograms to train a *Kernel SVM*.

**Testing Procedure**

1. Segmentation using *Mean shift* algorithm
2. *Adaptive Thresholding* to convert to binary.
3. Find *contours* on the images
4. Draw bounding boxes if the *contour area is > 500 pixels*
5. Compute *ORB features* on the Regions of interest
6. For each RoI create a *histogram* of the computed features
7. Use the *SVM* trained in the training procedure to *predict the labels* for the current histogram corresponding to a RoI

##Stages of the current Pipeline##

1. Segmentation
2. Preprocessing
3. Feature Extraction
4. Preparing Training data
5. Classification

###Optimization strategies for Segmentation###

*Idea1: Use different Segmentation strategies*

####Tried and Tested####

1. Bilateral Filtering with Adaptive Thresholding
2. Gaussian Blurring with Adaptive Thresholding
3. Mean shift with Adaptive Thresholding
4. Watershed Segmentation
5. Super pixel Segmentation

The idea is to use Bilateral Filtering and Mean shift filtering in parallel on different machines to generate contours.
Extract features on RoI's from both contours and continue the process. **Priority 1**

*Idea2: Use a convolution network with bounding box regressors to identify RoI's*

*Note*
      It would only make sense if we need better Segmentation quality. The current Segmentation process is effective enough to deal with most objects.
      We can only implement this if Idea1 does not pan out. **Priority 4**

*Idea3: Bypass the Segmentation completely and use a sliding window approach*

      Our current research indicates that this would require more time for processing and hence might not be real time.
      This has not been tried yet but may be we should. This would circumvent the Segmentation process entirely.
      We could make this **Priority 2**

      We have a 640 x 480 image, we can publish 10 topics which contain the region and transmit it to different computers.
      We can make use of the parallel processing in the lab to do the processing.


*Idea4: Use a faster-RCNN to identify RoI's and perform Classification with the help of shared features*
      The good thing is we have labeled data already ready to train this network and all the necessary changes to be made have been made.

      The bad thing is we have no idea if this would work for our scenario if we start with pre trained models and if we start the training from scratch it might take very long to converge.

      **Priority 3**

###Optimization strategies for Feature Extraction###

*Idea1: Use HOG and ORB features together*

*Idea 2: Learn the features using an Energy based model such as an autoencoder or a Restricted Boltzmann machine/DBN*
...Sklearn has an implementation of a Bernouli RBM.
...We implemented DBN in Theano. We want to use it with a sliding window approach.
*Idea3: Use a pretrained convolutional network as a feature extractor*
...Stages in implementation
1. Get a strategy to get the segmented results

###Optimization strategies for Preparing Training data###

...We currently use simple k-means for clustering which does not perform very well on uniformly sampled data. Kmeans assumes that 

If we want to choose a better representation for our codebook instead of k-means, *Histogram intersection kernels* can be used.
Check this paper for additional materials. **Efficient and Effective Visual Codebook Generation Using Additive Kernels, Journal of Machine Learning Research 12 (2011) 3097-3118, Jianxin Wu,Wei-Chian Tan,James M. Rehg**

This is especially useful when we use Histogram based features, meaning that when we want to compare two histograms, the Eucledian distance is not useful anymore and HIK becomes an useful measure. If we use HOG features then using this kernel would be a good choice. The code for codebook Generation using HIK is available and is written in c++. Using python wrapper over c++ using boost.python we can use it code to get integrated with the existing pipeline code.

The link to this *LibHIK* library is [here] (https://sites.google.com/site/wujx2001/home/libhik)


###Optimization strategies for classifier###

...We started training the SVM using the scikit-learn library instead of the OpenCV library. Benchmarks show that its one of the fastest implementations of an SVM out there. The grid search in SVM from scikit-learn uses multiple threads and hence can be processed faster.


##Problems Faced##

...When we trained the new dataset containing 4 objects, the algorithm always predicted the background class. This mostly is the problem of class imbalances. To measure this we are changing the metrics of accuracy that we have been using. We now use the *precision*, *recall* and the *roc_auc_score* to evaluate our classifier.
...**Recollection exercise**:

1. *Precision* is the *ratio tp / (tp + fp)* where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
2. *Recall* is the ratio *tp / (tp + fn)* where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
3. *roc_auc_score* is the area under the curve created by plotting the recall against the precision at various threshold settings
4. *Confusion Matrix* Here the The diagonal elements of the matrix indicate the number of points for which the predicted label is equal to the true label and the off-diagonal elements are those that are incorrectly classified by the classifier.


Like the good researchers that we are, we need to identify an experimental procedure to resolve this Problem.

Lets first identify where the Problems could potentially lie!

1.  *Problem:* Size of the codebook is too low and the noise added with the size of the background class.
2.  *Problem:* Simply the size of the Background class is too high and the SVM overfits to this class.
    ...This is the most obvious reason that we can think of right now.
    ...**To resolve this issue we have a couple of ideas in mind which we are going to implement now.**
    ...*Idea1: Use Bootstrapping as a sampling method*
    ...Bootstrapping or any other random under sampling or over sampling method would not really work because of the Codebook compression step that we have in the Pipeline. When we do sampling and then perform a kmeans on the codebook of the feature points, similar feature points would get combined into the same feature point and thus removing the redundancy.
    ...*Idea2: Differential Error Class - Use a greater weight for misclassification from the foreground classes.*
    ...The Problem with this idea is that our existing scenario can not afford false negatives. If we increase the weight for misclassification from this class we might end up with some false positives. We still have to test this.
    **We thought of using cross validation to get the weights of the classes in the Kernel SVM but we decided against it as it is 9 additional parameters to be optimized and using non parametric methods would be the better way to go.**
    ...*Idea3: Ensemble learning methods*
        We now work with the Random Forest approach.
    ...*Idea4: Use a z-SVM*
    ...*Idea5: Modify the kernel of the SVM*
        This works.
        The main problem that we had was that Eucledian distance between two histograms is not a meaningful measure
3.  *Problem* The SVM parameters are not optimized for the problem
    ...**We just realized that the size of the Codebook could be a significant problem for us since the compression is creating a lot of noise. We decided to use cross validation to identify good parameters for the size of the codebook as well as the SVM parameters. We should have done this earlier and we did this for the first scenario. Lack of time made us hard code some parameters. This seems to have created a lot of problems. This is** *Priority 1* **for us now. This would however take a lot of time. We will train these on the server and that would reduce the amount of time taken.**
 4. *Problem* The SVM overfits to our data.
    ...We already had plans for changing the Classification from an SVM to a Random Gaussian Ensemble or a Random Forest.
    ...This we would only try if everything else with the SVM fails or if we need better comparison for our classifiers. We already have a neural network and SVM voting for the final result. The parameters have not been optimized. If we check that the parameters are optimized then we eliminate this as the problem completely.
    **Also try a 1 vs all classifier**



##Plans for 17.02.2016##

In the order of Priority

1. Use cross validation for setting up parameters.  **Priority 1**
  ..1. Preparing the data to be divided into training, test and validation data. **Before Noon**
  *StratifiedShuffleSplit* is used to divide the data. The folds are made by preserving the % of samples for each class.
2. Identify a solution for class imbalances.
  ..1. Read on Bootstrapping methods for sampling
  ..2. Read on Boosting and Ensemble learning methods.
  ...Found a toolkit with different oversampling and undersampling methods implemented. The name of the toolkit is unbalanced_dataset. The link to the toolkit is [here] (https://github.com/fmfn/UnbalancedDataset). **I installed it on my computer but this later needs to be installed on all machines. Remember this**
##Plans for later.##
3. Combining features.
  ... scikit-learn has a module which is known as pipeline. The feature union function in the module can combine different features together into new features.
  This can be used to combine RBM/ORB/HOG features. [Here] (http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py) is a link to the documentation on the same.  

4. *Literature read today:*

  ..1. http://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1422_2011-Galar-IEEE_TSMCc-Ensembles.pdf
  ..2. http://www.cs.cmu.edu/~efros/exemplarsvm-iccv11.pdf
  ..3. Profiling Python code
  ..4. How to optimize for speed scikit-learn
  ..5. Different metrics for model evaluation
  ..6. Crossvalidation split strategies.
  ..7. Kernel Approximation using Nystrom methods to improve Training/test speed of kernel SVM.
  ..8. SMOTE: Synthetic Minority Over-sampling Technique
  ..9. Editorial: Special Issue on Learning from Imbalanced Data Sets
  ..10. Class imabalance learning methods for SVM's.
  ..11. Sharing Visual features for multi class and multi view object detection.
  ..12. Creating Efficient Codebooks for Visual Recognition https://jurie.users.greyc.fr/papers/05-jurie-triggs-iccv.pdf


We have implemented the solutions that we thought were possible to solve the issue of class imbalances. We selectively subsampled the background class to make the size of the background class much smaller. Our current background class has about 1000 images. We also created more foreground data and now each of our foreground classes have about 700 images. The solution however did not work.

We also tried the Differential Error Class approach by weighting the classifier to be more biased towards classes with less training samples.

The fact that the algorithm does not predict anything but the background class tells us that this is not a problem of class imbalances anymore. The classifier can not distinguish between the objects and the background.

The first reason could be that the features computed are not rich enough




Pick all the files with a certain name from one folder and move them to another folder.

read the files from listing directory
get the basename of the files

get rootoutputname add the basename to it
 and copy the file name into destination


 Shape structure.

 For every training image, generate the list of features.

 Spatial pyramid matching

 Voronoi tesellation

 Gaussian Mixture models to represent the concatenation of histograms created from centers of Voronoi tesellation.


We are trying to reduce the computational cost as the time taken for processing is about 10 - 12 secs an image.

Idea 1: Use a smaller codebook (smaller size 50,000 instead of 75000)
implementing right now kmeans in progress. Then run the random forest training. Then test on a image for time results.
Idea 2: Use ORB instead of SIFT (lesser number of dimensions)
ORB features / descriptors are not rich enough to distinguish the objects either from each other or from clutter
Training data is already created. Only have to run
Idea 3: use parallel processing

implementation of all three in progress.
