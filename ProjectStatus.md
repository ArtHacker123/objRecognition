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




##Problems Faced##

...When we trained the new dataset containing 4 objects, the algorithm always predicted the background class.

Like the good researchers that we are, we need to identify an experimental procedure to resolve this Problem.

Lets first identify where the Problems could potentially lie!

1.  *Problem:* Size of the codebook is too low and the noise added with the size of the background class.  
2.  *Problem:* Simply the size of the Background class is too high and the SVM overfits to this class.
    ...This is the most obvious reason that we can think of right now.
    ...**To resolve this issue we have a couple of ideas in mind which we are going to implement now.**
    ...*Idea1: Use Bootstrapping as a sampling method*
    ...*Idea2: Differential Error Class - Use a greater weight for misclassification from the foreground classes.*
    ...The Problem with this idea is that our existing scenario can not afford false negatives. If we increase the weight for misclassification from this class we might end up with some false positives. We still have to test this.
    **We thought of using cross validation to get the weights of the classes in the Kernel SVM but we decided against it as it is 9 additional parameters to be optimized and using non parametric methods would be the better way to go.**
    ...*Idea3: Ensemble learning methods*
    ...*Idea4: Use a z-SVM*
    ...*Idea5: Modify the kernel of the SVM*
3.  *Problem* The SVM parameters are not optimized for the problem
    ...**We just realized that the size of the Codebook could be a significant problem for us since the compression is creating a lot of noise. We decided to use cross validation to identify good parameters for the size of the codebook as well as the SVM parameters. We should have done this earlier and we did this for the first scenario. Lack of time made us hard code some parameters. This seems to have created a lot of problems. This is** *Priority 1* **for us now. This would however take a lot of time. We will train these on the server and that would reduce the amount of time taken.**
 4. *Problem* The SVM overfits to our data.
    ...We already had plans for changing the Classification from an SVM to a Random Gaussian Ensemble or a Random Forest.
    ...This we would only try if everything else with the SVM fails or if we need better comparison for our classifiers. We already have a neural network and SVM voting for the final result. The parameters have not been optimized. If we check that the parameters are optimized then we eliminate this as the problem completely.



##Plans for 17.02.2016##

In the order of Priority

1. Use cross validation for setting up parameters.  **Priority 1**
  ..1. Preparing the data to be divided into training, test and validation data. **Before Noon**
2. Identify a solution for class imbalances.
  ..1. Read on Bootstrapping methods for sampling
  ..2. Read on Boosting and Ensemble learning methods.

##Plans for later.##
3. Combining features.
