__TOC__
<div id=start></div>
[[Image:Logo_IRMA.png|85px|link=:Category:Project Human Robot Interaction 2015-16]]
[[:Category:Project Human Robot Interaction 2015-16 | <span style="font-size:110%;vertical-align:middle;">Project page</span>]]
<font style="font-size:300%;vertical-align:middle;"> | </font>
[[Image:Modules_icon.png|50px|link=Project_Human_Robot_Interaction_2015-16_Modules]]
[[Project_Human_Robot_Interaction_2015-16_Modules | <span style="font-size:110%">Modules Overview</span>]]
<font style="font-size:300%;vertical-align:middle;"> | </font>
[[Image:RelatedWork_icon.png|50px|link=Object_Recognition_Seminar_Contributions]]
[[Object_Recognition_Seminar_Contributions | Related Work]]

==Team==
* Faiz Ul Wahab
* Leena Chennuru Vankadara

==Module description==
The goal of this module is object detection using RGB-D camera. In specific, create visual representations for pre-defined objects which can be generalized to new objects and train a classifier to learn the objects based on their visual representations to distinguish objects from non-objects. In addition, the module is also responsible to locate the 3D position of the objects relative to the robot.

==Hardware==
<div style="float:right;clear:right">
[[File:Xtion pro live 2.jpg|none|thumb|200px|Asus Xtion Pro Live]]
</div>
We use an ASUS XTION PRO LIVE to receive RGB images and the depth images. The launch file in the ROS package "turtlebot_bringup" which is named as "3dsensor.launch" publishes the relevant ROS Topics. We subscribe to the topic camera/rgb/image_rect_color/compressed for the RGB data and camera/depth/image_rect for the depth data. The camera can be placed at any height as long as it gets a clear view of the object to be recognized.

==Datasets==
We have created three different datasets to be tested with for the scenario. The first dataset consisted of three different synthetic objects, Apple, Banana and a Cube. Sample images from the dataset are shown here. We picked these objects for the sole reason that there was already data available from previous semester's projects which we could use and relieve us off the effort of creating a new dataset. However the algorithm that we implemented seemed to have failed on these objects even in simple backgrounds. Debugging this issue told us that since we have a xtion placed at a static height on top of the turtlebot which is unlike that of the Nao's camera the objects on the ground need to be placed at least 2 meters away from the base of the robot to be even visible to the camera. Added to that, a poor resolution of the xtion RGB sensor(640 x 480) loses most of the meaningful data from the objects leading to poor recognition results. During the process of debugging we attributed the poor recognition rate to collecting the dataset from Nao's camera and testing it on the xtion. Since a poor noise model can lead to low recognition rate, we decided to collect a new dataset in the scenario. Sample Pictures from this dataset are shown below.
<div style="float:right;clear:right">
[[File:Firstscenario.png|none|thumb|400px|First Scenario Dataset(Basket, Milk carton, Water Kettle]]
</div>
However, since the real issue behind this was loss of meaningful data due to hardware constraints, creating a new dataset did not improve the rate of recognition. Hence we decided to use bigger objects for our scenario, preferably with a certain amount of texture on them. We selected a basket, water kettle and a milk carton as our new objects. We created a new dataset of these three objects in simple backgrounds to be tested for the First scenario defined in the project.

For the complex scenario that was setup, we created another dataset with the same objects(basket, kettle and milk carton). Sample images from this dataset are shown below. We segmented the images and manually picked out the bounding boxes which contained the image and does not contain any other objects from the Background. The background class was populated with Bounding boxes which do not contain the objects.

The full dataset consisted of ~6000 images, 5000 of which belong to the background class, 300 belong to the basket, 200 to the water kettle and 200 to the milk carton.

The class imbalances in the dataset might create significant bias towards the background class. Hence we decided to replicate the data by mirroring the images and from rotations to create additional synthetic data for each of the object classes.

==Methodology==
To achieve this task we worked with two different approaches.

* The first approach uses a pipeline of classical Image Processing techniques in literature piped with ORB/SIFT feature descriptors and a histogram of Bag of Visual Words model for representation and a Kernel based Support Vector Machine/Random Forest as a classifier.

* The second approach uses a pipeline of classical Image Processing techniques piped with a Convolutional Neural Network both for representation of visual features as well as classification of the images.  

==Approach 1==
This approach implements Object Detection by the following the Training and Testing pipelines as described below.

The process includes image acquisition, preprocessing, segmentation of the scene and then finding contours for the relevant objects. For each contour found, we compute some features on the object region and give it for recognition to a classifier.

===Training Procedure===
<div style="float:right;clear:right">
[[File:Pipelinetest.png|none|thumb|500px|Processing of a sample image in the test phase]]
</div>
# Segmentation using Mean shift algorithm/Bilateral Filtering
# Adaptive Thresholding to convert to binary.
# Find contours on the images
# Draw bounding boxes if the contour area is > 500 pixels
# Compute SIFT features on the Regions of interest
# Create a codebook of these features
# Compress the codebook using kmeans clustering to reduce redundancy in features
# For each training image create a histogram of these features
# Use the histograms to train a Histogram Intersection Kernel SVM / Random Forest Classifier.

===Testing Procedure===
# Segmentation using Mean shift algorithm/Bilateral Filtering
# Adaptive Thresholding to convert to binary.
# Find contours on the images
# Draw bounding boxes if the contour area is > 500 pixels
# Compute ORB features on the Regions of interest
# For each RoI create a histogram of the computed features
# Use the SVM trained in the training procedure to predict the labels for the current histogram corresponding to a RoI

===Stages of the Pipeline(Approach 1)===
Structurally this approach comprises of the following stages.
# Segmentation
# Preprocessing
# Feature Extraction
# Preparing Training data
# Classification

The next sections describe the details of different stages of implementation


====Segmentation====

Segmentation in this pipeline works by a filtering stage followed by conversion to gray scale, Adaptive Thresholding to convert to Binary and finding contours on the resulting binary image. Bounding boxes are drawn for each contour and are filtered out based on the contour size. The pipeline for segmentation is depicted in the picture below.
<div style="float:right;clear:right">
[[File:Segmentation.png|none|thumb|200px|Segmentation Pipeline]]
</div>

=====Methods/Optimization strategies=====
5 different approaches were implemented and tested for the filtering stage of Segmentation. These methods were benchmarked on a dataset that was collected in the scenario for speed vs accuracy tradeoff.

# Meanshift Filtering
# Bilateral Filtering
# Watershed Segmentation
# Superpixel Segmentation
# Gaussian Blurring

=====Meanshift Filtering=====
Given a color image(RGB/HSV), for each pixel a spatial window is defined and the mean of the data point is computed and the position of the window is shifted to the mean of the window and this procedure is repeated under convergence.

The size of the window can be determined in a parametric fashion (shape of the window being the parameter to be optimized). This method is non Adaptive but can be optimized for a particular scenario using cross validation. Using cross validation via a pipelined approach resulted in a window shape of (11,11). Another way to define the size of the window is in a non parametric fashion by using a generalized kernel based density function estimation and then finding the local maxima of the density function.  

The result of Meanshift filtering is an image where each pixel is replaced by the the local maxima of the density function. This essentially is a gradient ascent algorithm and each pixel converges to a mode of the denisty function. This algorithm like any other gradient ascent algorithm assumes certain structure in the data space. This results in local noise removal resulting in a smooth intensity function. An example of Meanshift filtering applied to a sample image is shown below.
<div style="float:right;clear:right">
[[File:MeanshiftBeforeAfter.png|none|thumb|500px|Example of Meanshift Segmentation applied to an image]]
</div>
After this, the image result is converted to grayscale. This is followed by the adaptive thresholding which acts as a clustering method to give us the spatial distribution of distinctive regions in the images. We use the output of the adaptive threshold to find contours. Bounding boxes are drawn for each contour and each Bounding Box constitutes a region proposal.

The Segmentation process utilizes the color information of the Image to generate region proposals. Segmentation is an essential step in this pipeline since using a pyramid based sliding window approach generates a large number of regions which would be tested for the presence of the object. Since we require nearly real time performance for object detection in the project reducing the number of region proposals is very essential.

=====Bilateral Filtering=====
<div style="float:right;clear:right">
[[File:BilFilter0021.png|none|thumb|400px|Bilateral Filtering Result]]
</div>
Bilateral Filtering is an special kind of filtering with two characteristics of edge preserving and noise smoothing. Given a color image, a window is defined and the intensity/color of any pixel is replaced by the in the neighboring values with a certain weight in addition to considering the intensity of the pixel. This means that the around a pixel, the neighboring values are checked so that edges are preserved as well as noise is removed.

This kind of filtering is a combination of the domain and range filtering which is filtering based on euclidean distance of pixels and color similarity respectively. Thus, the value of a pixel is replaced by the nearby values with similar color values. it has two parameters:
* spatial window size
* range window size

After the bilateral filtering, the image is converted to grayscale and then adaptive threshold is applied on the image, which is again a clustering based on pixel locations. Based on this result contours are found. and against each contour, a bounding box is generated which goes for feature extraction and classification.

The advantages of this bilateral filtering is that it provides sufficiently good region proposals for classification. Also our segmentation is object independent so a generic segmentation based on filtering filtering is a really good prospect for really time deployment.

=====Superpixel Segmentation=====
Superpixel segmentation divides the image into meaningful atomic regions, which can be replaced by the rigid structure of the pixel grid, depending on the application. There are many approaches of generating superpixel regions and depending on the application they vary alot. There are three main consideration for our application in mind:
* Superpixels should adhere to the image boundaries.
* They superpixels generated, if as a preprocessing step, should be computationally efficient, memory efficient and easy to use.
* They superpixels generated for segmentation should increase the speed and the quality of the results of the subsequent steps.

There are two types of superpixel segmentation algorithms, namely graph-based and gradient ascent based. In graph based approaches, each pixel is treated as a graph node and edge weights are based on the similarity of the pixel to its neighborhood. Then superpixels are created by minimizing some cost function over the graph. Some graph based segmentation algorithms are Normalized cuts and Felzenszwalb.
Gradient ascent based superpixel algorithms work by having a initial rough clustering and then iteratively improve the clusters until some convergence criteria is met. Some gradient ascent methods for superpixels are quick shift, watershed and turbopixels.

The methods that we use for our module is SLIC(single linear iterative clustering). Simply speaking its a modified version of k-means for superpixel segmentation. It is computationally efficient by limiting the number of distance computations in k-means through directional sarch and the covergence criteria depends on the spatial neighborhood as well as intensity values in the image. We have used this with different hyperparameters for our segmentation. Then edge detection is done on the region to find contours. These contours are then given for feature extraction and classification.

The challenges faced using this approach is due to its parametric nature. In specific determining the number of superpixel clusters is a parameter to be set and because of the varying degree of clutter in the scenario, its hard to find an ideal parameter in spite of using cross validation. Also another challenges was to find a way to do a compromise because uniformity of superpixel regions, number of clusters, and modifying the weights given to neighborhood in space and intensity values.

=====Gaussian Filtering based segmentation=====
This is segmentation procedure that we defined for testing purposes only to enable realtime object detection. But this highly depends on a good segmentation procedure for finding good contours for the training of the system.

Basically, the idea is to apply a gaussian blur on the image. The window size is defined by testing on a subset of images contain equa representation of clutter and/or object, in our scenario. After apply a gaussian blur, we do adaptive threshold to find spatial consist regions. Then we find contours on this result feature extraction and classification. This approach work well on the realtime object detection deployed on the robot.

=====Felzenswab based Segmentation=====
This is one of the methods that performs segmentation by representing image as graph. A pixel is a node in the graph and the edge between two pixel is the measured by color dissimilarity of some sort. By representing image as a graph, segmentation S is partition of a graph such that each component corresponds to a connected component in the graph. Generally, this means that edges between two nodes in the same component should have relatively low weights and edges between two nodes in different components should relatively have higher weights. Additionally the algorithm looks for evidence of a boundary between two components and incorporates it into the cost function. Regions that have no evidence of an boundary between them are merge together into one component.

After doing segmentation, canny edge detection or a simple derivative mask is applied on the result and then contours are computed. This is then given for feature extraction and classification.

====Feature Extraction====

Feature Extraction is the most important stage in the pipeline and selecting features which are meaningful, rich in their descriptive power and satisfy the necessary trade-off between in-variance and selectivity. In specific, we need the features to be invariant to scale, rotation, view-point, perspective geometry and they need to be selective enough to distinguish the objects from the background.

We considered working with different feature types and experiment with them to test them for this scenario. We experimented with ORB Feature descriptor, SIFT descriptor, HOG descriptor and a Restricted Boltzmann Machine to learn features in an unsupervised fashion.

=====ORB Feature Detector and Descriptor=====

ORB uses a FAST feature extractor which essentially extracts corners and the BRIEF descriptor(32 Dimensional) to represent the features. The computational cost of computing ORB features is much lesser compared to that of SIFT and hence makes it effective to be used in a real time object detection application. ORB features are scale and rotation invariant. When tested on images with white background, a simple linear SVM could achieve a precision and recall of 0.91 and 0.90 respectively indicating that the features are rich enough to distinguish the objects from plain backgrounds as well as from each other. The first basic scenario required object detection in plain and simple backgrounds. However, as the complexity of the environment increased, the descriptive power of the features became lesser and lesser. With images collected from the more complex scenario which included images of a tree, a couch(with fluid texture), blinds, heaters, chairs, cupboards, tables made of wood, speakers, projector, screen, ceiling, the tables and a clock, the precision and recall drastically reduced to 0.11 and 0.22 respectively with a simple linear SVM. Modification of the Kernel of the SVM also did not improve the results significantly(precision - 0.11, recall 0.33). Hence it required us to select features with a better descriptive power at the cost of more computational time.

=====SIFT Feature Detector and Descriptor=====

SIFT features are rotation and scale invariant. With a 128-dimensional descriptor, the SIFT features were proven to be rich enough to distinguish between the objects from the background class as well as from each other.
====ROI Representation====
A SIFT descriptor represents a feature in an image. In order to classify a ROI as an object, this ROI needs to have a certain representation in some R(D) space. Such representations for the training set is used to train the classifier to learn the objects as well as in the testing procedure. The following three models have been implemented in the current scenario and have been evaluated with our Training and Test data.

=====Histogram of Bag of Visual Words Model=====
Discriminative approaches have the ability to represent highly complex boundaries. However, they usually require the input to be of a fixed length. Histogram of Bag of Visual Words model is one way to achieve a fixed length representation of every ROI, irrespective of the size of each ROI.

This approach is usually referred to in literature as the Codebook approach. The idea is as follows. Given a training set (X, Y), extract pre-defined features on every training image, and store the features in what is referred to as a codebook. The codebook simply contains all the features extracted from the images. Since there would be a great deal of redundancy in the features, a Vector Quantization method is applied in order to reduce the redundancy in the codebook. We call it the VQ Codebook. To create a representation for each training image, compute features on each training image, look for its nearest(pre defined distance function, preferably same as the one used to compress the Codebook) neighbor and replace the feature with that neighbor. Create a histogram for each training image whose length is the same as the size of the VQ Codebook and the values represent the frequency with which a feature of the VQ Codebook occurs in the image.

Note that this histogram would be sparse and it would also not take into consideration, the spatial structure of the features in the image(Hence the name bag of visual words). Each feature essentially is an image patch. Intuitively it represents the frequency with which each image patch occurs in the image. In spite of the fact that this representation destroys all spatial structure in the image, good results have been reported with this approach.

======Our Codebook======
The size of the VQ codebook and the method for Vector Quantization are the two decisions that were made. The size of the VQ codebook was set via cross validation for the entire pipeline and we used a linear K-Means clustering to perform Vector Quantization. Through cross validation, for the first scenario, the size of the VQ codebook was set to 750 and for the final scenario, it was set to 10,000.

=====Spatial Pyramid Matching Model=====
This approach is an extension to the simple Bag of Visual words model. Each ROI is represented in the following fashion according to this approach. Each ROI is represented at different levels of coarseness. Each level comprises of grids of the ROI and each grid of the lower layer is divided further into coarser grids in the next layers.
<div style="float:right;clear:right">
[[File:Spatial pyramid.jpg|none|thumb|400px|Spatial Pyramid Representation(Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories, S. Lazebnik, C. Schmid, and J. Ponce, CVPR 2006)]]
</div>

======Implementation======
At each level, a bag of visual words model is used to create a histogram of features for each grid. Histograms of all the grids are concatenated at each level to form a level histogram. All the level histograms are concatenated to form a ROI histogram. The following picture depicts the representation of an image in this approach. Image is represented at three different levels H0, H1, H2. H0 simply represents the histogram of Bag of Visual Words model. Since we have 10,000 Visual features in the codebook, the size of this codebook. At H1, the image is divided into 4 grids. For each grid we compute the histogram of the VQ Codebook features. Each grid histogram is a vector of length 10,000. The histograms are of each grid at H1 are concatenated to form a bigger histogram of length 4 x 10,000. Each level at H1 further divided into 4 grids at H2 and the resulting histogram has a length 16 x 10,000. The histograms of all the levels are concatenated together to form a histogram of length (1 + 4 + 16) x 10,000.

=====Graph Based Representation=====
Bag of Visual Words model destroys the spatial structure of the objects completely in order to achieve in-variance. Spatial pyramid matching attempts to achieve a trade off between in-variance and spatial structure preservation by creating histograms at different spatial locations of the ROI. It preserves a certain amount of spatial structure information of the features in the ROI.

However, as the scenario gets more and more complex, differentiating the objects from the background becomes more and more difficult for the algorithm since the same features could occur in the background classes and the objects in a different spatial structure. In order to circumvent this without having to make a trade off with in-variance, we have a graph based representation of the ROI(which is a structured collection of pre-computed features). The method works as follows.

<div style="float:right;clear:right">
[[File:Samplegraph.png|none|thumb|200px|A sample graph with labels.]]
</div>

For each ROI we create what we define as an '''Adjacency frequency Matrix'''. The computation of this matrix is done in the following fashion. Given an ROI, we compute the features on the ROI and look for the nearest neighbor of each feature in the VQ Codebook. We create a K-Nearest Neighbour graph of the features of each ROI. This forms a graph where each node has a label attribute. We create a matrix of shape [N(Codebook) x N(Codebook)]. Each element (i,j) in the matrix represents the frequency with the feature i of the VQ Codebook is a neighbor of feature j of the VQ Codebook.

An example of this representation is shown below. Given the graph as shown in the picture to the right, the following is its corresponding Adjacency frequency matrix.

<div style="float:center;clear:center">
[[File:matrix.png|none|thumb|100px|Adjaceny Frequency Matrix.]]
</div>

Note that this is a symmetric matrix and the size of the vector for representation is hence reduced by half. However depending on the size of the codebook, the matrix still could have a large O(n^2) vector size. However this is a sparse matrix and by simply concatenating the rows of the lower triangular matrix along with the diagonal elements, we have a fixed length representation for the ROI. This representation can be used in conjunction with any of the above two models.

We use this method in conjunction with the Visual Bag of words model to improve the accuracy of the model. We train a classifier with this representation to vote for the final result.

====Classification====
The problem o classification is basically assigning a image or region in the image to a set of object categories based. There are various methods used for classification like support vector machines, kernel(SVM), neural networks, linear discriminant analysis, and decision trees. In our module, we have implemented random forest classifier and Support Vector Machines(with different kernels).
=====Support Vector Machines=====
Support Vector Machines are supervised learning models that try to assign a set of points to two categories by finding the hyperplane between them. From two categories, this has been extended to multiple number of categories. Although the original support vector was a linear one, it has been extended to use different types of kernels specific to task and types of data. There will be many hyperplanes that separate the data, so an optimal criteria is maximizing the width of the margin while at the same time minimizing the classification error rate. This is the case in the soft margin SVM, which find a tradeoff between the maximizing width of the margin and minimizing the error rate. Kernel Trick is a tranforming the data into the features space and finding a non linear boundary between the set of points.

<div style="float:center;clear:center">
[[File:svm.png|none|thumb|400px|Soft margin Support Vector Machine]]
</div>
======Linear Kernel======
A linear support vector is a classifier with a linear kernel, which is defined as the dot product between two data points. Also this will find a linear boundary seperating the two categories. For the first scenario, we used a linear kernel SVM

======Radial Basis Functions Kernel======
In SVM, radial basis function is frequently used as kernel. The kernel is described by defining a gaussian of certain size on a point. Since, defining a gaussian will allow the value of gaussian to drop if the neighboring points, it can be taken as a similarity measure between two points. The transformation used in this kernel will allow to find a nonlinear decision function.

<div style="float:center;clear:center">
[[File:KernelSVM.png|none|thumb|400px|Decision function of a SVM with an Arbitrary kernel is the Sign of this function]]
</div>

======Histogram Intersection Kernel======
Given the extremely poor results of both a linear Kernel SVM as well as a RBF SVM on our dataset, one of the ways in which we attempted to improve the recognition accuracy was modifying the kernel of the SVM to better suit our approach. Since the size of the codebook is very large, the histogram is a sparse vector in a high dimensional space. Both the RBF as well as the Linear SVM rely on the Euclidean distance as a distance measure between histograms. This however is not very meaningful and hence we decided to use the Histogram Intersection Kernel for the SVM.

Given two histograms a(a1,a2,..,an), b(b1,b2,..,bn), where n is the number of bins in the histogram. Histogram Intersection Kernel measures the distance measure between a, b in the following way.

<div style="float:center;clear:center">
[[File:HIKernel.png|none|thumb|400px|Histogram Intersection Kernel]]
</div>

This function has been proven to be a mercer kernel in [Annalisa Barla1, Francesca Odone2, Alessandro Verri(Histogram Intersection Kernel for Image Classification)]. Hence this function can be used as a kernel for the Support Vector Machine.

A Support Vector Machine with Histogram Intersection Kernel generates a decision function which is sign of the function in the following picture.

<div style="float:center;clear:center">
[[File:Hiksvm.png|none|thumb|400px|Histogram Intersection Kernel SVM]]
</div>

======Pyramid Matching Kernel======
The Pyramid match kernel is a natural extension of the Histogram Intersection Kernel to adapt to the Spatial Pyramid Matching model. Each ROI is represented by histograms at multiresolutions and then the weighted histogram intersection computation is used to compare the two Histograms.

=====Random Forest Classifier=====

<div style="float:right;clear:right">
[[File:RandomForest.png|none|thumb|300px|Example of a Decision Tree]]
</div>
Random Forest Classifier works by constructing several decision trees and creating an ensemble of those decision trees. These decision trees are constructed in the following fashion.

Given N training Images, N Images are sampled at random with Bootstrapping(with replacement), from the original data. This sample forms the training set to build one decision tree. If the total number of features are M, then at each node m<<M (We used sqrt(M)) features are selected randomly and a split is made only based on these features. The depth of the tree can be fixed by pruning or the tree can be allowed to grow till the maximum depth. Decision trees are prone to gross over fitting and due to the Randomness of samples as well as the Randomness of the features at each node, Random Forests do not over fit.

Given that Random Forest looks at each features and can extract features which have higher significance in classification, using Random Forest is a viable alternative to the HIK SVM. Random forests are really fast at both the training and the testing phase.

==Approach 2==
This approach implements Object Detection by the following the Training and Testing pipelines as described below.

The process includes image acquisition, preprocessing, segmentation of the scene and then finding contours for the relevant objects. For each contour found, we draw a bounding box and resize the image to a standard size(100 x 100). Each image is given as an input to a Convolutional Neural Network for representation as well as classification.

===Convolutional Neural Network===
Convolutional neural networks are biologically inspired variations of multilayer layer perceptrons(MLP). In addition to representation  of hierarchically complex features, these networks have certain characteristics that make them really effective in presence of visual data. Due to the inherent 2D representation of the network, it preserves the inherit structure of the images and there is redundancy of weight sharing which is useful in visual data. The convolutional layers learn features like edges and corners in a local neighborhood, which is the receptive field of the filter used. Pooling layers introduce invariance in the feature extraction process. So basically we use feature extraction and classification implicitly in the convolutional neural networks. An important distinction of these networks is that feature engineering is not done and the kind of features depend on the data provided. So in these networks, different kinds of features do not have to computed by the types of features used are also learned form the data.

<div style="float:right;clear:right">
[[File:lenet5.png|none|thumb|600px|Topology of a typical CNN]]
</div>

We tried a number of different convolutional neural networks for object classification. The lower layers are features detectors and the last layers is used for classification of the features extracted.
====Topology of the Network====
Topology of the convolutional network defines the structure of the network itself. This comprises of the size of the input, the type of activation function used, the number of convolutional layers used, the type of pooling used and how many pooling layers used, the number of hidden layers used and the type of regularization used. Here are the different topologies used for object detection. The topology defines the receptive field of each filter and also defines the type of features learned. This is crucial to the architecture of the neural network as its fundamental for both feature extraction and classification. Every Convolutional Layer is followed by maxpooling layer of size(2,2). All networks use dropout for regularization and no data augmentation is used.

{| class="wikitable" style="border: 1px solid gray; style="border-collapse: collapse; background: #F2F2F2; color: black; margin-left: 10px;"  border="1px"  cellpadding="5"
|+  style="caption-side:bottom; text-align: left; color: solid gray"| Table 1: Different topologies of the neural network for experiments.
|-
|'''Type of Input'''
|'''No of Conv.Layers'''
|'''No of Hidden Layers(size)'''
|'''Activation Function'''
|-
|Grayscale
|2
|1(1000)
|ReLU
|-
|Whiten
|2
|1(1000)
|ReLU
|-
|Grayscale
|2
|1(1000)
|Tanh
|-
|Whiten
|2
|1(1000)
|Tanh
|-
|Grayscale
|2
|2(1000,500)
|ReLU
|-
|Whiten
|2
|2(1000,500)
|ReLU
|-
|Grayscale
|2
|2(1000,500)
|Tanh
|-
|Whiten
|2
|2(1000,500)
|Tanh
|}

====Feature Extraction and Representation====
The convolutional layers followed by the max pooling layers are used for feature extraction. The features themselves can be learned or they can also be hard coded. The common hard coded features for convolutional layers are Sobel Operators and Gabor Filters of different orientations. The filters can be learned by unsupervised pre-training using convolutional Restricted Boltzmann Machines(C-RBMs) or Convolutional Autoencoders(CA). For object detection in our scenario, we initialize the filters randomly. The output of the hidden layer before the output layer is used as feature representation for classification. The features and representations are learnt through back propagation.


<div style="float:right;clear:right">
[[File:FeaturesLearnt.png|none|thumb|400px|Features learned by the first layer.]]
</div>

====Classification====
The learning in the convolutional networks are done through back propagation. Batch gradient descent is used with a batch size of 20. Momentum is also used in the optimization. Momentum do

====Regularization Strategies====
There are many regularization strategies used in learning in the CNNs. Some of them are early stopping, Maxout units, L1, L2 regularizers and dropout. In our scenario, we use L1 regularizers because it leads to sparsity in the features learned which helps in classification. Dropout is also used in the hidden layers. This is because rather than using one CNN, we use an ensemble of networks. Both L1 and dropout strategies help in reducing overfitting.

<!--**Feature Extraction:
***Since each contour is a meaningful region that might contain a object, we find the bounding box containing the contour. A corner detector is used to find points that has corners. Then we use ORB feature descriptor to compute the descriptor on those points.


* Describe your module, what's its function within the project:
The module Object Recognition encapsulates the following flow of tasks. It Begins with image acquisition then we preprocess the image to a standard size and format for further processing. The next step would be to identify regions of interest which could be done in several different ways. Then we classify the regions of interest into a list of pre-defined classes.

Within the project we first start with simple object recognition of 2 3D printed objects and instead of using colour based segmentation we use hough transform, histogram thresholding and background subtraction. We then use a voting based system to achieve robust segmentation. The segmented images will be inputted to a simple linear svm for the final classification. In addition, we will also have a HOG implementation for backup to get features and it could be directly given to the linear svm for classification.

We more or less finalised on this scenario because 1) It is simple to implement as we have out of the box code for these functions in python libraries and image data is more or less available already from the previous year's projects. 2) We will most likely be implementing a voting system either for ensemble of classifiers or for good segmentation and nearly everything that we use here can be reused for a more complex scenario as well.

==Brainstorming for ideas
Possible scenarios, implementations...  
-->

==Evaluation==
evaluation

==Conclusion==
pros and cons, possibilities and limitations within the module, what have you concluded after the research

==Links and Resources==
Useful links to tutorials, software or papers

For the first milestone, we will try and implement everything using opencv and scikit-learn. For the second milestone we would like to implement a fast Region - CNN on Caffe(A deep learning framework from UCBerkeley). However there are a few hardware requirements to do so and we will adapt to a different or may be even a simpler method if necessary.

All the implementations are already available and we will use a plug and play approach and spend more time in improving the methods that already work well.

=== Frameworks and Libraries ===
Theano, TensorFlow, Caffe, Scikit-learn, openBLAS are various frameworks that we have worked with and these could be useful for anyone who wants to work in the interface of Deep Learning and Computer Vision.
There are also other toolkits available such as the INRIA toolkit.

These are some useful links to get started with these frameworks

* http://www.computervisiononline.com/software/inria-object-detection-and-localization-toolkit
* http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf
* https://github.com/rbgirshick/fast-rcnn
* https://www.tensorflow.org/tutorials
* http://deeplearning.net/software/theano/tutorial/
* http://www.openblas.net/

=== Papers ===

=== Services to Implement ===

# Needed services for integrating Object Recognition with Behavior
# Please change the names if you like here.

def object_recognitionget_state():
    raise Exception("NotImplementedException")
    #return state of the module

def object_recognition_obj_pos(obj_features):
    raise Exception("NotImplementedException")
    # return either not found or relative pos of the object
    # obj_features: feature array of an object

==Notes==
<references />

[[#start | (Back to the top) ]]
