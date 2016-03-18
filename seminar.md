__TOC__
<div id=start></div>
[[Image:Logo_IRMA.png|85px|link=:Category:Project Human Robot Interaction 2015-16]]
[[:Category:Project Human Robot Interaction 2015-16 | <span style="font-size:110%;vertical-align:middle;">Project page</span>]]
<font style="font-size:300%;vertical-align:middle;"> | </font>
[[Image:RelatedWork_icon.png|50px|link=Project Human Robot Interaction 2015-16 RelatedWork]]
[[Project Human Robot Interaction 2015-16 RelatedWork | Related Overview]]
<font style="font-size:300%;vertical-align:middle;"> | </font>
[[Image:Modules_icon.png|50px|link=Object_Recognition]]
[[Object_Recognition | <span style="font-size:110%">Module Work</span>]]

==Team==
* Leena Chennuru Vankadara
* Faiz Ul Wahab

<div style="border:1px solid #ff0000; background-color:#FFFFFF; margin-bottom:1.0em; padding:0.2em 0.8em 0.4em 0.8em; margin-right:10px; color:#ff0000;">
You have a lot of material, I mean two approaches, one almost ready and starting a new one. Try to reflect that here two.
</div>

==Introduction==

Object recognition is the process of acquiring the image or the video from sensors, identify regions of interest in the image or the frame and classify the objects into one of the categories already learned by the system. In addition to this we also process the depth information we get to identify the 3D location of the object in space.

<div style="border:1px solid #ff0000; background-color:#FFFFFF; margin-bottom:1.0em; padding:0.2em 0.8em 0.4em 0.8em; margin-right:10px; color:#ff0000;">
Also explain how this would be useful for the project. Explain the scenario from the perspective of this module.
</div>

==Module description==
The module is responsible for image acquisition, preprocessing, segmentation of the scene and then finding contours for the relevant objects. For each contour found, we computing the region belonging to that contour. Thereafter, we compute some features on the object region and give it for recognition to the linear Support Vector Machine(SVM). Structurally, the module is composed of the following packages/sub-modules:
 * Object Detection by Segmentation
 * Feature Extraction
 * CodeBook Generation
 * Training Phase
 * Testing

==Methods==

Which methods have you worked on
==Segmentation==
Segmentation is a very important problem in computer vision. Although the idea of segmentation is to group the image into meaningful regions, it is very unclear as to what is meant by the term 'meaningful'. Also, it has different interpretation in video, image and audio analysis domains and the mean varies across tasks and different scenario and perspective. In our task for object detection, segmentation is an important task for the localization of the object in the scenario.

There is alot of literature work present in the area of segmentation. These can be divided into gradient ascent based techniques, graph based, filtering or derivative based techniques and threshold based techniques. In addition, a combination of different levels of segmentation as well as different types of segmentations are used with a clever strategy for finding different regions in an efficient manner.

Filtering based techniques are used for segmentation for a coarse purpose of finding edges in an image. Canny Edge Detection is a very useful method for finding edges in the images. This initial division into find regions can be used as input for finding contours and used for feature extraction and classification. Although finding edges can be susceptible to getting noisy edges and thus cannot be robust. An additionally preprocessing step that should be done is do a gaussian blur and then find meaningful edges. Bilateral Filtering is another approach that does noise remove from the image, and smooths out the image, while preserving the edges. This is a very important because simply smoothing out the image might lead to a loss of alot of information, specifically edges. But even this might have alot of noisy edges, so we added adaptive thresholding on to to make it more robust and efficient for finding good contours. Gaussian Blur is another approach that we used so that we can use a segmentation, although this approach also uses adaptive thresholding on top for robustness.

In graph based segmentation approaches, there is alot of literature like graph cuts, normalized cuts for segmentation using criteria like intensity differences, neigborhood function and partitioning the minimum spanning trees to partition the graph into meaningful subgraphs. Another approach is to represent the image as a graph and then do spectral clustering on the image using different cost functions. A few examples of these methods are mean shift filtering based on spectral clustering, normalized cuts, and spectral segmentation using multiscale graph decomposition. In gradient ascent methods, there is watershed segmentation, where it starts from local minima to form watersheds. Mean Shift filtering also uses a similar gradient ascent approach.

In addition to this, we have threshold techniques that local as well as global in scale. Otsu Binarization is one such methods that find  global threshold and based onto divides the image into different regions. Also, adaptive threshold has also been used for segmentation of grayscale image into meaningful regions. These threshold based techniques are common into scene segmentation as well as segmentation of document images alike.

==Feature Extraction and Feature Descriptions==
Feature Extraction or Detection as used in the literature is also a very challenging problem. For example deciding what kind or type of features to be compute vary from task to task as well as varies across types of data. Feature Detection in binary images or grayscale images or color images or video is a totally different topic. In addition, finding an efficient feature description for a problem is another big task.

Local as well as global features have been seen in the literature. In local features, we have simple edges detection as corner detection. But even in this we have Sobel operators, Marr operators, Canny edge detection. In corner detection, we have Harris corner detection, Foerstner corner, Rohr corner detection. In multimedia retrieval, Gist Features are global feature representation used in the literature for scene representation and multimedia retrieval of content from recorded videos or images.

For object detection specifically, there is SIFT(Scale Invariant Feature Tranform) Feature detection and descriptor that finds discriminative features in an image by constructing a scale space. Variations of SIFT exists like SURF and ORB, which are used for efficiency in terms of memory and time complexity. Histogram based features representation have also been there in the literature such as Histogram of Oriented Gradients, which are successful in person detection as well as general object detection.

Feature Extraction is basically extraction of important locations in the image based on some criteria while feature description is describing these features in terms of its local neighborhood. Locality based features are normally accompanied by a dictionary of features extraction from all the training images. This dictionary is named as codebook in the literature and different sizes and representation of the codebook are used for different tasks and types of features.

==Analysis==

What is your analysis ?

==Conclusions==
What are your conclusions

==Future Work==
How can this be improved

==Resources and References==

[[#start | (Back to the top) ]]
