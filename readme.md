# Table of Content

* Few Shot Learning Literature
  * General Setup and Datasets
  * Siamese Networks for One Shot Learning 
  * Matching Networks 
  * Meta-Agnostic Model Learning 
  * Triplet Networks 
  * Prototypical Networks 
  * Activations to Parameters 
  * Weight Imprinting 
* Human Robot Interaction Setting 
  * Differences to Few Shot Literature 
  * KUKA Innovation Challenge 

# Few Sot Learning using HRI

Few Shot Learning, the ability to learn from few labeled samples, is a vital step in robot manipulation. In order for robots to operate in dynamic and unstructured environments, they need to learn novel objects on the fly from few samples. The current object recognition methods using convolutional networks are based on supervised learning with large-scale datasets such as ImageNet, with hundreds or thousands labeled examples. However, even with large-scale datasets they remain limited in multiple aspects, not all objects in our lives are within the 1000 labels provided in ImageNet. 

As humans we can hold the object and check it from different viewpoints and try to interact with it to learn more about the object. Thus the robot should be able to teach itself from the few samples for the different object viewpoints. If we are aiming as well at human centered artificial intelligence, a natural step is to teach robots about their environment through human robot interaction. A human teacher can show the object with different poses and verbally instruct the robot on what it is and how it can be used. 

<div><img src="objects.png" class="img-responsive" alt=""> </div>

## Few Shot Learning Literature:
What motivated me to write on this topic was working on the KUKA innovation challenge, I was part of team Alberta that were in the 5 finalists. It turned out to be an exciting way of understanding the problem. While surveying and reading papers can give you the understanding of what the literature are working on. However, some new problems from working on the demo popped up that we realized are still lacking from the literature and my intention is to share these. 


## General Setup and Datasets:
The few shot learning is formulated as a **m shot n way** classification problem, where **m is the number of labeled samples per class**, and **n is the number of classes** to classify among. Two main datasets are used in the literature:
* Omniglot Dataset [1], the few-shot version of MNIST. It is a character recognition dataset which contains 50 alphabets, each alphabet has around 15 to 40 characters, and each character is produced by 20 drawers. 
* Mini ImageNet dataset [2] on the other hand is a more realistic setting. 100 random classes from ImageNet are chose, with 80 for training and 20 for testing.  
* In a newer work in CVPR'18 [3] instead of evaluating on the few-shot set solely, evaluating on both few-shot set and the large-scale set data on the whole ImageNet data with the 1000-way accuracy was reported.

<div><img src="omniglot.png" class="img-responsive" alt=""> </div>

### Siamese Networks
Metric learning methods have the advantage that they rapidly learn novel concepts without retraining. 

One approach is to learn a mapping from inputs to vectors in an embedding space where the inputs of the same class are closer than those of different classes. Once the mapping is learned, at test time a nearest neighbors method can be used for classification for new classes that are unseen. A siamese network is trained with the output features fed to a Contrastive Loss [4].

<div><img src="cl.png" width="50%" class="img-responsive" alt=""> </div>

Y label is 0 for similar class samples, and 1 for dissimilar. D_w is the distance function that is euclidean distance. So the loss will decrease the distance D when the samples are from the same class, on the other hand when they are dissimilar it will try to increase D with a certain margin m.


Triplet Loss [5]

 


One of the earliest attempts was using siamese networks [6]. It formulated the few shot learning problem as a **verification task**. A siamese network consists of two twin networks with shared weights which is then merged and the similarity between features is learned through stochastic gradient descent. 
The distance function used to learn the similarity can be in the form of a contrastive loss, or a weighted L1 distance. In Koch et al. paper they utilized the weighted L1 distance followed by a sigmoid function and cross entropy loss . 

Figure .. shows the network architecture used and an unofficial keras code is provided here.

### Matching Networks

### MAML

### Triplet Networks

### Activations to Parameters

### Imprinted Weights

## HRI Setting:
The fundamental differences between human robot interaction and the current few shot learning setting are: 
1. the abundance of temporal information for the different poses of the object. 
2. the hierarchy of category, different instances/classes within the same category, and different poses. 
3. The open set nature of the problem, which requires the identification of unknown objects. 
4. Different challenges introduced by the cluttered background, the different rigid and non-rigid transformations, occlusions and illumination changes. 
5. the continual learning of objects.


[1] Lake, Brenden, et al. "One shot learning of simple visual concepts." Proceedings of the Annual Meeting of the Cognitive Science Society. Vol. 33. No. 33. 2011.

[2] Vinyals, Oriol, et al. "Matching networks for one shot learning." Advances in Neural Information Processing Systems. 2016.

[3] Qiao, Siyuan, et al. "Few-shot image recognition by predicting parameters from activations." CoRR, abs/1706.03466 1 (2017).

[4] Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality reduction by learning an invariant mapping." null. IEEE, 2006.

[5] Hoffer, Elad, and Nir Ailon. "Deep metric learning using triplet network." International Workshop on Similarity-Based Pattern Recognition. Springer, Cham, 2015.


Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." ICML Deep Learning Workshop. Vol. 2. 2015.

