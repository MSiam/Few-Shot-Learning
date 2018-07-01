# Table of Content

* Few Shot Learning Literature 
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

If we are aiming at human centered artificial intelligence, a natural step is to teach robots about their environment through human robot interaction. A human teacher can show the object with different poses and verbally instruct the robot on what it is and how it can be used. My focus in this article is on how to enable robots to learn from few samples using human robot interaction.

## Few Shot Learning Literature:
The few shot learning literature in the computer vision community does not address the HRI setting, and lacks multiple aspects. So I thought to cover a brief explanation of the few shot learning literature, and then highlight the differences to the HRI setting. What motivated me to write on that was working on the KUKA innovation challenge, where I was part of team Alberta that were in the 5 finalists. It turned out to be an exciting way of understanding the problem. While surveying and reading papers can give you the understanding of what the literature are working on. However, some new problems from working on the demo popped up that we realized are still lacking from the literature and my intention is to share these. Lets start by first understanding the problem and the datasets used in the evaluation. The few shot learning is formulated as a m shot n way classification problem, where m is the number of labeled samples per class, and n is the number of classes to classify among. Two main datasets are used in the literature the Omniglot and the mini ImageNet datasets. The omniglot is the few-shot version of MNIST, it is a character recognition dataset which contains 50 alphabets, each alphabet has around 15 to 40 characters, and each character is produced by 20 drawers. Mini-ImageNet on the other hand is a more realistic setting with natural images being evaluated upon. 

### Siamese Networks
Metric learning methods have the advantage that they rapidly learn novel concepts without retraining. One of the earliest attempts was using siamese networks by Koch et al. and formulating the few shot learning problem as a verification task. A siamese network consists of two twin networks with shared weights which is then merged and the similarity between features is learned through stochastic gradient descent. The distance function used to learn the similarity can be in the form of a contrastive loss, or a weighted L1 distance. In Koch et al. paper they utilized the weighted L1 distance followed by a sigmoid function and cross entropy loss . Figure .. shows the network architecture used and an unofficial keras code is provided here.

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




