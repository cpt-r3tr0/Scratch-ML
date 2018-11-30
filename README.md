# Machine Learning From Scratch

## About
Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.

The purpose of this project is not to produce as optimized and computationally efficient algorithms as possible
but rather to present the inner workings of them in a transparent and accessible way.

## Table of Contents
- [Machine Learning From Scratch](#machine-learning-from-scratch)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Examples](#examples)
    + [Polynomial Regression](#polynomial-regression)
    + [Classification With CNN](#classification-with-cnn)
    + [Density-Based Clustering](#density-based-clustering)
    + [Generating Handwritten Digits](#generating-handwritten-digits)
    + [Deep Reinforcement Learning](#deep-reinforcement-learning)
    + [Image Reconstruction With RBM](#image-reconstruction-with-rbm)
    + [Evolutionary Evolved Neural Network](#evolutionary-evolved-neural-network)
    + [Genetic Algorithm](#genetic-algorithm)
    + [Association Analysis](#association-analysis)
  * [Implementations](#implementations)
    + [Supervised Learning](#supervised-learning)
    + [Unsupervised Learning](#unsupervised-learning)
    + [Reinforcement Learning](#reinforcement-learning)
    + [Deep Learning](#deep-learning)
  * [Contact](#contact)

## Installation
    $ git clone https://github.com/cpt-r3tr0/Scratch-ML.git
    $ cd Scratch-ML
    $ python setup.py install

## Examples
### Polynomial Regression
    $ python scratchml/examples/polynomial_regression.py


### Classification With CNN
    $ python scratchml/examples/convolutional_neural_network.py

    +---------+
    | ConvNet |
    +---------+
    Input Shape: (1, 8, 8)
    +----------------------+------------+--------------+
    | Layer Type           | Parameters | Output Shape |
    +----------------------+------------+--------------+
    | Conv2D               | 160        | (16, 8, 8)   |
    | Activation (ReLU)    | 0          | (16, 8, 8)   |
    | Dropout              | 0          | (16, 8, 8)   |
    | BatchNormalization   | 2048       | (16, 8, 8)   |
    | Conv2D               | 4640       | (32, 8, 8)   |
    | Activation (ReLU)    | 0          | (32, 8, 8)   |
    | Dropout              | 0          | (32, 8, 8)   |
    | BatchNormalization   | 4096       | (32, 8, 8)   |
    | Flatten              | 0          | (2048,)      |
    | Dense                | 524544     | (256,)       |
    | Activation (ReLU)    | 0          | (256,)       |
    | Dropout              | 0          | (256,)       |
    | BatchNormalization   | 512        | (256,)       |
    | Dense                | 2570       | (10,)        |
    | Activation (Softmax) | 0          | (10,)        |
    +----------------------+------------+--------------+
    Total Parameters: 538570

    Training: 100% [------------------------------------------------------------------------] Time: 0:01:55
    Accuracy: 0.987465181058


### Density-Based Clustering
    $ python scratchml/examples/dbscan.py



### Generating Handwritten Digits
    $ python scratchml/unsupervised_learning/generative_adversarial_network.py

    +-----------+
    | Generator |
    +-----------+
    Input Shape: (100,)
    +------------------------+------------+--------------+
    | Layer Type             | Parameters | Output Shape |
    +------------------------+------------+--------------+
    | Dense                  | 25856      | (256,)       |
    | Activation (LeakyReLU) | 0          | (256,)       |
    | BatchNormalization     | 512        | (256,)       |
    | Dense                  | 131584     | (512,)       |
    | Activation (LeakyReLU) | 0          | (512,)       |
    | BatchNormalization     | 1024       | (512,)       |
    | Dense                  | 525312     | (1024,)      |
    | Activation (LeakyReLU) | 0          | (1024,)      |
    | BatchNormalization     | 2048       | (1024,)      |
    | Dense                  | 803600     | (784,)       |
    | Activation (TanH)      | 0          | (784,)       |
    +------------------------+------------+--------------+
    Total Parameters: 1489936

    +---------------+
    | Discriminator |
    +---------------+
    Input Shape: (784,)
    +------------------------+------------+--------------+
    | Layer Type             | Parameters | Output Shape |
    +------------------------+------------+--------------+
    | Dense                  | 401920     | (512,)       |
    | Activation (LeakyReLU) | 0          | (512,)       |
    | Dropout                | 0          | (512,)       |
    | Dense                  | 131328     | (256,)       |
    | Activation (LeakyReLU) | 0          | (256,)       |
    | Dropout                | 0          | (256,)       |
    | Dense                  | 514        | (2,)         |
    | Activation (Softmax)   | 0          | (2,)         |
    +------------------------+------------+--------------+
    Total Parameters: 533762


### Deep Reinforcement Learning
    $ python scratchml/examples/deep_q_network.py

    +----------------+
    | Deep Q-Network |
    +----------------+
    Input Shape: (4,)
    +-------------------+------------+--------------+
    | Layer Type        | Parameters | Output Shape |
    +-------------------+------------+--------------+
    | Dense             | 320        | (64,)        |
    | Activation (ReLU) | 0          | (64,)        |
    | Dense             | 130        | (2,)         |
    +-------------------+------------+--------------+
    Total Parameters: 450


### Image Reconstruction With RBM
    $ python scratchml/examples/restricted_boltzmann_machine.py


### Evolutionary Evolved Neural Network
    $ python scratchml/examples/neuroevolution.py

    +---------------+
    | Model Summary |
    +---------------+
    Input Shape: (64,)
    +----------------------+------------+--------------+
    | Layer Type           | Parameters | Output Shape |
    +----------------------+------------+--------------+
    | Dense                | 1040       | (16,)        |
    | Activation (ReLU)    | 0          | (16,)        |
    | Dense                | 170        | (10,)        |
    | Activation (Softmax) | 0          | (10,)        |
    +----------------------+------------+--------------+
    Total Parameters: 1210

    Population Size: 100
    Generations: 3000
    Mutation Rate: 0.01

    [0 Best Individual - Fitness: 3.08301, Accuracy: 10.5%]
    [1 Best Individual - Fitness: 3.08746, Accuracy: 12.0%]
    ...
    [2999 Best Individual - Fitness: 94.08513, Accuracy: 98.5%]
    Test set accuracy: 96.7%



### Genetic Algorithm
    $ python scratchml/examples/genetic_algorithm.py

    +--------+
    |   GA   |
    +--------+
    Description: Implementation of a Genetic Algorithm which aims to produce
    the user specified target string. This implementation calculates each
    candidate's fitness based on the alphabetical distance between the candidate
    and the target. A candidate is selected as a parent with probabilities proportional
    to the candidate's fitness. Reproduction is implemented as a single-point
    crossover between pairs of parents. Mutation is done by randomly assigning
    new characters with uniform probability.

    Parameters
    ----------
    Target String: 'Genetic Algorithm'
    Population Size: 100
    Mutation Rate: 0.05

    [0 Closest Candidate: 'CJqlJguPlqzvpoJmb', Fitness: 0.00]
    [1 Closest Candidate: 'MCxZxdr nlfiwwGEk', Fitness: 0.01]
    [2 Closest Candidate: 'MCxZxdm nlfiwwGcx', Fitness: 0.01]
    [3 Closest Candidate: 'SmdsAklMHn kBIwKn', Fitness: 0.01]
    [4 Closest Candidate: '  lotneaJOasWfu Z', Fitness: 0.01]
    ...
    [292 Closest Candidate: 'GeneticaAlgorithm', Fitness: 1.00]
    [293 Closest Candidate: 'GeneticaAlgorithm', Fitness: 1.00]
    [294 Answer: 'Genetic Algorithm']

### Association Analysis
    $ python scratchml/examples/apriori.py
    +-------------+
    |   Apriori   |
    +-------------+
    Minimum Support: 0.25
    Minimum Confidence: 0.8
    Transactions:
        [1, 2, 3, 4]
        [1, 2, 4]
        [1, 2]
        [2, 3, 4]
        [2, 3]
        [3, 4]
        [2, 4]
    Frequent Itemsets:
        [1, 2, 3, 4, [1, 2], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 4], [2, 3, 4]]
    Rules:
        1 -> 2 (support: 0.43, confidence: 1.0)
        4 -> 2 (support: 0.57, confidence: 0.8)
        [1, 4] -> 2 (support: 0.29, confidence: 1.0)


## Implementations
### Supervised Learning
- [Adaboost](scratchml/supervised_learning/adaboost.py)
- [Bayesian Regression](scratchml/supervised_learning/bayesian_regression.py)
- [Decision Tree](scratchml/supervised_learning/decision_tree.py)
- [Elastic Net](scratchml/supervised_learning/regression.py)
- [Gradient Boosting](scratchml/supervised_learning/gradient_boosting.py)
- [K Nearest Neighbors](scratchml/supervised_learning/k_nearest_neighbors.py)
- [Lasso Regression](scratchml/supervised_learning/regression.py)
- [Linear Discriminant Analysis](scratchml/supervised_learning/linear_discriminant_analysis.py)
- [Linear Regression](scratchml/supervised_learning/regression.py)
- [Logistic Regression](scratchml/supervised_learning/logistic_regression.py)
- [Multi-class Linear Discriminant Analysis](scratchml/supervised_learning/multi_class_lda.py)
- [Multilayer Perceptron](scratchml/supervised_learning/multilayer_perceptron.py)
- [Naive Bayes](scratchml/supervised_learning/naive_bayes.py)
- [Neuroevolution](scratchml/supervised_learning/neuroevolution.py)
- [Particle Swarm Optimization of Neural Network](scratchml/supervised_learning/particle_swarm_optimization.py)
- [Perceptron](scratchml/supervised_learning/perceptron.py)
- [Polynomial Regression](scratchml/supervised_learning/regression.py)
- [Random Forest](scratchml/supervised_learning/random_forest.py)
- [Ridge Regression](scratchml/supervised_learning/regression.py)
- [Support Vector Machine](scratchml/supervised_learning/support_vector_machine.py)
- [XGBoost](scratchml/supervised_learning/xgboost.py)

### Unsupervised Learning
- [Apriori](scratchml/unsupervised_learning/apriori.py)
- [Autoencoder](scratchml/unsupervised_learning/autoencoder.py)
- [DBSCAN](scratchml/unsupervised_learning/dbscan.py)
- [FP-Growth](scratchml/unsupervised_learning/fp_growth.py)
- [Gaussian Mixture Model](scratchml/unsupervised_learning/gaussian_mixture_model.py)
- [Generative Adversarial Network](scratchml/unsupervised_learning/generative_adversarial_network.py)
- [Genetic Algorithm](scratchml/unsupervised_learning/genetic_algorithm.py)
- [K-Means](scratchml/unsupervised_learning/k_means.py)
- [Partitioning Around Medoids](scratchml/unsupervised_learning/partitioning_around_medoids.py)
- [Principal Component Analysis](scratchml/unsupervised_learning/principal_component_analysis.py)
- [Restricted Boltzmann Machine](scratchml/unsupervised_learning/restricted_boltzmann_machine.py)

### Reinforcement Learning
- [Deep Q-Network](scratchml/reinforcement_learning/deep_q_network.py)

### Deep Learning
  + [Neural Network](scratchml/deep_learning/neural_network.py)
  + [Layers](scratchml/deep_learning/layers.py)
    * Activation Layer
    * Average Pooling Layer
    * Batch Normalization Layer
    * Constant Padding Layer
    * Convolutional Layer
    * Dropout Layer
    * Flatten Layer
    * Fully-Connected (Dense) Layer
    * Fully-Connected RNN Layer
    * Max Pooling Layer
    * Reshape Layer
    * Up Sampling Layer
    * Zero Padding Layer
  + Model Types
    * [Convolutional Neural Network](scratchml/examples/convolutional_neural_network.py)
    * [Multilayer Perceptron](scratchml/examples/multilayer_perceptron.py)
    * [Recurrent Neural Network](scratchml/examples/recurrent_neural_network.py)
