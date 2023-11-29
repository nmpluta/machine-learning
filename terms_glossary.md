# **Glossary of Terms**
Below is a list of terms and definitions used throughout the course, they are organized by lesson. This list is not exhaustive, but it does cover the most important concepts. The defintions of these terms were generated using Open AI's [Chat GPT-3.5](https://chat.openai.com/) language model.

### **Lesson 1:** Simple DNN 1 - Getting started  
- **Deep Neural Network (DNN):** A type of artificial neural network with multiple layers (typically more than three). Deep neural networks are used in deep learning and are capable of learning complex hierarchical representations of data.

- **TensorFlow:** An open-source machine learning framework developed by the Google Brain team. It provides tools for building and training various machine learning models, including neural networks.

- **Keras:** An open-source high-level neural networks API written in Python. It acts as an interface for various deep learning frameworks, including TensorFlow. Keras simplifies the process of building and training neural networks.

- **Neural Network:** A computational model inspired by the structure and functioning of the human brain. It consists of interconnected nodes (neurons) organized in layers, with each connection having an associated weight. Neural networks are used for various machine learning tasks, including classification and regression.

- **MNIST Dataset:** A widely-used dataset in machine learning, consisting of a collection of 28x28 pixel grayscale images of handwritten digits (0 to 9). The MNIST dataset is often used as a benchmark for testing and developing machine learning algorithms.

- **Numpy:** A powerful numerical computing library in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.

- **Dense Layer:** In a neural network, a dense layer, also known as a fully-connected layer, connects each neuron in one layer to every neuron in the subsequent layer. It is a fundamental building block in designing neural network architectures.

- **Softmax Layer:** A layer in a neural network that applies the softmax activation function. It is often used as the output layer in classification problems to convert the raw output scores into probability distributions over multiple classes.

- **Compilation Step:** In the context of neural networks, the compilation step involves configuring the learning process. It includes specifying the optimizer, loss function, and metrics that the model will use during training.

- **Optimizer:** An algorithm or method used to adjust the weights of a neural network based on the calculated gradients, aiming to minimize the loss function. Examples include RMSprop and stochastic gradient descent (SGD).

- **Loss Function:** A measure of the difference between the predicted values of the model and the actual values in the training data. The goal during training is to minimize this function.

- **Metrics:** Evaluation criteria used to assess the performance of a model. In this context, accuracy (the fraction of correctly classified images) is the chosen metric.

- **Sequential Class:** In Keras, the `Sequential` class is a linear stack of layers, where you can simply add one layer at a time. It is a basic and straightforward way to build neural network architectures.

- **Data Preprocessing:** The transformation of raw data into a format suitable for training a machine learning model. This may include reshaping data, scaling values, and converting categorical variables into a suitable format.

- **Epochs:** In training a neural network, an epoch is one complete pass through the entire training dataset during the training process. The number of epochs influences how many times the model sees the entire dataset during training.

- **Batch Size:** In training a neural network, a batch is a set of samples used during training. The batch size is the number of samples used in one iteration. It is a hyperparameter that can be tuned for optimal performance.

- **Overfitting:** A phenomenon in machine learning where a model performs well on the training data but poorly on new, unseen data. It indicates that the model has memorized the training data rather than learning to generalize from it.

- **Model summary:** The model summary provides a concise representation of the neural network architecture, including layer types, output shapes, and trainable parameters.

### **Lesson 2:** Simple DNN 2 - Binary classification, Hyperparameters, Callbacks

- **Binary Classification:** A type of machine learning problem where the goal is to categorize items into one of two classes. In the context of neural networks, it involves training a model to make predictions with two possible outcomes.

- **IMDB Dataset:** A dataset containing 50,000 highly-polarized reviews from the Internet Movie Database (IMDB). They are split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews. Just like the MNIST dataset, the IMDB dataset comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) have been turned into sequences of integers, where each integer stands for a specific word in a dictionary.

- **One-Hot-Encoding:** A method of representing categorical data, such as word sequences in natural language processing. It involves converting categorical values into binary vectors, where each vector has a dimension equal to the number of categories, and only one bit is set to 1 to indicate the category.

- **ReLU (Rectified Linear Unit):** An activation function commonly used in neural networks. It replaces all negative values in the input with zero, allowing only positive values to pass through. ReLU is often used in intermediate layers to introduce non-linearity to the model.

- **Sigmoid Activation:** An activation function that squashes input values into the range [0, 1]. It is commonly used in the output layer of a binary classification neural network to produce probabilities. In the context of sentiment analysis, it indicates the likelihood of a review being positive.

- **Binary Crossentropy:** A loss function used in binary classification problems. It measures the "distance" between probability distributions, specifically between the ground-truth distribution and the model's predictions. In another sense, it measures the difference between the predicted probabilities and the true class labels.

- **Validation Set:** A subset of the dataset used during the training of a machine learning model to evaluate its performance on data it has never seen before. In this context, a validation set helps monitor the accuracy of the model during training and detect overfitting.

- **Overfitting:** A phenomenon in machine learning where a model performs well on the training data but poorly on new, unseen data. It indicates that the model has memorized the training data rather than learning to generalize from it.

- **Callbacks:** Functions in machine learning frameworks, such as Keras, that allow for specific actions to be taken at various points during training. Common uses include model checkpointing, early stopping, and dynamic adjustment of parameters.

- **Grid Search:** An optimization technique in machine learning used to search for the best hyperparameters of a model by systematically trying different combinations. It creates a grid of hyperparameter values and evaluates the model's performance for each combination.

- **Keras Autotuner:** A tool introduced in 2019 in Keras for automating the hyperparameter tuning process. It follows the same principles as grid search or random search but is integrated into the Keras framework.


### **Lesson 3:** ConvNet 1 - Introduction, Working with own dataset
- **ConvNet (Convolutional Neural Network):** A type of artificial neural network designed for processing structured grid data, such as images. ConvNets use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data.

- **Conv2D:** A convolutional layer in a neural network that performs a 2-dimensional convolution operation on the input, typically used for image processing tasks.

- **MaxPooling2D:** A pooling layer in a neural network that performs max pooling operation on the input, downsampling the spatial dimensions.

- **Filter Size:** The size of the convolutional filter or kernel, typically specified as width x height. Common filter sizes include 3x3 and 2x2.

- **Input Shape:** The shape or dimensions of the input data that is fed into a layer or model. For Conv2D layers in ConvNets, it is often specified as (image_height, image_width, image_channels).

- **Flatten Layer:** A layer in a neural network that transforms the input data from a 3D tensor to a 1D tensor, typically used to transition from convolutional layers to dense layers.

- **Rescaling:** A preprocessing step that involves scaling pixel values to a specified range, often [0, 1].

- **RMSprop (Root Mean Square Propagation):** An optimization algorithm used to adapt the learning rate during training. It helps in mitigating issues like vanishing or exploding gradients.

- **Categorical Crossentropy:** A loss function used in multi-class classification problems. It measures the distance between probability distributions, specifically between the predicted distribution and the true distribution of labels.

- **Learning Rate:** A hyperparameter that determines the step size at each iteration during the optimization process. It influences how quickly or slowly a neural network learns.

- **Outcome Analysis:** The examination and interpretation of the results obtained after training a machine learning model. This includes analyzing metrics such as accuracy, loss, and other relevant indicators.

- **Dropout:** A regularization technique in neural networks that involves randomly setting a fraction of input units to zero during each update. It helps prevent overfitting.

- **Weight Decay (L2 Regularization):** A regularization technique that adds a penalty term to the loss function based on the squared magnitude of model weights. It helps prevent overfitting by discouraging overly complex models.

- **Image_dataset_from_directory:** A function in TensorFlow used for creating a dataset from image files organized in directory structures. It is employed to efficiently load and preprocess images for training ConvNets.

- **ImageDataGenerator (replaced by image_dataset_from_directory):** In older versions of TensorFlow, a class in Keras that facilitated the preprocessing and augmentation of images for training neural networks.

- **Kaggle:** A platform for predictive modeling and analytics competitions. It hosts datasets and competitions in various domains, including machine learning.

### **Lesson 4:** ConvNet 2 - Data Augmentation, Dropout
- **Data Augmentation:** An approach in machine learning that involves generating more training data from existing samples by applying random transformations. In ConvNets, data augmentation, implemented using tools like Keras' `ImageDataGenerator`, helps expose the model to diverse aspects of the data, preventing overfitting.

- **ImageDataGenerator:** A class in Keras for efficiently preprocessing and augmenting images during training. It allows for various random transformations such as rotation, width and height shifts, shearing, zooming, and horizontal flipping.

- **Rotation Range:** A parameter in `ImageDataGenerator` that defines the range (in degrees, 0-180) within which to randomly rotate images. It introduces diversity to the training set.

- **Width Shift and Height Shift:** Parameters in `ImageDataGenerator` that specify ranges (as a fraction of total width or height) for random horizontal or vertical translation of images, respectively.

- **Shear Range:** A parameter in `ImageDataGenerator` for randomly applying shearing transformations to images, contributing to the augmentation process.

- **Zoom Range:** A parameter in `ImageDataGenerator` controlling the random zooming inside images during augmentation.

- **Horizontal Flip:** A parameter in `ImageDataGenerator` indicating whether to randomly flip half of the images horizontally. Useful when there are no assumptions of horizontal asymmetry in the data.

- **Fill Mode:** The strategy used in `ImageDataGenerator` for filling newly created pixels after a rotation or width/height shift.

- **Random Layers:** In ConvNets, layers such as `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, and `RandomContrast` used as part of data augmentation. These layers are added to the model before training to augment the input data.

- **RandomFlip:** A layer in ConvNets for randomly flipping images horizontally, contributing to data diversity.

- **RandomRotation:** A layer in ConvNets for applying random rotations to images during data augmentation.

- **RandomZoom:** A layer in ConvNets for randomly zooming inside images as part of the augmentation process.

- **RandomTranslation:** A layer in ConvNets for applying random horizontal and vertical translations to images.

- **RandomContrast:** A layer in ConvNets for introducing random contrast adjustments to augmented images.

- **Dropout Layer:** A regularization technique in neural networks involving randomly setting a fraction of input units to zero during each training update. The `Dropout` layer helps prevent overfitting by reducing interdependence among neurons.

- **Dropout Parameter:** The fraction of input units to drop in the `Dropout` layer during training. It is a hyperparameter that can be adjusted to control the extent of regularization.

- **Train Generator and Validation Generator:** Instances created using `ImageDataGenerator` for generating batches of augmented training and validation data, respectively. These generators are used during model training.

- **Accuracy:** A metric used to evaluate the performance of a model, representing the fraction of correctly classified images. Monitoring accuracy helps assess the model's generalization capability.

- **Densely-Connected Classifier:** The fully connected layer or layers in a neural network responsible for making final predictions. In ConvNets, the dropout layer is often added just before this classifier to combat overfitting.

- **Compile the Model:** The step in configuring a neural network that involves specifying the optimizer, loss function, and metrics to be used during training. It precedes the actual training process.

- **Hyperparameter Tuning:** The process of systematically adjusting hyperparameters, such as dropout rates and learning rates, to find the configuration that optimizes the model's performance.

- **Combined Methods:** Refers to the practice of combining multiple techniques, such as data augmentation and dropout layers, to enhance the performance of a ConvNet. This combination often yields better results than using each method individually.

### **Lesson 5:** Pre-trained network (VGG 16)

- **Pre-trained network:** A pre-trained network is a saved network trained on a large dataset, often for image classification tasks. The spatial feature hierarchy learned by such networks can act as a generic model of the visual world, making it useful for various computer vision problems. 

- **VGG 16 network:** VGG 16 is a pre-trained network trained on the ImageNet dataset for image classification. It consists of 16 layers, including 13 convolutional layers and 3 fully-connected layers. The convolutional layers are grouped into five blocks, with each block containing multiple convolutional layers and a max pooling layer. The fully-connected layers serve as the classifier.

- **Convolutional base:** The convolutional base of a convnet comprises convolutional and pooling layers responsible for feature extraction from input data.

- **Transfer learning:** Transfer learning involves using the convolutional base of a pre-trained network for feature extraction and then training a new classifier on top. The representations extracted from convolution layers depend on the layer depth, with early layers capturing generic features and higher layers learning more abstract concepts.

- **Feature maps:** Feature maps in a convnet are presence maps of generic concepts over an image, providing useful information regardless of the specific computer vision problem.

- **Densely-connected layers:** The densely-connected layers in a convnet serve as a classifier. However, the information they contain becomes specific to the classes the model was trained on, making them less useful for problems where object location matters.

- **Feature extraction:** Feature extraction involves using the convolutional base of a pre-trained network to extract features from new data. These features are then used to train a new classifier.

- **Fine-tuning:** Fine-tuning is a technique where a few top layers of a frozen model base are unfrozen, allowing joint training of the classifier and these layers. It adjusts more abstract representations to make them relevant for a new problem.

- **Freezing layers:** Freezing layers in a neural network involves preventing their weights from being updated during training. It is essential to preserve representations learned by the frozen layers.

- **Trainable weights:** Trainable weights are the parameters of a neural network that are updated during training to minimize the loss function.

- **Fine-tuning convolutional layers:** Fine-tuning involves selectively unfreezing and training specific layers in the convolutional base to adapt learned representations to a new problem.

- **Smoothed curves:** Smoothing curves involves reducing noise in plotted data by averaging adjacent points, providing a clearer visualization of trends.

### **Lesson 6:** Pre-trained VGG - Visualization

- **Interpretability:** Interpretability refers to the ability to understand and explain how a deep learning model makes decisions. In the context of deep learning, where models are often composed of many layers of interconnected neurons, interpretability involves techniques for visualizing internal workings and simplifying models to enhance transparency.

- **Heatmap:** A heatmap is a graphical representation of data where the individual values contained in a matrix are represented as colors. It is often used to visualize the output of an algorithm, such as a neural network, to understand its behavior.

- **CNN Visualization:** Convolutional Neural Network (CNN) visualization involves techniques to make the internal representations of convnets more accessible and interpretable. This includes visualizing heatmaps of class activation in an image and intermediate convnet outputs, aiding in understanding how the network identifies and transforms input data.

- **Class Activation Map (CAM):** A visualization technique that produces heatmaps of "class activation" over input images, indicating how important each location is for a specific output class. It helps understand which parts of an image led a convnet to its final classification decision.

- **Grad-CAM (Gradient-weighted Class Activation Mapping):** A specific implementation for generating CAM, using the gradients of the model's final prediction with respect to the activations of the last convolutional layer. It provides insights into the impact of different parts of the input data on the model's decision.

- **GradCAM++ (Grad-CAM++):** An extension of Grad-CAM, designed to provide better visualization and understanding of object localization in images.

- **ScoreCAM:** A visualization technique similar to GradCAM but uses the model's output logits instead of gradients to generate heatmaps, showcasing the impact of input data on the prediction.

- **Saliency Maps:** A visualization method displaying raw values of neural network output gradients, providing a simple way to understand the importance of input features.

- **Intermediate Activations:** Visualization of feature maps output by various convolutional and pooling layers in a network. It helps understand how input is decomposed into different filters learned by the network.

- **tf_keras_vis:** A library used for visualization, providing algorithms such as GradCAM, GradCAM++, ScoreCAM, and Saliency Maps.

- **Layer Outputs:** The outputs of individual layers in a neural network, visualized to understand the feature maps generated by different layers.

- **Activation Model:** A model that outputs the intermediate activations (feature maps) of various layers in a neural network when given a certain input. It is used for visualization purposes.

- **Experimentation Tasks:**
  - **Task 1:** Test visualization methods on images with multiple objects to observe the network's classification behavior.
  - **Task 2:** Visualize heatmaps for classes other than the one with the highest probability.
  - **Task 3:** Evaluate the performance on images with contradictory objects, like a dog and a cat.
  - **Task 4:** Experiment with different layer names to obtain diverse visualizations.
  - **Task 5:** Try layer names from early, middle, and final convolutional layers to observe differences.
  - **Task 6:** Reflect on understanding network behavior, considering varying numbers of images for different layers and the resemblance of some images to the input.

