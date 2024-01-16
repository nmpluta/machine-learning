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

### **Lesson 7:** Autoencoders and GANs

- **Generator Network:** In the context of GANs (Generative Adversarial Networks), the generator network takes a random vector from the latent space and decodes it into a synthetic image. It plays a crucial role in generating realistic images to fool the discriminator network.

- **Discriminator Network (Adversary):** The discriminator network assesses whether an input image is real (from the training set) or synthetic (generated by the generator network). It is in constant competition with the generator, adapting to the generator's evolving capabilities and setting a high standard for image realism.

- **GAN (Generative Adversarial Network):** GAN consists of a generator network and a discriminator network, each trained to outperform the other. The generator aims to produce synthetic images indistinguishable from real ones, while the discriminator strives to discern between real and generated images. The interplay between these networks leads to the generation of increasingly realistic images.

- **Encoding and Decoding:** The process of transforming input data into a different representation (encoding) and then reconstructing the original input from that representation (decoding). Autoencoders, like the one described, utilize encoding and decoding to learn latent representations.

- **Generator Network, Discriminator Network:** Components of a GAN architecture, where the generator creates synthetic data, and the discriminator assesses whether the data is real or generated.

- **Autoencoder:** An autoencoder is a neural network architecture comprising an encoder module that maps input data to a low-dimensional latent space and a decoder module that reconstructs the input data from the latent space. In the context of image generation, the decoder plays a role similar to the generator in GANs, generating images based on the learned latent representations.

- **Convolutional Autoencoder:** A variant of autoencoder designed for image inputs, utilizing convolutional neural networks (convnets) as both encoders and decoders. Convolutional autoencoders are preferred for image-related tasks due to their superior performance.

- **Latent Space:** A low-dimensional space of representations where any point can be mapped to a realistic-looking image. In the context of image generation, the latent space serves as the input for the generator or decoder in GANs or autoencoders, respectively.

- **Binary Crossentropy:** Binary crossentropy is a loss function commonly used in training autoencoders and GANs when dealing with binary classification problems. It measures the difference between predicted and true values for each binary output.

- **Adam Optimizer:** An optimization algorithm used to minimize the loss function during training. Adam combines ideas from RMSprop and Momentum, providing adaptive learning rates. It adjusts the learning rates of each parameter individually.

- **Visualizing Heatmaps:** Techniques for visualizing heatmaps, such as GradCAM and ScoreCAM, which help understand which parts of an input image contribute most to the final classification decision of a convnet. Introduced in Lesson 7 for interpretability in deep learning models.

- **tf.keras.datasets.mnist:** A dataset loader for the MNIST dataset, a collection of handwritten digits widely used for training various image processing systems.

- **tf_keras_vis:** A library used for visualization in deep learning, providing algorithms such as GradCAM, GradCAM++, ScoreCAM, and Saliency Maps. Introduced in Lesson 7 for visualizing pre-trained VGG models.

### **Lesson 8:** U-Net Segmentation

- **U-Net Segmentation:** U-Net is a convolutional neural network architecture designed for semantic segmentation tasks, such as image segmentation. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. The U-Net architecture consists of a contracting path (encoder) followed by an expansive path (decoder). The unique U-shape allows the network to capture contextual information and generate high-resolution segmentation masks. It is widely used in biomedical image segmentation and other applications.

  - **Contracting Path (Encoder):** The initial layers of the U-Net reduce spatial resolution and capture high-level features through convolutional and max-pooling operations. This path is responsible for extracting hierarchical representations.

  - **Expansive Path (Decoder):** The decoder upsamples the feature maps to recover spatial information and refine segmentation masks. Transposed convolutions and skip connections are employed to concatenate low-level features from the encoder, aiding precise localization.

  - **Skip Connections:** Skip connections, also known as residual connections, connect the corresponding layers between the contracting and expansive paths. These connections facilitate the flow of low-level features during upsampling, preventing loss of spatial information.

  - **Transposed Convolution:** Also known as fractionally strided convolution or deconvolution, this operation performs upsampling by inserting zeros between pixels and then applying convolution. It helps increase spatial resolution.

- **DataGen:** A custom generator class for loading images and their corresponding masks during training. It inherits from the `keras.utils.Sequence` class and implements the `__init__`, `__len__`, and `__getitem__` methods.

- **double_conv_block:** A helper function defining a double convolutional block with two consecutive convolutional layers. It takes an input tensor `x` and the number of filters `n_filters` as parameters.

- **downsample_block:** A helper function defining a downsampling block, which includes a double convolutional block followed by max-pooling and dropout operations. It takes an input tensor `x` and the number of filters `n_filters` as parameters.

- **upsample_block:** A helper function defining an upsampling block, which includes transposed convolutional and concatenation operations. It aligns the dimensions of the input tensor and the features from the corresponding downsampling block. It takes an input tensor `x`, the features from the downsampling block `conv_features`, and the number of filters `n_filters` as parameters.

- **get_model:** A function to create the U-Net model architecture. It takes the image size (`img_size`) and the number of classes (`num_classes`) as parameters and returns a U-Net model.

- **model_checkpoint_callback:** A callback to save the best model during training. It monitors the validation accuracy and saves the model with the highest accuracy as "best_model.h5."

- **SparseCategoricalCrossentropy:** A loss function suitable for multi-class segmentation problems where the target values are integers representing class labels.

- **Adam optimizer:** A popular optimization algorithm used for training deep learning models. It adjusts the learning rates of each parameter individually.

- **ModelCheckpoint:** A callback in Keras that saves the model after each epoch. It allows monitoring a specified metric on the validation set and saves the model when the metric improves.

- **train_test_split:** A function from the scikit-learn library used to split the dataset into training and validation subsets based on the paths of input images and target masks.

- **model.fit:** The method to train the U-Net model using the specified generator (`train_gen` and `val_gen`), loss function, optimizer, and callbacks. It runs for a specified number of epochs.

- **PIL:** Python Imaging Library, used for opening and manipulating image files.

### **Lesson 9:** Recurrent Neural Networks

- **Vectorization:** The process of converting sequences or lists of data into tensors. In the context of natural language processing (NLP), vectorization is crucial for preparing text data for input into neural networks.

- **One-Hot Encoding:** A method of vectorization where each word in a sequence is represented as a binary vector, with all elements set to zero except for the index corresponding to the word's position in the vocabulary.

- **Word Embedding:** A technique used for vectorization that involves mapping words into a continuous vector space. Word embeddings capture semantic relationships between words, allowing neural networks to understand the meaning of words and their contextual nuances.

  - **Embedding Matrix:** A matrix that contains the embeddings for all words in the vocabulary. It is used as a lookup table during training to convert word indices into dense vectors.

  - **Embedding Parameters:** The dimensions of the embedding space. The number of parameters is determined by the size of the vocabulary and the chosen embedding dimensionality.

- **Introduction to RNN:**
  Recurrent Neural Network (RNN) is a type of neural network architecture with internal memory. It processes sequences of inputs while maintaining a hidden state that captures information from previous inputs. RNNs are suitable for tasks involving sequential data, such as handwriting or speech recognition.

  - **Unrolled RNN:** A visual representation of how an RNN processes sequences over time, where each step represents an input and the network's hidden state.

  - **Return Sequences:** A parameter in recurrent layers like SimpleRNN in Keras. It determines whether the layer should return the full sequence of outputs for each timestep or only the last output for each input sequence.
  
  -  **SimpleRNN:** A type of recurrent layer in Keras that processes input sequences by iterating through the elements and maintaining a state. It can return either the full sequence of outputs or only the last output, depending on the `return_sequences` parameter.

- **LSTM (Long Short-Term Memory):** A type of recurrent layer designed to overcome the vanishing gradient problem in traditional RNNs. LSTMs have internal memory cells and gates that control the flow of information, making them effective for capturing long-term dependencies in sequential data.

- **Vanishing Gradient Problem:** A challenge in training deep neural networks where the gradients become extremely small during backpropagation, leading to slow or stalled learning.

- **Stacking Multiple Layers:** The practice of adding multiple recurrent layers one after the other to increase the representational power of a network. When stacking layers, it's essential to ensure that all intermediate layers return full sequences.

- **Input Padding:** In the context of sequence data, it involves adding zeros or other values to sequences to make them of equal length.

- **Validation Split:** The proportion of the training dataset used for validation during model training. It helps monitor the model's performance on unseen data and prevents overfitting.

- **IMDB Dataset:** The Internet Movie Database dataset used for sentiment analysis in this lesson. It contains movie reviews labeled with sentiment (positive or negative).

- **Task 1:** Training a simple recurrent network using an Embedding layer, SimpleRNN layers, and a Dense layer for binary classification on the IMDB movie review dataset.

- **Task 2:** Setting up a model using an LSTM layer and training it on the IMDB dataset for binary classification. The architecture includes an Embedding layer, LSTM layer, and a Dense layer.

### **Lesson 10:** Time Series

-  **Time Series Data:**
Time series data is a sequence of data points collected or recorded in a time-ordered fashion. In the context of deep learning, time series data presents a unique set of challenges and opportunities for predictive modeling.

-  **Weather Station at Max Planck Institute for Biogeochemistry Dataset:**
The weather time-series dataset recorded at the Weather Station at the Max Planck Institute for Biogeochemistry in Jena, Germany. This dataset includes 14 different quantities, such as air temperature, atmospheric pressure, humidity, wind direction, and more. The observations are recorded at 10-minute intervals over several years, specifically from 2009 to 2016.

-  **Data Normalization:**
Data normalization is the process of adjusting the scale of different variables to a standard range. In the context of time series data, normalization is crucial to ensure that all variables are on a similar scale, preventing one variable from dominating the learning process. In the provided example, data normalization involves subtracting the mean and dividing by the standard deviation for each variable independently.

#### **Lookback, Steps, and Delay:**
- **Lookback:** The number of observations that the model will consider in the past. In this context, it represents the number of timesteps the model will look back to make predictions.
- **Steps:** The interval between consecutive observations used for sampling the data. In this example, observations are sampled at one data point per hour.
- **Delay:** The number of timesteps into the future for which the model will predict. In this case, it is set to 144, corresponding to predicting the air temperature 24 hours in the future.

#### **Data Generator:**
A data generator is a function responsible for dynamically creating batches of input data and their corresponding targets during the model training process. The generator efficiently handles large datasets and allows the model to train on data that is generated on-the-fly rather than preloading the entire dataset into memory.

- **Generator Function Parameters:**
  - **data:** The input time series data.
  - **lookback:** The number of timesteps the model looks back.
  - **delay:** The number of timesteps into the future for prediction.
  - **min_index:** The minimum index of the data to be considered.
  - **max_index:** The maximum index of the data to be considered.
  - **shuffle:** A boolean parameter indicating whether to shuffle the data.
  - **batch_size:** The number of samples per batch.
  - **step:** The interval between consecutive observations for sampling.

- **Generator Function Output:**
  - **samples:** The input data samples, a 3D tensor.
  - **targets:** The target values for prediction.

#### **GRU (Gated Recurrent Unit):**
  - The GRU (Gated Recurrent Unit) layer is a type of recurrent neural network (RNN) layer that is particularly useful for sequence processing tasks. Introduced by Chung et al., GRU layers operate on the same principle as LSTM (Long Short-Term Memory) layers but are designed to be more computationally efficient.
  
    - **Key Characteristics:**
      - **Gating Mechanism:** GRU layers utilize gating units to selectively update and reset information in the hidden state, allowing them to capture long-term dependencies in sequential data.
      
      - **Simplified Architecture:** Compared to LSTM, GRU has a more streamlined architecture with a reduced number of parameters, making it computationally less expensive.
  
    - **Parameters:**
      - **Input Shape:** (None, features), where "None" represents variable-length sequences and "features" denote the number of input features at each timestep.
  
      - **Output Shape:** (None, units), where "units" is the dimensionality of the output space.
      
      - **Number of Parameters:** $3 \times \text{units} \times (\text{units} + \text{input\_dim} + 1)$, where "input\_dim" is the number of input features.

    - **Usage:**
      - GRU layers can be employed for sequence modeling and prediction tasks, capturing patterns and dependencies within sequential data.
  
    - **Example:**
      ```python
      from keras.models import Sequential
      from keras.layers import GRU
  
      model = Sequential()
      model.add(GRU(32, input_shape=(None, input_dim)))
      ```
  
    - **Training Tips:**
      - GRU layers are prone to overfitting, and incorporating dropout and recurrent dropout can be beneficial for regularization.
  
      - Experimentation with hyperparameters such as the number of units and training duration is recommended for optimal performance.
      
      - GRU layers are well-suited for tasks where capturing sequential dependencies is crucial, such as time series prediction and natural language processing.

#### **Neural Network Models:**
Different neural network models are employed to solve the time-series prediction problem, each with its unique architecture and characteristics.

- **Basic Approach Model:**
  - The basic approach involves a fully connected model with no recurrent layers.
  - It consists of a Flatten layer, followed by two Dense layers.
  - Mean Absolute Error (MAE) is used as the loss function for training.

- **GRU (Gated Recurrent Unit) Layer Model:**
  - The GRU layer, developed by Chung et al., is employed in this model.
  - The GRU layer is followed by a Dense layer.
  - RMSprop is used as the optimizer, and MAE is the loss function.

- **Stacked GRU Layers Model:**
  - This model includes multiple stacked GRU layers.
  - Dropout and recurrent dropout are added to the GRU layer to prevent overfitting.
  - RMSprop is used as the optimizer, and MAE is the loss function.

- **Bidirectional GRU Layer Model:**
  - A bidirectional recurrent neural network is used with a GRU layer.
  - The bidirectional layer processes the input sequence in both chronological and antichronological order.
  - RMSprop is used as the optimizer, and MAE is the loss function.

- **Conv1D (1D Convolutional) Layers Model:**
  - Conv1D layers are employed for sequence processing.
  - The model includes Conv1D layers, MaxPooling1D, and a GRU layer.
  - RMSprop is used as the optimizer, and MAE is the loss function.

### **Lesson 11:** NLP Transformers, Vision Transformers

#### **Transformers Networks:**
  - Transformers are a type of neural network architecture that has gained popularity in recent years. They are widely used in natural language processing (NLP) tasks such as machine translation, text summarization, and question answering.
  
  - Transformers were introduced by Vaswani et al. in 2017. They are based on the attention mechanism, which allows the model to focus on specific parts of the input sequence when making predictions. Transformers are designed to capture long-range dependencies in sequential data, making them suitable for NLP tasks.
  
  - **Transformers Networks Architecture:**  
    Transformers consist of an encoder and a decoder, each comprising multiple layers. The encoder processes the input sequence, while the decoder generates the output sequence. Transformers are designed to handle variable-length sequences, unlike recurrent neural networks (RNNs), which process sequences one element at a time.
    - Encoder: The encoder consists of a stack of identical layers, each containing two sublayers: a multi-head self-attention layer and a feedforward network. The encoder processes the input sequence and generates a representation for each element in the sequence.
    - Decoder: The decoder also consists of a stack of identical layers, each containing three sublayers: a multi-head self-attention layer, a multi-head attention layer, and a feedforward network. The decoder generates the output sequence by attending to the encoder's output representation and the previous elements in the output sequence.

- **Multi-Head Attention Layer:**
  - The multi-head attention layer is a key component of the transformer architecture. It allows the model to focus on different parts of the input sequence when making predictions. The multi-head attention layer consists of multiple attention heads, each responsible for learning a different representation of the input sequence.
  
  - **Key Characteristics:**
    - **Attention Heads:** The multi-head attention layer consists of multiple attention heads, each responsible for learning a different representation of the input sequence.
    - **Scaled Dot-Product Attention:** The attention mechanism used in the multi-head attention layer is scaled dot-product attention, which computes the dot product of the query and key vectors and scales the result by the square root of the dimensionality of the key vectors.
    - **Query, Key, and Value Vectors:** The multi-head attention layer computes the dot product of the query and key vectors and scales the result by the square root of the dimensionality of the key vectors. The query, key, and value vectors are linear projections of the input sequence.
    - **Masking:** The multi-head attention layer supports masking, which allows the model to focus on specific parts of the input sequence when making predictions. Masking is particularly useful for tasks such as machine translation, where the model should not be allowed to attend to future elements in the input sequence.
    - **Concatenation:** The outputs of the attention heads are concatenated and projected to the expected dimensionality.
  
  - **Parameters:**
    - **Input Shape:** (batch_size, seq_len, features), where "batch_size" represents the number of samples per batch, "seq_len" is the length of the input sequence, and "features" denote the number of input features at each timestep.
    - **Output Shape:** (batch_size, seq_len, features), where "batch_size" represents the number of samples per batch, "seq_len" is the length of the input sequence, and "features" denote the number of input features at each timestep.
    - **Number of Parameters:**  
    $3 \times \text{units} \times (\text{units} + \text{input\_dim} + 1)$,  
    where "input\_dim" is the number of input features.

  - **Usage:**
    - Multi-head attention layers can be employed for sequence modeling and prediction tasks, capturing patterns and dependencies within sequential data.

  - **Positional Encoding:**  
    A layer in allows transformers that allows the model to capture the sequential nature of the input sequence. The positional encoding layer maps relations between different elements e.g. words by checking all the possible combinations between them. It is added to the input sequence before feeding it to the encoder and decoder. The positional encoding layer is a sine and cosine function of different frequencies.
  
  - **Context Vector:**  
    Vector of different word combinations generated after positional encoding for improved understanding of word relations. It is used to generate the attention weights for each word in the input sequence. The context vector is computed by multiplying the attention weights with the value vectors. Those vectors are provided to tge Multi-Head Attention layer as inputs.
  
  - **Feed-Forward Layer:**  
    A layer in the transformer architecture that processes the output of the multi-head attention layer. It consists of dense layers with a ReLU activation function in between. The feed-forward layer is applied to each position separately and identically. It is designed to capture complex patterns and relationships in the data.

##### **Implementation in Keras**
- **Embedding Module:** A module combining traditional embedding and positional encoding to create enriched word representations. The `TokenAndPositionEmbedding` class defines this module, incorporating both token and positional embeddings.
  ```python
  class TokenAndPositionEmbedding(layers.Layer):
      def __init__(self, maxlen, vocab_size, embed_dim):
          super(TokenAndPositionEmbedding, self).__init__()
          self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
          self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
       
      def call(self, x):
          maxlen = tf.shape(x)[-1]
          positions = tf.range(start=0, limit=maxlen, delta=1)
          positions = self.pos_emb(positions)
          x = self.token_emb(x)
          return x + positions

- **Transformer Block:** A module implementing Multi-Head Attention and a feed-forward network. It captures complex relationships and patterns in the data. The `TransformerBlock`` class defines this block.
  ```python
  class TransformerBlock(layers.Layer):
      def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
          super(TransformerBlock, self).__init__()
          self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
          self.ffn = keras.Sequential(
              [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
          )
          self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
          self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
          self.dropout1 = layers.Dropout(rate)
          self.dropout2 = layers.Dropout(rate)
      
      def call(self, inputs, training):
          attn_output = self.att(inputs, inputs)
          attn_output = self.dropout1(attn_output, training=training)
          out1 = self.layernorm1(inputs + attn_output)
          ffn_output = self.ffn(out1)
          ffn_output = self.dropout2(ffn_output, training=training)
          return self.layernorm2(out1 + ffn_output)
  ```
  - **Attention Mechanism Notes:** In the `TransformerBlock` implementation, the attention mechanism calculates attention for each word, and the resulting output is added to the input ($\text{inputs} + attn_{output}$). This attention is then passed through a feed-forward network (FFN). This architecture enhances the model's ability to capture complex patterns and relationships in the data.

#### **Vision Transformers:**
- **ViT (Vision Transformer):**
  A tranformer architecture adapted for computer vision problems such as image classification. It was introduced by Dosovitskiy et al. in 2020. Like in the classical transformers, the networks tries to map the relationship between different parts of data and calculate the attention. But when the texts consisted of individual words that were easily separable, the images have to be converted into smaller patches.
- **Patch:** Discrete regions in an image to which transformers apply positional embeddings and Multi-Head Attention. The notion of patches enables ViTs to process images as sequences of smaller segments.
- **Interpretability:** The ease with which models, particularly transformers, can be understood and visualized. Transformers, with their attention mechanisms, provide interpretability by highlighting important regions in the data.
