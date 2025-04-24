## Classifying items of fashion into categories

## Predicting annual remuneration for STEM professionals

<table>
  <tr>
    <td>
      <figure>
        <img src="images/data_examples.png" width="200">
      </figure>
    </td>
    <td>
      <figure>
        <img src="images/STEM3.png" width="200">
      </figure>
    </td>
    <td>
      <figure>
        <img src="images/STEM2.png" width="200">
      </figure>
    </td>
    <td>
      <figure>
        <img src="images/STEM4.png" width="200">
      </figure>
    </td>
  </tr>
    <tr>
    <th>Examples of fashion images</th>
    <th>Mean salary by company</th>
    <th>Mean salary by year</th>
    <th>Feature importance</th>
  </tr>
</table>

### Project objectives

- Build a deep learning convnet model from scratch with tensorflow to classify images of fashion items into categories using the keras Fashion MNIST dataset. It comprises the following 10 categories:
    1. Ankle boots
    2. Bags
    3. Coats
    4. Dresses
    5. Pullovers
    6. Sandals
    7. Shirts
    8. Sneakers
    9. T-shirts/tops
    10. Trousers
- Use a best practice process of building a model which generalises as best as possible to new data by minimising under- and over-fitting and has an architecture which is neither under- or over-capacity
- The model should have significant power in terms of beating the accuracy performance of the following baseline models:
    - random classifier - 10% (as the data is balanced)
    - basic fully connected dense neural network

  and be comparable to results of published models

- Compare the model's performance with that of a model built using the pretrained VGG16 convnet

### Data

- Training data comprises 8,000 28 x 28 arrays representing grayscale images where each value in an array is a grayscale number, and 8,000 1D arrays of fashion category labels for each image
- Validation data used for hyperparameter tuning and test data used for model evaluation both comprise 2,000 randomly selected images and labels respectively

### Analysis approach

1. Data preparation involved reshaping input tensors to 'image height x image width x number of image channels' and they are homogeneously scaled down to float values between 0 and 1 and category labels are one binary hot-encoded with values set to float format
2. Built a simple fully-connected dense layer neural network as a baseline model comprising only one hidden dense layer of 128 units with no use of validation data for monitoring and no hyperparameter tuning or regularisation undertaken
3. Built a low capacity convnet model with only 32,000 parameters and comprising:
    - one 2D convolutional layer followed by one maxpooling layer to reduce the dimensionality ahead of input into the top dense layers
    - 16 filters used in the convolutional layer and a high filter patch size of 7 x 7
    - 2 fully connected dense layers on top which provide the classifier - returning softmax probabilities for each of the 10 categories - with 16 and 10 units respectively
    - optimizer of 'rmsprop', SGD method of 'categorical crossentropy' for parameter optimizer and evaluation metric of 'accuracy'
    - low number of 20 epochs with a high batch rate
    - validation data used for monitoring purposes
    - no hyperparameters tuning (number of layers, number of filters, patch size, number of neurons per layer, loss rate) or regularization
4. Built a high capacity convnet model with only 534,000 parameters and comprising:
    - three 2D convolutional layers
    - filters ranging from 64 to 128
    - filter patches of size 2 x 2 and 3 x 3
    - two MaxPooling layers at the bottom (to control dimensionality)
    - three dense layers
    - the last dense layer being an output layer with softmax activation function (to supply class probabilities)
5. Based on the finding that the high capacity model started to overfit at epoch 12 

- Once hyperparameters are tuned the model will be trained on all training data (including validation) and evaluated on the unseen test data.

cf. code 'fashion-nmist-classifier.ipynb'

### Results/findings
