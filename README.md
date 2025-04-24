## Classifying items of fashion into categories

## Predicting annual remuneration for STEM professionals

<table>
  <tr>
    <td>
      <figure>
        <img src="images/data_exmples.png" width="200">
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
    <th>Mean salary by experience</th>
    <th>Mean salary by company</th>
    <th>Mean salary by year</th>
    <th>Feature importance</th>
  </tr>
</table>

### Project objectives

- Build a deep learning convnet model from scratch which classifies images of fashion items into categories using the keras Fashion MNIST dataset which comprises the following 10 categories:
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
- Use a best practice process of building a model which minimises under- and over-fitting and has an architecture which is neither under- or over-capacity 
- The model should have significant power in terms of beating the accuracy performance of the following baseline models:
    - random classifier - 10% (as the data is balanced)
    - basic fully connected dense neural network

  and be comparable to results of published models

- Compare the model's performance with that of a model built using the pretrained VGG16 convnet

### Analysis approach

- The training data comprises 60,000 28 x 28 arrays representing grayscale images where each value in an array is a grayscale number, and 60,000 1D arrays of fashion category labels for each image. The test data comprises 10,000 randomly selected arrays and labels


cf. code 'fashion-nmist-classifier.ipynb'

### Results/findings
