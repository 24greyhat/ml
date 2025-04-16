# fully connected layer

- we pass the `nn.Linear` function (input, output)

    - input size which is 28*28 when flattened aka reshaped into a 1d array

    - output size can be whatever we want and must be the input size of the following layer if any.
 
    - output size is essentially the amount of neurons the layer has.



- The `nn.Linear` method creates a Linear layer aka a fully connected layer

      - a linear layer is better 


- The `nn.Conv` method creates a Convolutional layer



# Linear Layer

- Use Cases:

    - **Tabular Data:** When dealing with structured data (like spreadsheets) where each feature is independent, linear layers are appropriate.
    
    - **Final Layers:** Often used as the last layer in a network to produce outputs after feature extraction has been performed by other layers (e.g., after convolutional layers).
    
    - **Low-Dimensional Data:** Suitable for data with fewer dimensions where spatial relationships are not significant.

    
    - **Characteristics:** Each neuron in a linear layer is connected to every neuron in the previous layer, making it suitable for capturing complex relationships but potentially leading to a large number of parameters.
    
    It does not take into account the spatial structure of the data, which can be a limitation for certain types of input.



# Convolutional Layer

- Use Cases:
    - **Image Data:** Ideal for processing grid-like data such as images, where spatial hierarchies and local patterns (like edges and textures) are important.
    
    - **Time Series Data:** Can also be used for sequential data (like audio or time series) by treating it as a 1D signal.
    
    - **Feature Extraction:** Commonly used in the early layers of deep learning models to automatically learn spatial hierarchies of features from the input data.
    
    - **Characteristics:** Convolutional layers apply filters (kernels) that scan across the input data, allowing them to learn spatial hierarchies and local patterns efficiently.
    
    - They have fewer parameters compared to fully connected layers because the same filter is applied across different parts of the input, making them less prone to overfitting and more efficient for high-dimensional data.



- Takeaway: Choose Linear Layers for tasks involving tabular data or as final output layers in a network.
            Choose Convolutional Layers for image data, spatially structured data, or when you need to capture local patterns and features.
            
> **Note:** In practice, many neural network architectures will combine both types of layers, using convolutional layers to extract features from data and linear layers to perform final classification or regression tasks.





# Feed Forward Neural Network

- A simple neural network is called a feed forward neural network meaning data passes in one direction, from the first input layer to the last output layer.




# The view method

- view() reshapes the tensor without copying memory, similar to numpy's reshape().

- for example .view(2,2) will return a 2x2 tensor.

- The -1 parameter

    - if there is any situation that you don't know how many rows you want but are sure of the number of columns, then you can specify this with a -1.
      
    - (Note that you can extend this to tensors with more dimensions. Only one of the axis value can be -1 so either x or y)
 




# Zeroing the gradient

- The `nn.zero_grad()` method will zero the gradient of the network.

- gradients basically contain the loss and the optimizer depends on these gradients to optimize the weights.

- in PyTorch, for every batch during the training phase, we typically want to explicitly set the gradients to zero before starting to do backpropagation (i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes.

- This accumulating behavior is convenient while training RNNs or when we want to compute the gradient of the loss summed over multiple batches. So, the default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call.

- Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly. Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters and the newly-computed gradient.

- It would therefore point in some other direction than the intended direction towards the minimum (or maximum, in case of maximization objectives).




# Optimization curve

- The learning rate in part dictates the size of the step that the optimizer will take to get to the best place.
  
- Everytime we foward some input data into our neural network we get a loss and it is entirely calculable to determine what weights we need to get a loss of zero from our current loss but we can't do that because the model will overfit everything we pass thru.
  
- We use the learning rate to tell the optimizer to optimize to lower the loss butu only take certain size steps then overtime as we train the model batch after batch the optmizer will adjust the weights in such manner that what remains is the general principle or pattern we want to capture, the size of the steps the optimizer will take is dependent upon the learning rate.

- a higher learning rate means larger optimizer steps which means the model will never really learn the general pattern we want to capture and instead will only best fit the last batch it was passed and all the other batches it was passed are rendered useless.

- There is no way of identifying the perfect size step to get to the optimal point for a given model, the solution to that is a **Decaying Learning Rate** which means we start of with gigantic steps but overtime the learning rate gradually decends until we reach the optimal point, this is essential for modeling complex data but not so much extremely basic problems.



# The argmax and argmin methods


- torch.argamx - will return the index of the tensor with the maximum value in the given tensor of tensors
  
- torch.argmin - will return the minimum value in the given tensor of tensors
