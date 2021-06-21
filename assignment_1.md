# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 1 - Python / Collab and MNIST

In this first assignment, the objective was to get more familiar with pytorch and google colab environment. So, i spent some time reading the documentation to be able to understand how to handle tensors with Pytorch and how to create a simple neural network to perform digit recognition. The code of this assignment can be found [here](https://github.com/lquatrin/i3d21/blob/main/code/a1/Assignment_1.ipynb).

### PyTorch Tensors

The first part of the work consisted of creating and manipulating tensors using the PyTorch. I was able to observe some differences using the different ways to create Tensors:

Both **torch.Tensor** and **torch.empty** use the default type that can be checked by calling the function **torch.get_default_dtype()**. We can also change the current default type by using:


```python
torch.set_default_dtype(torch.float)
torch.get_default_dtype()
> torch.float32
``` 

However, the type inferred by **torch.empty** can also be modified through a parameter. In the case of **torch.tensor**, the type ends up being inferred according to the input data.

Both the **torch.tensor** and **torch.Tensor** functions end up creating a copy of the data. Unlike the other methods, **torch.as_tensor** does not store a copy of the value that was passed. In this case, if a value is modified both from the tensor and from the input data, the change will be made in the same memory space.

### Tensor operations

When we need to compute large batch operations on tensor, it is better to use the available methods from PyTorch. If we perform a multiplication operation on each row of a matrix, the simplest way to do this would be to implementing a for, multiplying each row by a number. However, in cases where we have larger tensors, this turns out to be very costly.
Thus, a test was performed using the **torch.mul** function and * operator. When comparing each implementation, we see how the performance improved, using a tensor of size 1000x600:

```python
# 6.054009437561035 s
def multLines(ts):
  for l in range(ts.shape[0]):
    for r in range(ts.shape[1]):
      ts[l,r] = ts[l,r] * (l+1) 
  return ts

# 0.0017743110656738281 s
def multLinesTensor(ts):
  ta = torch.arange(1, ts.size()[0]+1).view(-1, 1)
  return torch.mul(ta, ts)

# 0.0011224746704101562 s
def multLinesTensor2(ts):
  ta = torch.arange(1, ts.size()[0]+1).view(-1, 1)
  return ta * ts
``` 

### Neural network for digit recognition

The last part of this assignment consisted on making some tests on a simple 2-layer neural network for digit recognition using the MNIST dataset. Each image is 28x28 pixels with 1 color channel, defining grayscale images. We can use the **torchvision.utils.make_grid** function to see each batch in a more compact view.

```python
def show_batch(batch):
    im = torchvision.utils.make_grid(batch, nrow=8)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.axis('off')
show_batch(images)
``` 

<p align="center">
  <img src="data/imgs/a1/bach_of_images.png" width="70%">
  <br>
  <em>Fig. 1: Batch of 32 digits.</em>
</p>

Through the **torch.bincount** function, it is also possible to check how balanced a batch is:

```python
print("Batch bincount:", torch.bincount(labels))
print("Train dataset bincount:", mnist_train_data.targets.bincount())
print("Test dataset bincount:", mnist_test_data.targets.bincount())
> Batch bincount: tensor([3, 5, 3, 2, 2, 2, 4, 5, 1, 5])
> Train dataset bincount: tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
> Test dataset bincount: tensor([980, 1135, 1032, 1010,  982,  892,  958, 1028,  974, 1009])
``` 

Now, to perform the digit recognition using the MNIST dataset, a network was created using a nn.Sequential class, with an intermediate layer (usually defined by 128 nodes), followed by a ReLU activation function. In order to pass each set of images to the network, it was necessary to transform the image into a vector. In addition, the stochastic gradient gradient method was chosen:

```python
mnist_model = nn.Sequential(
    nn.Linear(dim_in, dim_hidden),
    nn.ReLU(),
    nn.Linear(dim_hidden, dim_out)
).to(device_gpu)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mnist_model.parameters(), lr = 0.01)
``` 

The optimization loop was defined by:

```python
for epoch in loop:
  loss_train = 0
  for batch_images, batch_labels in data_loader_train:
    batch_images = batch_images.to(device_gpu)
    batch_labels = batch_labels.to(device_gpu)

    batch_size = batch_images.shape[0]
    outputs = mnist_model(batch_images.view(batch_size, -1)) 
    loss = loss_function(outputs, batch_labels)
    loss_train += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

A validation loop is also added to compute the accuracy for each epoch using the validation data.

When using a low learning rate, it was necessary more iterations to converge on a reasonable accuracy, which was also noticed when increasing the number of nodes in the intermediate layey. However, using a high learning rate can make the model never converge.

The batch size also affects the performance of the training. For smaller batches, the loss function decreased faster, obtaining greater accuracy: 97.71 for 8 images per batch, 97.64 for 16, and 96.95 to 32. However, it took a longer time per iteration: 11.68, 8.99 and 7.16 respectively.

When passing the model to GPU, i realized that each iteration of the training ended up taking more time compared to CPU (7.16 to 9.68 seconds), using an intermediate layer of 128 nodes. Then, i did a test extrapolating the number of nodes in the intermediate layer to 10000. In this case, trainin the model in CPU took about 50 s per iteration, while in GPU remained at 9 s. I could note that a simple network does not end up having such a performance impact, as it cannot exploit GPU parallelization effectively. In addition, passing the model to the GPU can generate additional time due to memory allocation and transfer issues.

Finally, i generate the results using a intermediate layer using 128 nodes, trained with batches of 32 images, and using a learning rate of 0.01. In the graphs below, we can see that the accuracy using the validation data increasing according to each interation, and the loss function decreases. Then, an accuracy of 97.07% was achieved when using the data from test set. Below are the graphs with the values of precision and loss function throughout the training. The loss validation was not evaluated in this report.

![Accuracy per Iteration](data/imgs/a1/accuracy.png)

![Loss Function](data/imgs/a1/loss_function.png)

I also plotted the confusion matrix to check the performance of the model for each category. As we can see, the majority of cases were correctly classified using the simple neural network implemented in this assignment.

<p align="center">
  <img src="data/imgs/a1/confusion_matrix.png" width="70%">
  <br>
  <em>Figure 1 training process of NN.</em>
</p>
