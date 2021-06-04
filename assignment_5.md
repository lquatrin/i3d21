# 3D Graphics Systems Course - IMPA 2021

#### Leonardo Quatrin Campagnolo

---------

## Assignment 5 - ShapeNet Data and Plotly visualization TODO

In this assignment, the idea was to manipulate and visualize some mdoels of the ShapeNet dataset, also trying to explore other different ways to visualize 3D models with plotly and PyTorch3D. We also made a further exploration on the generation of models using parametric functions with surfaces of revolution.

The paper ShapeNet: An Information-Rich 3D Model Repository [1] shows how they produce a dataset with CAD models. In fact, most of the advances with machine learning were only possible thanks to the availability of huge amounts of data, enabling networks to learn distributions with high data diversification. In addition, having standardized information in databases is important for the whole community to have access to the same data, also facilitating comparisons and evaluations in general. Finally, the fact that we have databases increasingly richer in different aspects also makes it possible to generate solutions for different problems, enabling advances and applications for different areas, and also raising new challenges and ideas to be solved.

One of the challenges on producing a dataset is how to develop a structure that can organize data in a coherent way, especially if the database will be updated and expanded from time to time. It is also important to provide a diversified amount of data. In this case, the authors themselves made reservations about the data that were categorized, which have a bias due to the fact that they used CAD models. In this case, the database has a smaller number of natural objects due to the used format.

It is possible to note several challenges encountered when determining a vast database such as ShapeNet. According to the authors, the main challenge for creating a database like ShapeNet is to be able to define a good methodology to acquire and validate the notes written for each object, since it is expensive to use only manual intervetion for each model. One of the ways used by the authors was to apply algorithms to generate initial predictions, and then verify these predictions through crowd-sourcing pipelines and inspection by human experts, what they called as a hybrid strategy. It is a fact that using only manual intervention is more costly than using algorithms to make predictions, and both algorithms and humans can also be subject to errors. For that, the authors also added the annotation source, being a way of considering how reliable the information is, so each person can use it in the way he see most convenient.

### Data visualization with Plotly

In this assignment, it was used a subset of ShapeNet containing 329 models.

<img src="imgs/a5/3_1.png" width="30%">

<img src="imgs/a5/3_2.png" width="30%">

<img src="imgs/a5/3_3.png" width="30%">

<img src="imgs/a5/3_4.png" width="30%">

<img src="imgs/a5/e_2.png" width="30%">


### Creating a mesh using a parametric model


### References

[1] ShapeNet: An Information-Rich 3D Model Repository
