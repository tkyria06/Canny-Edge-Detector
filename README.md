## About

This is an implementation of Canny Edge Detector Algorithm with Python

## Prerequisites 

Install OpenCV:
```
npm install --global gulp
```

Install Matplotlib:
```
pip install matplotlib
```

Install NumPy:
```
pip install numpy
```

## Author

Theodoros Kyriakou

## Results

Original Image             |  Edge Detection Results
:-------------------------:|:-------------------------:
<img src="Results/building.jpg" width="450" height="340">  |  <img src="Results/Edge_Detection_Results.png" width="450" height="340">
Magnitude | 1st order partial derivatives
<img src="Results/Image_gradient_magnitude.png" width="450" height="340"> | <img src="Results/Image_1st_order_partial_derivatives.png" width="450" height="340">
Edge Thinning | Gaussian Blurring
<img src="Results/Non-maximum_Suppression.png" width="450" height="340"> | <img src="Results/Gaussian_Blurring.png" width="450" height="340">
