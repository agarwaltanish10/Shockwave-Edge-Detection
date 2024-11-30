# **Shockwave Edge Detection**

We first prompt the user in the main() function to upload the image to the notebook, which then goes through the following steps:


1.  **Preprocessing:** Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter. Then Canny edge detection algorithm is used with minVal and maxVal thresholds *experimentally* chosen to be 40 and 80, respectively, which covers all the major edges in the image while ignoring much of the noise/redundant edges. We then find contours among these edges. In the given images, the required lines correspond to the longest contours, hence we sort the array in decreasing order for easy access of the longer contours.

2.   **Angles Computation:** Then, we call the compute_slope() function on the two edges corresponding to the cone and the shockwave, which will give us the cone angle and the wave angle, respectively. Computing a close enough approximation to a part of the required contour, we get a list of points on this contour and then we fit a line through these points. We get the slope of this line and arctan(slope) gives the angle the edge makes with x-axis (thus, we obtain the cone angle and wave angle).

3. **Mach Number Computation:** We have obtained cone angle and wave angle but the calculator requires M1 and one angle to compute all other flow parameters. So, we access the webpage using Selenium, enter the cone angle, perform binary search on values of M1 in the range [1.0, 7.0], and scrape the resultant wave angle value, to determine what value of M1 corresponds to the correct wave angle as computed earlier. This can be computed upto the desired precision by specifying the precision limit and step-size for binary search.

**Output:** (All angles are in degrees)

1.   cone17.jpg:

```
Cone angle: 9.77522080589
Wave angle: 39.9182119396
Mach number: 1.5862244078
```


2.   cone22.jpg:

```
Cone angle: 9.13338345526
Wave angle: 27.3149758487
Mach number: 2.2656202622
```

3. cone30.jpg:

```
Cone angle: 9.15574843012
Wave angle: 21.6659587232
Mach number: 2.9320512574
```