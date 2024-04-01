This is a project to calculate the best estimation of unknown point coordinates based on the field survey measurements. The project uses Least Square Adjustment on survey networks, and global and local outlier detection for observations to remove the potential error and output the best estimations. This project can handle the Horizontal and angle measurement networks.

# Getting started
The project uses multiple packages. They are all listed in requirements.txt. To install all the packages or check whether they are available, type the following code in the console:

`pip3 install -r requirements.txt`

The project requires 3 files to run: the observation file, the control point file, and the unknown points coordinate approximation file.

The observation file stores the field survey measurements. The format is as follows:

**measurement ID, measurement type, from, at, to, value 1, value 2, value 3**
For more details, please check the demo **Observations.csv**

control point file stores the control point coordinates. The format is as follows:

**Point, X, Y**
For more details, please check the demo **Control.csv**

The unknown Points coordinate approximation file stores the approximation of unknown point coordinates. It has the same format as the control point file.
*This file will no longer be required in the future update*

Make sure the units are unified - use one unit for all observations and one unit for all angles (including errors)

# Quick demo
Please check **Main.py**

# Future Update
1. The unknown point coordinate approximation file will be optional. The project can get a coarse approximation based on the observations and the control point
2. Slope distance will be added to the measurement types
