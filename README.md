This is a project to calculate the best estimation of unknown points coordinates based on the field survey measurments. The project uses Least Square Adjustment on survey netowrks, global and local outlier detection for observations to remove the potential error and output the best estimations. Ths project can handle the Horizontal and angle measurements networks.

# Getting started
The project uses multiple packages. They are all listed in requirements.txt. To install all the packages or checked whether they are available, type the following code in the console:
`pip3 install -r requirements.txt`

The project requires 3 files to run: observation file, control point file, and unknown points coordinate approximation file.

Observation file stores the field survey measurements. The format is as follow:
**measurement ID, measurement type, from, at, to, value 1, value 2, value 3**
For more detail, please check the demo **Observations.csv**

control point file stores the control point coordinates. The format is as follow:
**Point, X, Y**
For more detail, please check the demo **Control.csv**

Unknown Points coordinate approximation file stores the approximation of unknown point coordinates. It has the same format as control point file.
*This file will no longer be required in the future update*

# Quick demo
The following is a quick demo of how to use the project.
