# RM_fans(RM GKD relevant)

### Zhai Shichao

#### email: 13754808860@163.com

## Strategy

- Video frame by frame extraction
- Operator selects ROI area by frame
- Calculate the current target position through the contour tree
- Determine the center and radius of the big sign movement
- Determine the rotation direction of the big sign
- Motion prediction
- Send the strike command

## Data

- red_test1.mp4

## Package

- Python 3.7
- opencv
- NumPy 1.19.5
