# Identifying-road-lanes-for-autonomous-driving-systems
## Introduction:
Image segmentation is a key process in image processing and computer vision that
enables detailed analysis by separating distinct regions or objects within an image.
In this project, we explored two distinct approaches for binary segmentation:
Canny Edge Detection, a traditional image processing technique, and the U-Net
architecture, a deep learning model. Both methods aim to achieve precise
segmentation, offering complementary insights into the effectiveness of traditional
and modern approaches.
## Methods used:
Canny Edge Detection
A multi-stage algorithm used to detect edges in an image. The process involves:
➢ Grayscale conversion.
➢ Histogram equalization for contrast enhancement.
➢ Application of Gaussian filtering to remove noise.
➢ Calculation of intensity gradients and non-maximum suppression.
➢ Double thresholding and edge tracking by hysteresis to identify edges.
• Implementation: Processed input images to produce a binary edge map as
the output.
## U-Net Model
A deep learning architecture comprising an encoder-decoder framework with skip
connections to merge low-level and high-level features for precise localization.
➢ Implementation: Used normalized images as input and trained the model to
output pixel-level predictions using ground truth segmentation masks. The
network was trained with Binary Cross-Entropy (BCE) loss and the Adam
optimizer.
## Preprocessing
➢ Resized all images to 80×80
➢ Normalized pixel values to the range [0, 1] for consistency across inputs.
➢ Applied histogram equalization for contrast enhancement before feeding
images to the edge detection algorithm.
## Parameters chosen:
### Canny Edge Detection
➢ Threshold1: 100 (lower bound for gradient intensity).
➢ Threshold2: 200 (upper bound for gradient intensity).
### U-Net Model
➢ Input Dimensions: 80×80×1 (grayscale images).
➢ Batch Size: 16.
➢ Learning Rate: 0.001, with adaptive adjustments using a scheduler.
➢ Loss Function: Binary Cross-Entropy (BCE).
➢ Optimizer: Adam.
## Challenges faced and Solutions applied:
### Model Training and Overfitting
➢ Challenge: The U-Net model overfitted the training data due to the small
dataset size.
➢ Solution: Applied data augmentation techniques (flipping, rotation, scaling)
and incorporated dropout layers to improve generalization.
### Edge Detection Parameters
➢ Challenge: Canny Edge Detection results were sensitive to threshold values,
leading to either excessive or insufficient edge detection.
➢ Solution: Conducted multiple experiments to determine optimal threshold
values (100 and 200) for effective edge detection.
### Project Deployment
➢ Challenge: We faced some issues saving the model into an appropriate
format suitable for fast deployment options like Streamlit.
➢ Solution: We used FastAPI as a suitable method for Front-end testing.
![image](https://github.com/user-attachments/assets/eb69b822-1f0e-400e-a410-c069491f2c1b)
