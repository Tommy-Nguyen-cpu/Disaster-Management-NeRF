# Data Science Foundations - NeRF-- In Disaster Management
## Introduction
The growing occurrence of natural disasters in Africa (Libya flooding and Morocco earthquake) underscores the critical role of disaster management in addressing and mitigating the aftermath of these calamities. Specifically, efficient response teams and disaster management analysts are vital in ensuring the safe rescue and support of those affected by natural disasters. Their roles are instrumental in delivering necessary aid and assistance to those in need. An essential element in disaster management planning involves assessing the landscape and determining the most efficient routes to rescue and safeguard lives. However, this task becomes challenging during disasters, as videos and images often lack the interactivity necessary for response teams to locate and assist individuals in distress. In turn, 3D models serve as a viable alternative to videos and images, offering response teams the ability to identify optimal routes for rescue operations. Moreover, disaster management analysts can leverage these 3D models to conduct real-time and post-disaster analyses. This includes assessing the extent of damage, identifying the causes of the disaster, exploring preventive measures, and devising more efficient strategies to aid a larger population in need. As of the current writing, several algorithms endeavor to generate 3D models from image inputs. Yet, these methods often demand costly tools, additional inputs that might be inaccessible during disasters, or extensive memory usage for running the application. In contrast, NeRF Minus Minus (NeRF--) presents an attractive alternative. NeRF-- removes the necessity for expensive tools, relies solely on images as input, and optimizes memory usage. Consequently, NeRF-- emerges as an optimal solution for creating realistic forward-facing 3D models pertinent to disaster management.
## Selection of Data
NeRF-- differs from many other algorithms in that they are trained each time a new scene is rendered. In other words, the process of training a NeRF-- model and producing a 3D representation of a scene at test time are combined into one process/step. As such, I do not need a massive dataset. Instead, I simply need images of a scene to train NeRF-- on. For the specific scene, there are 2 criterias that needed to be met in order to produce sufficient results using NeRF-- model: the images must be captured in a manner outlined by the LLFF paper and the scene must be forward-facing. <br />
![LLFF instructions for capturing images.](https://github.com/Tommy-Nguyen-cpu/Disaster-Management-NeRF/blob/main/Images/LLFFDinosaur.png) <br />
The red squares represent the area in which we have to capture the image from (the images must be formatted properly and be consistent throughout). <br />
Given these requirements, here is the first row of images (first 3 images taken): <br />
![First image captured]([https://github.com/Tommy-Nguyen-cpu/DisasterManagementNeRF/blob/main/20231121_123031.jpg?raw=true](https://github.com/Tommy-Nguyen-cpu/Disaster-Management-NeRF/blob/main/custom_upload/20231121_123031.jpg)) <br />
![Second image captured]([https://github.com/Tommy-Nguyen-cpu/DisasterManagementNeRF/blob/main/20231121_123033.jpg?raw=true](https://github.com/Tommy-Nguyen-cpu/Disaster-Management-NeRF/blob/main/custom_upload/20231121_123033.jpg)) <br />
![Third image captured]([https://github.com/Tommy-Nguyen-cpu/DisasterManagementNeRF/blob/main/20231121_123035.jpg?raw=true](https://github.com/Tommy-Nguyen-cpu/Disaster-Management-NeRF/blob/main/custom_upload/20231121_123035.jpg)) <br />
## Methods
### Tools Used
- NumPy, Matplotlib, and PyTorch
- VS Code
<!-- --> 
<br />
The model used was a fully connected neural network or a MLP. The MLP contains 8 fully connected layers using ReLU as the activation function, 2 fully connected layers for the density and feature (no activation function), and 2 RGB layers. Originally, NeRF-- had 4 fully connected layer using ReLU. The reason for this was to save computational cost, but my goal is to see the full extent/power of NeRF--, as such I increased the layers to 8. There are, of course, other modifications I can make (such as changing the dimensions, etc), but these may make NeRF-- be computationally infeasible to run on my computer. <br />
The hyperparameters are left as the default parameters in the NeRF-- model and the metric used to optimize NeRF-- is PSNR (Peak Signal-to-Noise Ratio).

## Results
Given the 6 images, here is one of the images rendered by NeRF--: <br />
![Final Rendered Result]([https://github.com/Tommy-Nguyen-cpu/DisasterManagementNeRF/blob/main/Quad2ndFLResult.png?raw=true](https://github.com/Tommy-Nguyen-cpu/Disaster-Management-NeRF/blob/main/Images/Quad2ndFLResult.png)) <br />
The model ran for 10,000 epochs, which should have been a sufficient amount of time to learn the scene, but the final rendered image (and consequently the final rendered model) was incapable of rendering a high-detail and low-noise model. There could be a number of reasons for this, which I will speak about in the "Discussion" section. <br />
The chart below shows the PSNR score of the model after 10,000 iterations: <br />
![First image captured]([https://github.com/Tommy-Nguyen-cpu/DisasterManagementNeRF/blob/main/PSNR2ndFL.png?raw=true](https://github.com/Tommy-Nguyen-cpu/Disaster-Management-NeRF/blob/main/Images/PSNR2ndFL.png)) <br />
A good PSNR score for images typically fall within the 30-40 range. However in our instance, our model plateaued at approximately 22. Despite the low PSNR score, the rendered image produced by NeRF-- seemed to display general colors and shapes of objects within the scene. This implies that the model did learn the shape and color well, but may have oversmoothed some details in the scene. Once more, there could be a number of reasons for this result which we will discuss in the "Discussion" section.

## Discussion


## NeRF-- Setup
In order to get NeRF-- to work on your computer, you must create a virtual environment that has the required libraries. <br />
"python -m venv {Some-Environment-Name}"<br />
Once an environment has been created, download "torch" and "torchvision" following the steps outlined for your specific system on the following site: https://pytorch.org/get-started/locally/ <br />
Once "torch" and "torchvision" has been installed, pip install the libraries listed in "requirements.txt": <br />
"pip install -r requirements.txt" <br />
Now that the corresponding libraries have been created, you should now be able to run the jupyter notebook!
