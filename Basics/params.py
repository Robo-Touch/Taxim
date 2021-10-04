# initial frame processing params
noMarkerMaskThreshold = 30
kscale=50;
diffThreshold = 5;
frameMixingPercentage = 0.15;

# shadow params
max_rad = 100
shadow_step = 1.25
shadow_threshold = -10.0
num_step = 120
discritize_precision = 0.1
height_precision = 0.1
fan_angle = 0.1
fan_precision = 0.05

# deform params
contact_scale = 0.4
pyramid_kernel_size = [201,101,51,21,11,5] #[51,21,11,5]

# general Gaussian blur
kernel_size = 5
sigma = 2

# friction factors
normal_friction = 1
shear_friction = 0.07
