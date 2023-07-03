# Setting
SCRN_model = ""

# Dataset Info
classes = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "dining" "table", "pottedplant", "sofa", "tvmonitor"]
data_root = "./PASCAL5i/"
fold = 4
split = 2
sup_bbox_threshold = 5

# Transform
scale_min= 0.9  # minimum random scale
scale_max= 1.1 # maximum random scale
rotate_min= -10  # minimum random rotate
rotate_max= 10  # maximum random rotate
padding_label= 255

# Training setting
epochs = 200
img_size = 473
batch_size = 7 
shot = 1
fix_random_seed_val = True
manual_seed= 605
aux_weight= 1.0
save_path = "./exp/train"

# Model Setting
layers= 50
ppm_scales= [60, 30, 15, 8]
vgg = False
base_lr= 0.0025
momentum= 0.9
weight_decay= 0.0001
power= 0.9
