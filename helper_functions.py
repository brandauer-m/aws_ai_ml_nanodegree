import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

def get_data_loaders(data_dir):
	''' Loads and preprocesses images from a given directory using torchvision's transform, 
			ImageFolder and DataLoader'''
	
	print('Preprocess images from image directory...', end='')
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
		
	# Initiate image transformer for test and training images
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
									transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

	test_transforms = transforms.Compose([transforms.Resize(255),
										transforms.CenterCrop(224),
										transforms.ToTensor(),
										transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	# Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

	# Define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
	print('DONE')
		
	return train_data, trainloader, validloader
	
def save_checkpoint(save_dir, model, arch, train_data):
	''' Stores important information about the model to a given path in form of an PyTorch (.pth) file'''
	print(f"Save checkpoint to {save_dir} ...", end='')
	
	# Create checkpoint with most important info about the model and training
	checkpoint = {'model_architecture': arch,
							  'state_dict': model.state_dict(),
							  'classifier_architecture': model.classifier,
							  'class_mapping': train_data.class_to_idx}
	filepath = save_dir + '\checkpoint.pth'
	
	# Save checkpoint
	torch.save(checkpoint, filepath)
	print("DONE")
	
def load_checkpoint(filepath):
	''' Loads a model from a given PyTorch (.pth) file '''
	print(f"Loading checkpoint from {filepath} ...", end='')
	
	# Load base model from checkpoint
	checkpoint = torch.load(filepath)
	if checkpoint['model_architecture'] == 'vgg16':
		model = models.vgg16(weights=None)
	else:
		model = models.efficientnet_v2_m(weights=None)
	
	# Load classifier
	model.classifier = checkpoint['classifier_architecture']
	model.load_state_dict(checkpoint['state_dict'], strict=False)
	model.class_to_idx = checkpoint['class_mapping']
	print("DONE")
	
	return model

def process_image(image):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model
	'''
	print('Preprocess image...', end='')
	with Image.open(image) as img:
		# Get width and height of image
		width, height = img.size
		
		# Resize image to the shortest side being 256px long
		if width < height:
			img.thumbnail((256, 256 * (height / width)))
		else:
			img.thumbnail((256 * (width / height), 256))
		
		# Calculate corners for cropping
		left = (img.width - 224) / 2
		top = (img.height - 224) / 2
		right = (img.width + 224) / 2
		bottom = (img.height + 224) / 2

		# Crop the image
		img = img.crop((left, top, right, bottom))
		
		# Get image color channel values as an numpy array
		np_img = np.array(img) / 255
		
		# Normalize image for later use in network
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
	
		np_img = (np_img - mean) / std
		
		
		# Rearrange dimensions to set color channel as first dimension
		np_img = np_img.transpose((2, 0, 1))
		print('DONE')
		return torch.tensor(np_img, dtype=torch.float32)