import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
import json
from helper_functions import load_checkpoint, process_image

def create_model(arch, hidden_units, gpu):
	''' Create a model from given architecture and hidden units '''
	
	print('Creating model...', end='')
	# Detect which model it is and instantiate accordingly
	if(arch == 'vgg16'):
		model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
		input_units = 25088
	else:
		model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
		input_units = 1280
	
	# Freeze model parameters	
	for param in model.parameters():
		param.requires_grad = False
	
	# Create classifier with 1 hidden layer
	classifier = nn.Sequential(OrderedDict([
								('fc1', nn.Linear(input_units, hidden_units)),
								('relu', nn.ReLU()),
								('dropout', nn.Dropout(0.2)),
								('fc2', nn.Linear(hidden_units, 102)),
								('relu', nn.ReLU()),
								('output', nn.LogSoftmax(dim=1))
								]))

	model.classifier = classifier
	print('DONE')
	
	# Move model to cuda if possible
	device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
	print(f'Moving model to {device}...', end='')
	model.to(device)
	print('DONE')
	
	return model
	
def train_model(model, trainloader, validloader, epochs, learning_rate, gpu):
	''' Train a given model with user specific hyperparameters including epochs and learning rate '''
	
	# Define criterion and optimizer
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
	
	# Initiate variables for monitoring
	steps = 0
	running_loss = 0
	print_every = 1
	
	# Detect cuda availability
	device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
	
	print('Start training...')
	for epoch in range(epochs):
		for inputs, labels in trainloader:
			steps += 1
			
			# Move input and label tensors to the default device
			inputs, labels = inputs.to(device), labels.to(device)
			
			# Reset optimizer
			optimizer.zero_grad()
			
			# Feed forward image and calculate loss
			logps = model.forward(inputs)
			loss = criterion(logps, labels)
			
			# Update weights with back propagation
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			
			# Calculate current loss and accuracy with validation set
			if steps % print_every == 0:
				test_loss = 0
				accuracy = 0
				
				# Turn on evaluation mode
				model.eval()
				
				with torch.no_grad():
					for inputs, labels in validloader:
						if torch.cuda.is_available() and gpu:
							inputs, labels = inputs.to('cuda'), labels.to('cuda')
					
						logps = model.forward(inputs)
						batch_loss = criterion(logps, labels)
						
						test_loss += batch_loss.item()
						
						# Calculate accuracy
						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
								
				print(f"Epoch {epoch+1}/{epochs}.. "
						f"Train loss: {running_loss/print_every:.3f}.. "
						f"Test loss: {test_loss/len(validloader):.3f}.. "
						f"Test accuracy: {accuracy/len(validloader):.3f}")
				running_loss = 0
	
	print('Finished training')
	
def predict(image_path, model_checkpoint, topk, gpu, category_names):
	''' Predict the class (or classes) of an image using a trained deep learning model.
	'''
	print('Start prediction...')
	# Load model from checkpoint
	model = load_checkpoint(model_checkpoint)
	
	# Process image for further use as input
	img = process_image(image_path)
	img = img.unsqueeze(0)
	
	# Initiate category to name dict if given by user
	if category_names != "":
		with open('cat_to_name.json', 'r') as f:
			cat_to_name = json.load(f)
	
	# Move img and model to GPU, if available
	device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
	model.to(device)
	img = img.to(device)
	
	# Set model to evaluation mode
	model.eval()
	
	# Turn off gradients
	with torch.no_grad():
		
		# Calculate prediction
		prediction = model.forward(img)
		prediction = torch.exp(prediction)
		
		# Get top k predicted indexes and their probability
		top_p, top_class  = prediction.topk(topk, dim=1)
		
		# Reform classes to show flower name if category names were given
		if category_names != "":
			idx_to_class = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
		else:
			idx_to_class = {v: k for k, v in model.class_to_idx.items()}
		top_class = [idx_to_class[index] for index in top_class.cpu().numpy()[0]]
		
		# Print top k classes with their respective probabilities
		print(f"Top {topk} class(es):")
		for prob, t_class in zip(top_p[0].cpu().numpy(), top_class):
			print("\t-{format_class} ({format_prob:.2f}%)".format(format_class=t_class, format_prob = prob * 100))