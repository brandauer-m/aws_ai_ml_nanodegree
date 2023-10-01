# PROGRAMMER: Marco Brandauer
# DATE CREATED: 2023/10/01                                
# REVISED DATE: 
# PURPOSE: Trains a flower image classifier based on a neural network. The user is expected
#					 to input the choice of network architecture, the directory with images and optional
#					 arguments including directory to save the checkpoint, hyperparameters for training
#          and choice to train on GPU. During the training process the current loss and
#					 and accuracy of the model are evaluated and printed out on the console.
#
#   Example call:
#   python train.py ./flowers/ --arch=vgg16 --save-dir=./checkpoints --learning_rate=0.003 --epochs=5 --hidden_units=512 --gpu
##

# Import necessary modules
import argparse
from helper_functions import get_data_loaders, save_checkpoint
from model_functions import create_model, train_model

# Define function to read in command line arguments
def get_input_args():
	parser = argparse.ArgumentParser(
			description='This file trains a flower image classifier on a given dataset with option to choose a neural network model and choosing custom hyperparameters',
	)
	parser.add_argument("data_dir", type=str, help="Path to the image data directory (must have a 'train' and 'valid' subfolder)")
	parser.add_argument('--save-dir', action="store",
						dest='save_dir', type=str, help="Directory for saving checkpoints", required=True)
	parser.add_argument('--arch', action="store",
						dest="arch", type=str, help="Choice of architecture (vgg16, efficientnet_v2_m)", required=True)
	parser.add_argument('--learning_rate', action="store",
						dest="learning_rate", type=float, help="Learning rate for training", default=0.01)
	parser.add_argument('--hidden_units', action="store",
						dest="hidden_units", type=int, help="Number of units in hidden layer", default=512)
	parser.add_argument('--epochs', action="store",
						dest="epochs", type=int, help="Number of training epochs", default=1)
	parser.add_argument('--gpu', action="store_true",
						dest="gpu", default=False, help="Enable inference with GPU")
	 
	return parser.parse_args()

def main():
	args = get_input_args()
	
	# Check if given architecture is not valid
	if args.arch != 'vgg16' and args.arch != 'efficientnet_v2_m':
		print("Invalid 'arch' argument. Can only be 'vgg16' or 'efficientnet_v2_m'.")
		return
	
	model = create_model(args.arch, args.hidden_units, args.gpu)
	train_data, trainloader, validloader = get_data_loaders(args.data_dir)
	train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)
	save_checkpoint(args.save_dir, model, args.arch, train_data)

if __name__ == "__main__":
	main()