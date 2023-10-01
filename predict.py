# PROGRAMMER: Marco Brandauer
# DATE CREATED: 2023/10/01                                
# REVISED DATE: 
# PURPOSE: Predicts the class/name of a flower from a given image and model checkpoint. The script reads
#					 in the model parameters and architecture from the checkpoint and afterwards predicts the
# 				 top k classes with their respective probabilities. The user can choose the amount of classes
#					 predicted as well if the inference should run on the GPU and if the classes should be mapped
#					 to a given dictionary including custom names.
#
#   Example call:
#   python predict.py ./flowers/test/01/image_06743.jpg ./checkpoints/checkpoint.pth --top_k=3 --category_names=./cat_to_name.json --gpu
##

# Import necessary modules
import argparse
from model_functions import predict

# Define function to read in command line arguments
def get_input_args():
	parser = argparse.ArgumentParser(
			description='This file trains a flower image classifier on a given dataset with option to choose a neural network model and choosing custom hyperparameters',
	)
	parser.add_argument("image_path", type=str, help="Path to image to be classified")
	parser.add_argument('checkpoint', type=str, help="Path to checkpoint")
	parser.add_argument('--top_k', action="store",
						dest="top_k", type=int, help="Number of returned classes by probability", default=3)
	parser.add_argument('--category_names', action="store",
						dest="category_names", type=str, help="Mapping of categories to real names", default="")
	parser.add_argument('--gpu', action="store_true",
						dest="gpu", default=False, help="Enable inference with GPU")
	 
	return parser.parse_args()

def main():
	args = get_input_args()
	predict(args.image_path, args.checkpoint, args.top_k, args.gpu, args.category_names)

if __name__ == "__main__":
	main()