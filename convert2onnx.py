import torch
from torch import nn
import os
import json

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from scene import Scene

class ClassifierWithSoftmax(nn.Module):
    def __init__(self, classifier):
        super(ClassifierWithSoftmax, self).__init__()
        self.classifier = classifier
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        x = self.classifier(x)
        x = self.softmax(x)
        return x

def convert2onnx(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt : OptimizationParams, select_obj_id : int, removal_thresh : float):


    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)

    print(gaussians._objects_dc.shape)

    torch_input = gaussians._objects_dc.permute(2,0,1)

    # Assuming 'gaussians' and 'num_classes' are defined
    # and 'dataset.model_path' and 'scene.loaded_iter' are available
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "classifier.pth")))
    classifier.eval()

    combined_model = ClassifierWithSoftmax(classifier)
    combined_model.cuda()
    combined_model.eval()

    point_cloud_path = os.path.join(dataset.model_path, "point_cloud_object_removal/iteration_{}".format(scene.loaded_iter))

    model_name = os.path.join(point_cloud_path, "classifier.onnx")

    # Export the model to ONNX format
    # torch.onnx.export(combined_model, torch_input, model_name)

    torch.onnx.export(
        combined_model,
        torch_input,
        model_name,
        input_names=['input'],
        dynamic_axes={'input': [1]}
    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_removal/figuritas.json", help="Path to the configuration file")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    convert2onnx(model.extract(args), args.iteration, pipeline.extract(args),  opt.extract(args), args.select_obj_id, args.removal_thresh)