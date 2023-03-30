import torch.onnx
import torch
from pathlib import Path
from model.MobileNetV3 import mobilenet_v3_large
from model.MobileNetV3 import mobilenet_v3_small
from model.MobileNetv2 import MobileNetv2
from model.LeNet import LeNet
from data.classes import NUMBER_CLASSES, CLASSES_NAME
from config.global_config import global_config
from model.fcnet import Net


def convert_ONNX(model, param_dir, input_size:tuple):
    # param_file = Path(param_dir) / (global_config.MODEL_NAME + '.pt')
    param_file="H:/DigitalRecognition-master/model_params" + "/" + global_config.MODEL_NAME + ".pt"
    model_params = torch.load(param_file)
    model.load_state_dict(model_params)#['state']
    # model.load_state_dict()
    # model=torch.load(param_file)
    print(input_size)
    # print((3, *input_size))
    model.eval()
    # dummy_input = torch.randn(1, (3, *input_size), requires_grad=True)
    # print(dummy_input)
    dummy_input = torch.randn((1,1,20,28), requires_grad=True) # the second is channels
    # print(dummy_input)
    torch.onnx.export(model,  # model_params being run
                      dummy_input,  # model_params input (or a tuple for multiple inputs)
                      str(Path(param_dir) / (global_config.MODEL_NAME + ".onnx")),  # where to save the model_params
                      export_params=True,  # store the trained parameter weights inside the model_params file
                      opset_version=11,  # the ONNX version to export the model_params to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['digitalInput'],  # the model_params's input names
                      output_names=['digitalOutput'],  # the model_params's output names
                      # dynamic_axes={'digitalInput': {0: 'batch_size',1 : 'channels' , 2:'column', 3:'row'},  # variable length axes
                      #               'digitalOutput': {0: 'batch_size',1 : 'channels' , 2:'column', 3:'row'}}
                      )
    print(" ")
    print(f'Model {global_config.MODEL_NAME} has been converted to ONNX')
    return

if __name__ == '__main__':
    root_path = Path.cwd().parent
    # model = mobilenet_v3_small(num_classes=5)
    model=Net()
    # model = MobileNetv2(wid_mul=1, output_channels=NUMBER_CLASSES)

    # model = LeNet(global_config.CLASSES_NUM)
    # param_dir = root_path / 'model_params'
    param_dir=r"C:\Users\75464\Desktop\Digit_Net"
    convert_ONNX(model, str(param_dir), global_config.INPUT_SIZE)