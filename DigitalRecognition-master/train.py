import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import copy
from model.MobileNetv2 import MobileNetv2, ont_hot_cross_entropy
from model.MobileNetV3 import mobilenet_v3_large
from model.MobileNetV3 import mobilenet_v3_small
from model.LeNet import LeNet
from config.global_config import global_config
from data.dataset import NumberDataset
from data.classes import NUMBER_CLASSES
from model.fullConnection import fullConnection
from model.fcnet import Net

@torch.no_grad()
def test(model, test_dataloader, model_save_path, writer, ep):
    print(f"Testing model\n")
    test_loop = tqdm(test_dataloader)
    correct_num = 0
    print(model_save_path)
    # model.load_state_dict(torch.load(model_save_path, map_location=torch.device(global_config.DEVICE)))
    model.eval()
    for img, label in test_loop:
        output = model(img)
        # print("-------------------")
        # print("output")
        # print(output)
        # # # print(output)
        # # print(torch.sum(torch.max(output[0], dim=1)[-1]))
        # print(output[0]) #row
        # print(torch.max(output[0], dim=1))
        # print(torch.max(output[0], dim=1)[-1])
        # print("label")
        # print(label)
        # print(torch.sum(torch.max(label, dim=1)[-1]))
        # # print(torch.max(label, dim=1))
        # # print(torch.max(label, dim=1)[-1])
        # print("-------------------")
        #output=output.squeeze()

        correct_num += torch.sum(torch.max(output, dim=1)[-1] == torch.max(label, dim=1)[-1])
    # print("-------------------")
    # print(correct_num)
    # print(len(test_dataloader.dataset))
    # print("-------------------")
    precision = correct_num / len(test_dataloader.dataset)
    # #len (test_dataloader.dataset)=640
    print(f"Test precision is {precision}\n")
    writer.add_scalar("Test precision", precision, ep)

g_precision=0

g_precision_list=[]

@torch.enable_grad()
def train(model, dataloader, test_dataloader, epoch, writer,g_precision,precision_list):
    # test_model = copy.deepcopy(model)
    # test_model.to(torch.device(global_config.DEVICE))
    model_save_path = Path.cwd() / "model_params"
    if not model_save_path.is_dir():
        Path.mkdir(model_save_path)
    model_save_path = model_save_path / (global_config.MODEL_NAME + '.pt')
    optim = SGD(model.parameters(), 0.02) #lr
    # optim = Adam(model.parameters(),0.02)
    epoch_loop = tqdm(range(epoch), total=epoch)
    train_count = 1  # 用于计算runtime_loss
    print(model_save_path)
    for ep in epoch_loop:
        print("---------------------------- eopch : {} ----------------------------".format(ep))
        runtime_loss = 0
        data_loop = tqdm(dataloader)
        correct_num = 0
        for img, label in data_loop:
            optim.zero_grad()
            output = model(img)
            output = output.squeeze()
            # print("output")
            # print(output)
            # print("label")
            # print(label)
            # print(output)
            # output=output.reshape(1,output.shape[0])  #if batch size is 1
            # print(output)
            loss = ont_hot_cross_entropy(output, label)
            loss.backward()
            optim.step()
            runtime_loss += loss
            if train_count % 3 == 0:
                print("runtime_loss= {:s}\n".format(str(runtime_loss / 3)))
                writer.add_scalar('Loss', runtime_loss, train_count)
                runtime_loss = 0
            correct_num += torch.sum(torch.max(output, dim=1)[-1] == torch.max(label, dim=1)[-1])
            # print("torch.max(output,dim=1)[-1]")
            # print(torch.max(output, dim=0)[-1])
            # print(output)
            # print("torch.max(label, dim=0)[-1]")
            # print(torch.max(label, dim=0)[-1])
            # print(label)
            train_count += 1
        precision = correct_num / len(dataloader.dataset)
        precision_list.append(precision)
        print(f"precision is: {precision}\n")
        print(f"the last best precision is : {g_precision}\n")
        # writer.add_scalar('Precision', precision, ep)
        if precision>g_precision:
            torch.save(model.state_dict(), str(model_save_path))
            print("good precision!")
            g_precision=precision
            # print(f"Successfully save model_params state.\n")
            test_model = copy.deepcopy(model)
            test_model.to(torch.device(global_config.DEVICE))
            test(test_model, test_dataloader, model_save_path, writer, ep)

if __name__ == '__main__':
    root_path = global_config.DATASET_PATH
    # v2model = MobileNetv2(wid_mul=1, output_channels=NUMBER_CLASSES).to(global_config.DEVICE)
    # # model = LeNet(classes_num=NUMBER_CLASSES).to(global_config.DEVICE)
    # v3model=mobilenet_v3_small(num_classes=5).to(global_config.DEVICE)
    # pretrained_model=torch.load('weights/mobilenet_v3_large_pretrained.pth',map_location=global_config.DEVICE)
    # pre_dict={k: v for k, v in pretrained_model.items() if v3model.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = v3model.load_state_dict(pre_dict, strict=False)

    fcmodel=Net().to(global_config.DEVICE)

    dataset = NumberDataset(root_path, input_size=global_config.INPUT_SIZE, classes_num=NUMBER_CLASSES) #phase="train"
    test_dataset = NumberDataset(root_path, classes_num=global_config.CLASSES_NUM, input_size=global_config.INPUT_SIZE, phase="test")
    training_config = {
        "batch_size": 8,
        "epoch": 2
    }
    dataloader = DataLoader(dataset, training_config['batch_size'], True)
    test_dataloader = DataLoader(test_dataset, 1, True)
    writer = SummaryWriter()
    train(fcmodel, dataloader, test_dataloader, training_config['epoch'], writer, g_precision, g_precision_list)
    print(g_precision_list)
    print("end training!")