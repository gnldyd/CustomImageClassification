import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import PIL


def get_device(gpu_name):
    device = gpu_name if torch.cuda.is_available() else 'cpu'
    return device


def get_loaders(data_path, transform, batch_size, is_shuffle):
    train_datasets = datasets.ImageFolder(root=data_path + "train", transform=transform)
    test_datasets = datasets.ImageFolder(root=data_path + "test", transform=transform)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=is_shuffle)
    test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=is_shuffle)
    return train_loader, test_loader


def get_model(model_name, is_pretrained, num_classes):
    model = None
    if "alex" in model_name.lower():
        model = models.alexnet(pretrained=is_pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "vgg" in model_name.lower():
        model = models.vgg19(pretrained=is_pretrained)
        if "11" in model_name:
            model = models.vgg11(pretrained=is_pretrained)
            if "bn" in model_name.lower():
                model = models.vgg11_bn(pretrained=is_pretrained)
        elif "13" in model_name:
            model = models.vgg13(pretrained=is_pretrained)
            if "bn" in model_name.lower():
                model = models.vgg13_bn(pretrained=is_pretrained)
        elif "16" in model_name:
            model = models.vgg16(pretrained=is_pretrained)
            if "bn" in model_name.lower():
                model = models.vgg16_bn(pretrained=is_pretrained)
        if "bn" in model_name.lower():
            model = models.vgg19_bn(pretrained=is_pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "res" in model_name.lower():
        model = models.resnet152(pretrained=is_pretrained)
        if "18" in model_name:
            model = models.resnet18(pretrained=is_pretrained)
        elif "34" in model_name:
            model = models.resnet34(pretrained=is_pretrained)
        elif "50" in model_name:
            model = models.resnet50(pretrained=is_pretrained)
        elif "101" in model_name:
            model = models.resnet101(pretrained=is_pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "mnas" in model_name.lower():
        model = models.mnasnet1_0(pretrained=is_pretrained)
        if "0_5" in model_name:
            model = models.mnasnet0_5(pretrained=is_pretrained)
        elif "0_75" in model_name:
            model = models.mnasnet0_75(pretrained=False) # don't have pretrainted model
        elif "1_3" in model_name:
            model = models.mnasnet1_3(pretrained=False) # don't have pretrainted model
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    if model is None:
        raise Exception("There is no matching model name.")
    return model


def train(device, model, epochs, train_loader, criterion, optimizer, printable=True):
    model.train()
    for epoch in range(1, epochs+1):
        train_loss = 0
        train_accuracy = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)

            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predict = output.max(1)[1]
            train_accuracy += predict.eq(label).sum().item()
        else:
            train_loss /= len(train_loader)
            train_accuracy *= (100 / len(train_loader.dataset))
            if printable:
                print("Train Result Epoch = {}, Loss = {:.4f}, Accuracy = {:.4f}%)".format(epoch, train_loss, train_accuracy))
    else:
        return train_loss, train_accuracy


def test(device, model, test_loader, criterion, printable=True):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_accuracy = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()
            predict = output.max(1)[1]
            test_accuracy += predict.eq(target).sum().item()
        else:
            test_loss /= len(test_loader)
            test_accuracy *= (100 / len(test_loader.dataset))
            if printable:
                print("Test Result: Loss = {:.4f}, Accuracy = {:.4f}%)".format(test_loss, test_accuracy))
        return test_loss, test_accuracy


def predict(device, predict_data_path, model, transform, classes, printable):
    predict_datasets = os.listdir(predict_data_path)
    for image in predict_datasets:
        if os.path.isfile(predict_data_path + image):
            data = transform(PIL.Image.open(predict_data_path + image)).to(device).unsqueeze(0)
            result = classes[model(data).max(1)[1]]
            if printable:
                print("Classification: " + predict_data_path + image + " is classified as " + result + ".")
            if not os.path.isdir(predict_data_path + result):
                os.mkdir(predict_data_path + result)
            shutil.copy2(predict_data_path + image, predict_data_path + result + "/" + image)


def run(args):
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    device = get_device("cuda:" + str(args.gpu_index))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, test_loader = get_loaders(args.data_path, transform, args.batch_size, args.shuffle)

    model = get_model(args.model_name, args.model_pretrained, args.classes)
    model = model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_name))
        print("Model loaded at:", args.load_model_name)
    if args.gpu_parallel:
        model = nn.DataParallel(model)  
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.epochs > 0:
        lastest_train_loss, lastest_train_accuracy = train(device, model, args.epochs, train_loader, criterion, optimizer, args.printable)
        if not args.printable:
            print("Lastest Train Result: Loss = {:.4f}, Accuracy = {:.4f}%)".format(lastest_train_loss, lastest_train_accuracy))

    test_loss, test_accuracy = test(device, model, test_loader, criterion, args.printable)
    if not args.printable:
        print("Test Result: Loss = {:.4f}, Accuracy = {:.4f}%)".format(test_loss, test_accuracy))

    if args.save_model:
        torch.save(model.state_dict(), args.save_model_name)
        print("Model saved at:", args.save_model_name)

    if args.predict:
        predict(device, args.predict_data_path, model, transform, train_loader.dataset.classes, args.printable)
        print("Image classification completed at", args.predict_data_path)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_parallel', type=str2bool, default=False, help='When there are multiple GPUs, you can train GPUs in parallel by setting this option to True.')
    parser.add_argument('--gpu_index', type=int, default=0, help='You can choose which GPU to use. Index starts from 0. (The gpu_parallel option should be False.)')
    parser.add_argument('--data_path', type=str, default="./data/", help='You can set the location of your image data. (The internal structure should be data/train, data/test, and data/predict.) (It may be helpful to refer to the data folder in the example.)')
    parser.add_argument('--batch_size', type=int, default=8, help='When training the model, you can set the batch size.')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Determine whether to shuffle training data and test data.')
    parser.add_argument('--model_name', type=str, default="resnet", help='Set up the model to be trained. resnet152 is set by default.')
    parser.add_argument('--model_pretrained', type=str2bool, default=True, help='This option allows you to import the parameters of the pre-trained model.')
    parser.add_argument('--classes', type=int, default=10, help='(Important) Set how many categories your data is in.')
    parser.add_argument('--lr', type=float, default=0.001, help='Set the learning rate of the model.')
    parser.add_argument('--epochs', type=int, default=30, help='Set how many times the model will be trained. If it is 0, it is tested immediately without training.')
    parser.add_argument('--printable', type=str2bool, default=True, help='Print the progress on the screen. True is recommended.')
    parser.add_argument('--load_model', type=str2bool, default=False, help='This option is to set whether to use the saved model.')
    parser.add_argument('--load_model_name', type=str, default="./data/model.pt", help='Set the location of the saved model. (The value of load_model must be True.)')
    parser.add_argument('--save_model', type=str2bool, default=False, help='Set the location to save the model. (The value of save_model must be True.)')
    parser.add_argument('--save_model_name', type=str, default="./data/model.pt", help='Location of save model files. (save_model must be True)')
    parser.add_argument('--predict', type=str2bool, default=False, help='You can classify images using models.')
    parser.add_argument('--predict_data_path', type=str, default=parser.parse_args().data_path + "predict/", help='Set the location of the images you want to classify.')
    run(parser.parse_args())