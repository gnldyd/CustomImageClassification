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
    if model is None:
        raise Exception("There is no matching model name.")
    return model


def train(device, model, epochs, train_loader, criterion, optimizer, batch_size, printable=True):
    model.train()
    for epoch in range(1, epochs+1):
        train_loss = 0
        train_accuracy = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)

            output = model(data)    # batch_size 차원의 데이터를 model에 투입하여 훈련
            loss = criterion(output, label)   # 손실 함수의 값 계산
            optimizer.zero_grad()   # pytorch의 변화도를 0으로 설정한다. (기존 변화도에 누적되는 방식이기 때문)
            loss.backward()         # 역전파를 수행하여 가중치의 변화량을 계산
            optimizer.step()        # 가중치의 변화량을 적용하여 가중치 업데이트

            train_loss += loss.item()
            predict = output.max(1)[1]  # (batch_size, classes) 데이터에서 가장 큰 값을 가진 class 노드의 index 추출
            train_accuracy += predict.eq(label).sum().item()   # batch_size 데이터 중 정답과 일치한 개수
        else:
            train_loss /= len(train_loader)  # len(train_loader) = (전체 훈련 데이터 수 / batch_size)
            train_accuracy *= (100 / len(train_loader.dataset))  # len(train_loader.dataset) = 전체 훈련 데이터 수
            if printable:
                print("Train Result Epoch = {}, Loss = {:.4f}, Accuracy = {:.4f}%)".format(epoch, train_loss, train_accuracy))
    else:
        return train_loss, train_accuracy


def test(device, model, test_loader, criterion, printable=True):
    model.eval()            # 평가 모드 적용 - 드롭아웃, 배치정규화 비활성화
    with torch.no_grad():   # 역전파 비활성화 -> 메모리 절약 -> 연산 속도 상승
        test_loss = 0
        test_accuracy = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()
            predict = output.max(1)[1]  # (batch_size, classes) 데이터에서 가장 큰 값을 가진 class 노드의 index 추출
            test_accuracy += predict.eq(target).sum().item()   # batch_size 데이터 중 정답과 일치한 개수
        else:
            test_loss /= len(test_loader)  # len(test_loader) = (전체 시험 데이터 수 / batch_size)
            test_accuracy *= (100 / len(test_loader.dataset)) # len(test_loader.dataset) = 전체 시험 데이터 수
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

    # 필요 폴더 생성
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)

    # gpu 또는 cpu 장치 설정
    device = get_device("cuda:" + str(args.gpu_index))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 데이터 타입을 Tensor로 변형
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 데이터의 Nomalize
    ])

    # DataLoader 생성
    train_loader, test_loader = get_loaders(args.data_path, transform, args.batch_size, args.shuffle)

    model = get_model(args.model_name, args.model_pretrained, args.classes)
    model = model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_name))  # 모델 불러오기
        print("Model loaded at:", args.load_model_name)
    if args.gpu_parallel:
        model = nn.DataParallel(model)  # 데이터 병렬 처리  
    criterion = nn.CrossEntropyLoss().to(device)  # 손실 함수 설정
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # 최적화 설정

    # 훈련
    if args.epochs > 0:
        lastest_train_loss, lastest_train_accuracy = train(device, model, args.epochs, train_loader, criterion, optimizer, args.batch_size, args.printable)
        if not args.printable:
            print("Lastest Train Result: Loss = {:.4f}, Accuracy = {:.4f}%)".format(lastest_train_loss, lastest_train_accuracy))

    # 시험
    test_loss, test_accuracy = test(device, model, test_loader, criterion, args.printable)
    if not args.printable:
        print("Test Result: Loss = {:.4f}, Accuracy = {:.4f}%)".format(test_loss, test_accuracy))

    if args.save_model:
        torch.save(model.state_dict(), args.save_model_name)
        print("Model saved at:", args.save_model_name)

    # 예측
    if args.predict:
        predict(device, args.predict_data_path, model, transform, train_loader.dataset.classes, args.printable)
        print("Image classification completed at", args.predict_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_parallel', type=bool, default=False, help='When there are multiple GPUs, you can train GPUs in parallel by setting this option to True.')
    parser.add_argument('--gpu_index', type=int, default=0, help='You can choose which GPU to use. Index starts from 0. (The gpu_parallel option should be False.)')
    parser.add_argument('--data_path', type=str, default="./data/", help='You can set the location of your image data. (The internal structure should be data/train, data/test, and data/predict.) (It may be helpful to refer to the data folder in the example.)')
    parser.add_argument('--batch_size', type=int, default=8, help='When training the model, you can set the batch size.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Determine whether to shuffle training data and test data.')
    parser.add_argument('--model_name', type=str, default="resnet", help='Set up the model to be trained. resnet152 is set by default.')
    parser.add_argument('--model_pretrained', type=bool, default=True, help='This option allows you to import the parameters of the pre-trained model.')
    parser.add_argument('--classes', type=int, default=10, help='(Important) Set how many categories your data is in.')
    parser.add_argument('--lr', type=float, default=0.001, help='Set the learning rate of the model.')
    parser.add_argument('--epochs', type=int, default=30, help='Set how many times the model will be trained. If it is 0, it is tested immediately without training.')
    parser.add_argument('--printable', type=bool, default=True, help='Print the progress on the screen. True is recommended.')
    parser.add_argument('--load_model', type=bool, default=False, help='This option is to set whether to use the saved model.')
    parser.add_argument('--load_model_name', type=str, default="./data/model.pt", help='Set the location of the saved model. (The value of load_model must be True.)')
    parser.add_argument('--save_model', type=bool, default=False, help='Set the location to save the model. (The value of save_model must be True.)')
    parser.add_argument('--save_model_name', type=str, default="./data/model.pt", help='Location of save model files. (save_model must be True)')
    parser.add_argument('--predict', type=bool, default=False, help='You can classify images using models.')
    parser.add_argument('--predict_data_path', type=str, default=parser.parse_args().data_path + "predict/", help='Set the location of the images you want to classify.')
    run(parser.parse_args())