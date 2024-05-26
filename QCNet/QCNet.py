import csv
import cv2
import numpy as np
from sort.sort import Sort
import time
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

#RPN网络
class RPN(nn.Module):
    def __init__(self, in_channels):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.cls_score = nn.Conv2d(256, 18, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(256, 36, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

#FPN网络
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            
    def forward(self, x):
        laterals = [lateral_conv(x[i]) for i, lateral_conv in enumerate(self.lateral_convs)]     
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(laterals[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return tuple(outs)

#主干网络（Queue Counting Network)
class QCNet(nn.Module):
    def __init__(self):
        super(QCNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fpn = FPN(in_channels_list=[64, 128, 256, 512], out_channels=256)
        self.rpn = RPN(256)
        
        self.fc1 = nn.Linear(512 * 14 * 14, 4096)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        c1 = self.relu1(x)
        x = self.pool1(c1)
        x = self.conv2(x)
        c2 = self.relu2(x)
        x = self.pool2(c2)
        x = self.conv3(x)
        c3 = self.relu3(x)
        x = self.pool3(c3)
        x = self.conv4(x)
        c4 = self.relu4(x)
        x = self.pool4(c4)
        
        fpn_outs = self.fpn([c1, c2, c3, c4])
        rpn_out = [self.rpn(out) for out in fpn_outs]
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        return x

#创建模型实例
model = QCNet()

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def load_model(model_path):
    model = QCNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    return model

def main(video_path, yolo_result_path, model_path):
    cap = cv2.VideoCapture(video_path)
    model = load_model(model_path)
    model.train()  #训练模式
    
    criterion = nn.MSELoss()  #使用均方误差作为损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001)  #使用SGD优化器
    
    frame_count = 0
    detection_interval = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open('people_count.csv', mode='w', newline='') as file, open(yolo_result_path, 'r') as yolo_file:
        writer = csv.writer(file)
        headers = ["Frame", "Time", "People_Count"]
        writer.writerow(headers)
        yolo_results = csv.reader(yolo_file)
        next(yolo_results)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = preprocess_image(frame)
            img = img.to(device)
            output = model(img)
            model_count = int(torch.round(output).item())
            try:
                yolo_row = next(yolo_results)
                yolo_count = int(yolo_row[2])  
            except StopIteration:
                break
            #将引导表转换为张量
            yolo_count_tensor = torch.tensor([yolo_count], dtype=torch.float32).to(device)
            #计算模型输出与引导表之间的损失
            loss = criterion(output, yolo_count_tensor)
            #反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if frame_count % detection_interval == 0:
                data_row = {'Frame': frame_count, 'Time': time.strftime("%H:%M:%S"), 'People_Count': model_count}
                writer.writerow([data_row[h] for h in headers])

            frame_count += 1

        cap.release()

    #保存训练后的模型权重
    torch.save(model.state_dict(), 'Source/QCNet/updated_counting_model.pth')

if __name__ == "__main__":
    video_path = 'Dataset/7700_1714833871.mp4'
    yolo_result_path = 'Source/QCNet/guide_table.csv'
    model_path = 'Source/QCNet/counting_model.pth'
    main(video_path, yolo_result_path, model_path)