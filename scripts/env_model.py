import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd


# 定义LSTM模型
class TorqueToAngleLSTM(nn.Module):
    def __init__(self):
        super(TorqueToAngleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)  # 输出层

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM层
        out = self.fc(out)  # 输出层
        return out

def load_data(file_path):
    knee_angle = pd.read_excel(file_path, sheet_name='knee_angle').to_numpy()
    knee_moment = pd.read_excel(file_path, sheet_name='knee_moment').to_numpy()
    return knee_angle.T, knee_moment.T

if __name__ =='__main__':
    # file_path = 'Datasets/merged_data.xlsx' 
    file_path = 'Datasets/AB01/Left.xlsx' 
    angle_data, moment_data = load_data(file_path)
    # 将数据转换为Tensor
    num_samples = angle_data.shape[0]
    sequence_length = angle_data.shape[1]
    torque_tensor = torch.tensor(moment_data, dtype=torch.float32).view(num_samples, sequence_length, 1)
    angle_tensor = torch.tensor(angle_data, dtype=torch.float32).view(num_samples, sequence_length, 1)

    # 创建数据集和数据加载器
    dataset = TensorDataset(torque_tensor, angle_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # 初始化模型
    model = TorqueToAngleLSTM()

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 600
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算每个时间步的误差

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 预测结果
    with torch.no_grad():
        model.eval()
        predicted_angle = model(torque_tensor[:2]).numpy()  # 获取完整的预测序列

    # 可视化预测结果与真实结果
    for i in range(2):
        plt.plot(angle_tensor[i].numpy(), label=f'Target Angle {i+1}', linestyle='dashed')
        plt.plot(predicted_angle[i], label=f'Predicted Angle {i+1}')
        plt.legend()
        plt.title(f'Comparison of Target vs Predicted Angles for Sample {i+1}')
        plt.show()

    # 保存模型的参数
    torch.save(model.state_dict(), 'model.pth')
