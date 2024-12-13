# 필요한 모듈 임포트
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 설정값 정의
class config:
    MOMENTUM = 0.9

# 디렉토리 설정 (필요에 따라 수정)
run_folder = './'
run_archive_folder = './'

# 로깅 설정
lg = logging.getLogger('model_logger')
lg.setLevel(logging.INFO)
# 핸들러 추가 (필요 시)
if not lg.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    lg.addHandler(ch)

# Gen_Model 클래스 정의
class Gen_Model(nn.Module):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim):
        super(Gen_Model, self).__init__()
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim  # (채널 수, 높이, 너비)
        self.output_dim = output_dim
        self.optimizer = None  # 모델 정의 후에 설정됩니다.

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            v, p = self.forward(x)
        return v, p

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        self.train()
        dataset = torch.utils.data.TensorDataset(states, targets)
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_targets in train_loader:
                self.optimizer.zero_grad()
                v_pred, p_pred = self.forward(batch_states)

                v_target = batch_targets[:, 0].unsqueeze(1)  # Value 타깃
                p_target = batch_targets[:, 1:]  # Policy 타깃 (원핫 인코딩)

                loss_v = criterion_value(v_pred, v_target)
                loss_p = criterion_policy(p_pred, p_target.argmax(dim=1))

                loss = 0.5 * loss_v + 0.5 * loss_p

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if verbose:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return {'loss': total_loss, 'value_loss': loss_v.item(), 'policy_head_loss': loss_p.item()}

    def write(self, game, version):
        model_path = run_folder + f'models/{game}_' + "{0:0>4}".format(version) + '.pt'
        torch.save(self.state_dict(), model_path)

    def read(self, game, run_number, version):
        model_path = run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.pt'
        self.load_state_dict(torch.load(model_path))

    def printWeightAverages(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy()
                lg.info('PARAMETER {}: ABSAV = {:.6f}, SD = {:.6f}, ABSMAX = {:.6f}, ABSMIN = {:.6f}'.format(
                    name, np.mean(np.abs(weights)), np.std(weights), np.max(np.abs(weights)), np.min(np.abs(weights))
                ))
        lg.info('******************')

    def viewLayers(self):
        for name, param in self.named_parameters():
            weights = param.data.cpu().numpy()
            print('LAYER ' + name)
            try:
                s = weights.shape
                if len(s) == 4:
                    # Conv2d 가중치 시각화
                    num_filters = s[0]
                    num_channels = s[1]
                    fig = plt.figure(figsize=(num_channels, num_filters))
                    for i in range(num_filters):
                        for j in range(num_channels):
                            sub = fig.add_subplot(num_filters, num_channels, i * num_channels + j + 1)
                            sub.imshow(weights[i, j, :, :], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                    plt.show()
                elif len(s) == 2:
                    # Linear 가중치 시각화
                    fig = plt.figure(figsize=(3, 3))
                    sub = fig.add_subplot(1, 1, 1)
                    sub.imshow(weights, cmap='coolwarm', clim=(-1, 1), aspect="auto")
                    plt.show()
                else:
                    pass
            except Exception as e:
                print(f"An error occurred while visualizing layer {name}: {e}")
        lg.info('------------------')

# Residual_CNN 클래스 정의
# Residual_CNN 클래스 정의
class Residual_CNN(Gen_Model):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers):
        super(Residual_CNN, self).__init__(reg_const, learning_rate, input_dim, output_dim)
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self._build_model()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=config.MOMENTUM, weight_decay=self.reg_const)

    def conv_layer(self, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

        layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return layer

    def residual_layer(self, in_channels, out_channels, kernel_size):
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

        # 첫 번째 Conv 레이어
        conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        bn1 = nn.BatchNorm2d(out_channels)
        lrelu = nn.LeakyReLU()

        # 두 번째 Conv 레이어
        conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        bn2 = nn.BatchNorm2d(out_channels)

        # 입력 채널과 출력 채널이 다를 경우 크기 맞추기 위한 Conv 레이어
        match_channels = None
        if in_channels != out_channels:
            match_channels = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )
        #match_channels = None    

        return nn.ModuleDict({
            'conv1': conv1,
            'bn1': bn1,
            'lrelu1': lrelu,
            'conv2': conv2,
            'bn2': bn2,
            'match_channels': match_channels  # 각 Residual Layer에 맞춤 채널 추가
        })

    def value_head(self, in_channels):
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1 * self.input_dim[1] * self.input_dim[2], 20, bias=False),
            nn.LeakyReLU(),
            nn.Linear(20, 1, bias=False),
            nn.Tanh()
        )
        return layer

    def policy_head(self, in_channels):
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.input_dim[1] * self.input_dim[2], self.output_dim, bias=False)
            # 활성화 함수는 손실 함수에서 적용
        )
        return layer

    def _build_model(self):
        self.layers = nn.ModuleList()
        # 초기 Conv 레이어
        in_channels = self.input_dim[0]
        self.layers.append(self.conv_layer(in_channels, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size']))

        # Residual 레이어들
        for h in self.hidden_layers[1:]:
            in_channels = h['filters']   # 수정
            out_channels = h['filters']
            kernel_size = h['kernel_size']
            self.layers.append(self.residual_layer(in_channels, out_channels, kernel_size))
            in_channels = out_channels  # 다음 레이어의 입력 채널 수를 업데이트

        # Value와 Policy 헤드
        self.value_head_layer = self.value_head(in_channels)
        self.policy_head_layer = self.policy_head(in_channels)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.ModuleDict):
                # Residual Layer
                residual = x

                # 입력 채널 수와 출력 채널 수가 다를 경우 크기 맞추기
                if layer['match_channels'] is not None:
                    print("model.py line 254 input_dim", self.input_dim)
                    residual = layer['match_channels'](residual)

                out = layer['conv1'](x)
                out = layer['bn1'](out)
                out = layer['lrelu1'](out)
                out = layer['conv2'](out)
                out = layer['bn2'](out)

                # Residual 연결
                x = F.leaky_relu(out + residual)
            else:
                x = layer(x)

        v = self.value_head_layer(x)
        p = self.policy_head_layer(x)

        return v, p

    def convertToModelInput(self, state):
        inputToModel = state.binary  # state.binary가 NumPy 배열이라고 가정
        inputToModel = np.reshape(inputToModel, self.input_dim)
        inputToModel = torch.tensor(inputToModel, dtype=torch.float32)
        inputToModel = inputToModel.unsqueeze(0)  # 배치 차원 추가
        return inputToModel
