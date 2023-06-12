import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#랜덤 시드 설정
torch.manual_seed(1)

#훈련데이터(x), 테스트데이터(y) 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#가중치 w를 0으로 초기화하고, 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True) #requires_grad=True : 텐서에 대한 기울기 저장

# 편향 B를 0으로 초기화하고, 학습을 통해 값이 변경되는 변수임을 명시함.
b = torch.zeros(1, requires_grad=True)

#경사하강법
optimizer = optim.SGD([W, b], lr=0.01) #SGD : 경사하강법의 일종, lr : 학습률(learning rate)

nb_epochs = 1999 #원하는만큼 경사 하강법을 반복

for epoch in range(nb_epochs + 1):

    #가설 설정(h(x) = wx+b)
    hypothesis = x_train * W + b

    #비용 함수 선언
    cost = torch.mean((hypothesis - y_train) ** 2)

    #gradient를 0으로 초기화
    optimizer.zero_grad()

    #cost를 미분하여 gradient 계산
    cost.backward()

    #가중치와 편향 업데이트
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))