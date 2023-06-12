import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from mnist_cnn import mnist_test, device, model

model.load_state_dict(torch.load('model.pt'))

#테스트, 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())