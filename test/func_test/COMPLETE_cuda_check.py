import torch
import torchvision

''' Test 1 '''
print("Test 1")
print("1 - torch version")
print(torch.__version__)
print("\n2 - torch device count")
print(torch.cuda.device_count())
print("\n3 - cuda device name")
print(torch.cuda.get_device_name(0))
print("\n4 - is cuda available?")
print(torch.cuda.is_available())

''' Test 2 - check torchvision version'''
print("\nTest - 2")
torch.set_default_device('cuda')
print(f"Torchvision version: {torchvision.__version__}")



''' Test 3 - simple torch calculation'''
# device = 'cuda'
# boxes = torch.tensor([[0., 1., 2., 3.]]).to(device)
# scores = torch.randn(1).to(device)
# iou_thresholds = 0.5


'''Test 4 - test cudnn if available'''
print("\ntest-4 - cudnn if available:")
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())

