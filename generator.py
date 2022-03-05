from network import NetG, weights_init
from config import opt, nz
from utils import TextureDataset, setNoise, learnedWN
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

resultsFolder = str(opt.outputFolder)
print(f"Reading from {resultsFolder}")

netG = NetG(opt.ngf, opt.nDep, nz)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

try:
    netG.apply(weights_init)
except Exception as e:
    print(e, "weightinit")
pass
net = netG.to(device)
print(net)

if opt.GenFile is not None:
    print(f"Loading previous generator weights {opt.GenFile}...")
    netG.load_state_dict(torch.load(opt.GenFile))
    print("Loaded successfully")

NZ = opt.imageSize // 2 ** opt.nDep
bignoise = torch.FloatTensor(opt.batchSize, nz, NZ*4, NZ*4)
bignoise = setNoise(bignoise)
noise = bignoise[:, :, :NZ, :NZ]

noise = noise.to(device)
bignoise = bignoise.to(device)
print(f"noise: {noise.shape}")
print(f"bignoise: {bignoise.shape}")

bigimg = netG(bignoise)
img = netG(noise)

for i in range(0, opt.batchSize):
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].imshow(((bigimg[i] + 1) / 2).cpu().detach().permute(1, 2, 0))
    ax[1].imshow(((img[i] + 1) / 2).cpu().detach().permute(1, 2, 0))
    fig.show()
    plt.close()

print(f"Finished...")
