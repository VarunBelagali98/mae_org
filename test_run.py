import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([512, 3, 224, 224], dtype=torch.half, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(3, 192, kernel_size=[16, 16], padding=[0, 0], stride=[16, 16], dilation=[1, 1], groups=1)
net = net.cuda().half()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()