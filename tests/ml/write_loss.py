#****************************************************
# Import Statements
#****************************************************

import torch
from pcace.ml import loss_fn

#****************************************************
# Loss Parameters
#****************************************************

a=1.0
ndata=200
xdata=torch.linspace(-4.0,4.0,ndata).unsqueeze(-1)
lossMSE=loss_fn.LossMSE(a=a)
lossHuber=loss_fn.LossHuber(a=a)
lossAsinh=loss_fn.LossAsinh(a=a)

#****************************************************
# Compute Loss
#****************************************************

lmse=torch.tensor([lossMSE(x,torch.zeros_like(x)) for x in xdata])
lhuber=torch.tensor([lossHuber(x,torch.zeros_like(x)) for x in xdata])
lasinh=torch.tensor([lossAsinh(x,torch.zeros_like(x)) for x in xdata])

#****************************************************
# Write Loss
#****************************************************

xx=xdata.flatten().numpy()
y1=lmse.flatten().numpy()
y2=lhuber.flatten().numpy()
y3=lasinh.flatten().numpy()
writer=open("loss.dat","w")
writer.write("#X MSE HUBER ASINH\n")
for i in range(ndata):
    writer.write(f"%f %f %f %f\n"%(xx[i],y1[i],y2[i],y3[i]))
writer.close()
