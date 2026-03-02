#****************************************************
# Import Statements
#****************************************************

import torch
from pcace.basis import radial
import time

#****************************************************
# Radial Parameters
#****************************************************

rc=6.0
ndata=600
dr=rc/ndata
rdata=torch.linspace(0.0,rc,ndata).unsqueeze(-1)
nr=8

radialBessel=radial.RadialBessel(rc,nr)

#****************************************************
# Test - Gradient
#****************************************************

# ==== Bessel ====

errg=0
for i in range(2,ndata-1):
    g1=0.5*(radialBessel(rdata[i+1])-radialBessel(rdata[i-1]))/dr
    r=rdata[i].clone().detach().requires_grad_(True)
    v=radialBessel(r).mean()
    v.backward()
    g2=float(r.grad)
    errg=errg+abs(float(g1.mean())-g2)
errg=errg/ndata

start = time.perf_counter()
radialBessel(rdata)
end = time.perf_counter()

print("***************************************")
print("Test - Radial - Gradient - Bessel")
print(radialBessel)
print("num radial = %i" %(nr))
print("num points = %i" %(ndata))
print("dr         = %.4f" %(dr))
print("error      = %.4f" % (errg))
print("time       = %.4e" % ((end - start)/ndata))
print("***************************************")

#print(radialBessel.state_dict())
