#****************************************************
# Import Statements
#****************************************************

import torch
import time
from pcace.basis import cutoff

#****************************************************
# Cutoff definitions
#****************************************************

rc=6.0
nr=600
dr=rc/nr
rdata=torch.linspace(0.0,rc,nr).unsqueeze(-1)

cutoffStep=cutoff.CutoffStep(rc)
cutoffCos=cutoff.CutoffCos(rc)
cutoffPoly3=cutoff.CutoffPoly3(rc)

#****************************************************
# Test - Integration
#****************************************************

# ==== Step ====

intg=0
intg=intg+0.5*cutoffStep(rdata[0])
for i in range(1,nr-1):
    intg=intg+cutoffStep(rdata[i])
intg=intg+0.5*cutoffStep(rdata[nr-1])
intg=intg*dr
inte=rc
erri=abs(float(intg)-inte)

start = time.perf_counter()
cutoffStep(rdata)
end = time.perf_counter()

print("***************************************")
print("Test - Cutoff - Integration - Step")
print("num points       = %i" %(nr))
print("dr               = %.4f" %(dr))
print("integral - exact = %.4f" % (inte))
print("integral - numer = %.4f" % (float(intg)))
print("error - integral = %.4f" % (erri))
print("time             = %.4e" % ((end - start)/len(rdata)))
print("***************************************")

# ==== Cosine ====

intg=0
intg=intg+0.5*cutoffCos(rdata[0])
for i in range(1,nr-1):
    intg=intg+cutoffCos(rdata[i])
intg=intg+0.5*cutoffCos(rdata[nr-1])
intg=intg*dr
inte=0.5*rc
erri=abs(float(intg)-inte)

start = time.perf_counter()
cutoffCos(rdata)
end = time.perf_counter()

print("***************************************")
print("Test - Cutoff - Integration - Cosine")
print("num points       = %i" %(nr))
print("dr               = %.4f" %(dr))
print("integral - exact = %.4f" % (inte))
print("integral - numer = %.4f" % (float(intg)))
print("error - integral = %.4f" % (erri))
print("time             = %.4e" % ((end - start)/len(rdata)))
print("***************************************")

# ==== Poly3 ====

intg=0
intg=intg+0.5*cutoffPoly3(rdata[0])
for i in range(1,nr-1):
    intg=intg+cutoffPoly3(rdata[i])
intg=intg+0.5*cutoffPoly3(rdata[nr-1])
intg=intg*dr
inte=0.5*rc
erri=abs(float(intg)-inte)

start = time.perf_counter()
cutoffPoly3(rdata)
end = time.perf_counter()

print("***************************************")
print("Test - Cutoff - Integration - Poly3")
print("num points       = %i" %(nr))
print("dr               = %.4f" %(dr))
print("integral - exact = %.4f" % (inte))
print("integral - numer = %.4f" % (float(intg)))
print("error - integral = %.4f" % (erri))
print("time             = %.4e" % ((end - start)/len(rdata)))
print("***************************************")

#****************************************************
# Test - Gradient
#****************************************************

# ==== Step ====

errg=0
for i in range(1,nr-1):
    g1=0.5*(cutoffStep(rdata[i+1])-cutoffStep(rdata[i-1]))/dr
    r=rdata[i].clone().detach().requires_grad_(True)
    v=cutoffStep(r)
    v.backward()
    g2=float(r.grad)
    errg=errg+abs(g1-g2)
errg=errg/nr

print("***************************************")
print("Test - Cutoff - Gradient - Step")
print("num points       = %i" %(nr))
print("dr               = %.4f" %(dr))
print("error - gradient = %.4f" % (errg))
print("***************************************")

# ==== Cos ====

errg=0
for i in range(1,nr-1):
    g1=0.5*(cutoffCos(rdata[i+1])-cutoffCos(rdata[i-1]))/dr
    r=rdata[i].clone().detach().requires_grad_(True)
    v=cutoffCos(r)
    v.backward()
    g2=float(r.grad)
    errg=errg+abs(g1-g2)
errg=errg/nr

print("***************************************")
print("Test - Cutoff - Gradient - Cos")
print("num points       = %i" %(nr))
print("dr               = %.4f" %(dr))
print("error - gradient = %.4f" % (errg))
print("***************************************")

# ==== Poly3 ====

errg=0
for i in range(1,nr-1):
    g1=0.5*(cutoffPoly3(rdata[i+1])-cutoffPoly3(rdata[i-1]))/dr
    r=rdata[i].clone().detach().requires_grad_(True)
    v=cutoffPoly3(r)
    v.backward()
    g2=float(r.grad)
    errg=errg+abs(g1-g2)
errg=errg/nr

print("***************************************")
print("Test - Cutoff - Gradient - Poly3")
print("num points       = %i" %(nr))
print("dr               = %.4f" %(dr))
print("error - gradient = %.4f" % (errg))
print("***************************************")
