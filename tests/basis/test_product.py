#****************************************************
# Import Statements
#****************************************************

from pcace.basis import product
from pcace.basis import AngularProduct

#****************************************************
# Angular Definitions
#****************************************************

o_max=4 # max body order
l_max=4 # max angular momentum

for n in range(1,o_max+1):
    print("**************** order ",n," ****************")
    for l in range(0,l_max+1):
        lvec=product.lvec(n,l)
        print("lvec[",l,"] = ",lvec)

print("=============================================")
prod = AngularProduct(o_max=2,l_max=3)
print("prod = ",prod)
print("lprod = ",prod.lprod)

lim=[(prod.beg(o+1),prod.end(o+1)) for o in range(0,prod.o_max)]
print('lim = ',lim)
for i in range(0,prod.o_max):
    order=i+1
    print('lim[',order,'] = ',lim[i])

for o in range(0,prod.o_max):
    order=o+1
    print("order = ",order)
    for i in range(prod.beg(order),prod.end(order)):
        print(prod.lprod[i])
