###############################################
# This module contains functions to compute the angular part of the 
# edge basis functions
###############################################

import torch

"""
    Cartesian Angular Basis
    @member l_max - the maximum angular momentum
    @member l_size - the total number of angular functions corresponding
        to a given total angular momentum l.  Thus, l_size[0] is the number
        of functions with total angular momentum 0, l_size[1] is the number of functions
        with total angular momentum 1, etc.
    @member offset - Angular momentum offset in a stacked list of Tensors.
        When computed for a given distance vector dr, the angular basis functions are returned
        as a stacking of a list of Tensors.  These Tensors are stored with increasing 
        total angular momentum, and it's important to know the total angular momentum
        associated with each Tensor (angular function) in the stack.  This information is 
        not available from the functions themselves.  However, each function with the same
        total angular momentum is stored contiguously.  Thus, if one knows the l_size and
        offset for each l, one can find all Tensors within the stack representing the 
        calculation of angular basis functions with total angular momentum l.
    @member size - the total number of angular basis functions.  This is equivalent
        to the sum over all elements of l_size
    This class stores information about the angular basis and computes the angular basis
        for a given set of difference vecs dr.  This class computes all angular basis functions
        up to a maximum total angular moment l_max (inclusive).  The angular basis functions
        are stored in a stacked list of Tensors such that all angular functions with the
        same total angular momentum are stored contiguously.  The beginning and end of sections
        of the stacked list with the same total angular momentum l can be accessed with
        the functions beg(l) and end(l).
    The angular basis functions are Cartesian spherical harmonics.  Thus, for a given total
        angular momentum L the basis functions are all combinations of x^Lx*y^Ly*z^Lz
        such that Lx+Ly+Lz=L.  Note that this includes repeated combinations of (Lx,Ly,Lz).
        For example, for L=2, the functions (1,1,0), (1,0,1), and (0,1,1) are included twice
        in the stacked list.  This method of storage simplifies construction and products of
        the basis.  For instance, to get all functions for L=x+1 for a given distance vector
        dr one simply multiplies each basis function with L=x with dr[0], dr[1], and dr[2].
        This storage method also greatly simplifies products of two angular bases as well
        as vector operations expressed using the angular basis.
"""
class AngularBasis(torch.nn.Module):
    # ==== initialization ====
    """
		l_max - max angular momentum
	"""
    def __init__(self, l_max: int):
        super().__init__()
        self.l_max = l_max
        self.l_size = [0]*(l_max+1)
        self.offset = [0]*(l_max+1)
        # compute size
        cSize = 1
        self.size = 0
        for l in range(l_max+1):
            self.l_size[l]=cSize
            self.size+=self.l_size[l]
            cSize*=3
        for l in range(1,self.l_max+1):
            self.offset[l]=self.offset[l-1]+self.l_size[l-1]
    
    # ==== calculation ====
    """
		dr - distance vector from central to neighbor atom
	"""
    def forward(self, dr: torch.Tensor) -> torch.Tensor:
        # l = 0 term
        abasis=[torch.ones(dr.size()[0],device=dr.device,dtype=dr.dtype)]
        # l > 0 terms
        for l in range(1,self.l_max+1):
            for m in range(0,self.l_size[l-1]):
                c=self.offset[l-1]+m
                abasis.append(abasis[c]*dr[:,0])
                abasis.append(abasis[c]*dr[:,1])
                abasis.append(abasis[c]*dr[:,2])
        # return stack over list of basis functions
        return torch.stack(abasis,dim=1)

    # ==== interval ====
    """ 
        beg of stack section for total angular momentum l 
        note: l can be thought of as either the total angular momentum
        or the index of the arrays offset and l_size
        this is because the minimum angular momentum is zero
    """
    def beg(self, l: int):
        return self.offset[l]
    """ 
        end of stack section for total angular momentum l 
        note: l can be thought of as either the total angular momentum
        or the index of the arrays offset and l_size
        this is because the minimum angular momentum is zero
    """
    def end(self, l: int):
        return self.offset[l]+self.l_size[l]
    
    # ==== representation ====
    def __repr__(self):
        return f"AngularBasis(l_max={self.l_max},size={self.size})"
        