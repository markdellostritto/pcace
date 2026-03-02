#****************************************************
# Import Statements
#****************************************************

import numpy as np

__all__ = ["AngularProduct"]

#****************************************************
# Angular Product
#****************************************************

"""
	lvec
	@argument - order : the body order associated with the
		angular basis product
	@argument - l_max : the maximum angular momentum associated
		with the angular basis product
	This function returns a list of all possible products of angular basis
		functions associated with a given body order and maximum angular
		momentum l_max.  These products are guaranteed to have a final 
		total angular momentum which is less than l_max (inclusive).
	Each product is guaranteed not to be equivalent to another product.
		For instance, this means that no product except for the trivial 
		case of order=1 will have a zero included.
"""
def lvec(order: int, l_max: int):
	vec=[]
	match order:
		case 1:
			vec.append([0])
		case 2:
			for i in range(1,l_max+1):
				vec.append([i])
		case 3:
			for i in range(1,l_max+1):
				for j in range(i,l_max+1):
					if(i+j<=l_max): vec.append([i,j])
		case 4:
			tmp=[]
			for i in range(1,l_max+1):
				for j in range(1,l_max+1):
					for k in range(1,l_max+1):
						if(i+j<=l_max and j+k<=l_max):
							tmp.append([i,j,k])
			for n in range(0,len(tmp)):
				match=False
				for m in range(n+1,len(tmp)):
					diff=np.array(tmp[n])-np.array(tmp[m][::-1])
					if(np.linalg.norm(diff)==0):
						match=True
						break
				if not match: vec.append(tmp[n])
		case _:
			raise NotImplementedError("Max body order exceeded.")
	return vec

"""
	Angular Product Class
	@member o_max - The maximum body order.
	@member l_max - The maximum angular momentum.
    @member lprod - List of all possible angular basis products for every
		combination of order o and total angular momentum l up to the 
		limits (inclusive) of o_max and l_max.  
		Each product is stored as a list of total angular momenta, such that
		the product can be computed by taking the product of the angular bases
		(stored elsewhere) associated with each total angular momentum in the list.
		The products are stored in a single list, such that the maximum body 
		order associated with each product is contiguous.  The beginning and end
		of each contiguous section of the total list associated with a given
		body order can be accessed using the beg and end functions.
	@member p_size - A list giving the total number of products associated
		with the given body order o.  This is also the length of the subsection
		of the list lprod with products associated with the body order o.
		Note that the index of the list is offset from the body order, thus
		the integer stored at p_size[i] gives the number of products associated
		with the body order o=i+1.
	@member offset - Body order offset in the list of angular products.
		Note that the index of the list is offset from the body order, thus
		the integer stored at offset[i] gives the offset for products associated
		with the body order o=i+1.
	@member size - The total number of angular basis products.  Equivalent to 
		the sum over all elements of p_size.
	This class stores list of products of angular basis functions.  The actual products
		are not stored within this class, rather, a list of lists of total angular momenta
		are stored.  Each entry is a list of total angular momenta that can be used to
		take products of angular basis functions with the associated momenta to create
		a feature with the associated body order.  
	The products are stored in a list in such a way that the body order is contiguous 
		within the list.  The beginning and end of the sections of list associated with
		a given body order can be accessed using the beg and end functions.  Note that
		these take the body order itself, which has a minimum of one, and thus the input
		to these beg and end functions should not be thought of as a python list index.
"""
class AngularProduct:
	# ==== initialization ====
	"""
		o_max - max body order
		l_max - max angular momentum
	"""
	def __init__(self, o_max: int, l_max: int):
		# set order and lmax
		self.o_max=o_max
		self.l_max=l_max
		# compute lprod and size
		self.lprod=[]
		self.size=0	
		self.p_size = [0]*(o_max)
		for i in range(0,o_max):
			order = i+1
			vec=lvec(order,self.l_max)
			self.size+=len(vec)
			self.p_size[i]=len(vec)
			for lv in vec: self.lprod.append(lv)
		# compute offset
		self.offset = [0]*(o_max)
		for i in range(1,o_max):
			self.offset[i]=self.offset[i-1]+self.p_size[i-1]
	
	# ==== interval ====
	""" 
		beg of stack section for body order o
		note: o is the body order NOT the index
		thus, o must be greater than zero
	"""
	def beg(self, o: int):
		return self.offset[o-1]
	""" 
		end of stack section for body order o 
		note: o is the body order NOT the index
		thus, o must be greater than zero
	"""
	def end(self, o: int):
		return self.offset[o-1]+self.p_size[o-1]
	
	# ==== representation ====
	def __repr__(self):
		return (
			f"{self.__class__.__name__}(order={self.o_max}, l_max={self.l_max}, size={self.size})"
		)