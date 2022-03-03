""" 
MultiComplexMat implements a wrapper for any objects (reals, np.arrays 
and sparse matrices) which can have multiple imaginary like units.

E.g. rules i*i = -1, j*j = -1 but i*j = j*i will not simplify further. 

The MultiComplexMat overloads all common 
numerical operations: +,-,*,@ etc. such that these rules are preserved. 

For example 

x = a + bi + cj
y = d + ei
x*y = (a*d - b*e) + i*(a*e + d*b) + j*(c*d) + ij*(c*e)
x + y = a+d + (b+e)*i + cj       

Here a,b,c,d,e can be whatever objects implementing the common numerical
operations, e.g. numpy arrays or scipy sparse matrices.

One can use whatever characters as indices and one can query specific
components from the matrix with A["i"] or A["ij"]. Missing, or zero, 
components are indicated with "None" and they will translate to zeros 
in numerical operations. 

Warning: The objects inside MultiComplexMat objects are aggressively 
recycled, i.e. in a sum C = A + B, if A["i"] == None, C["i"] will be the 
exact object which was stored in B["i"] and thus any mutations done for 
C["i"] will be visible in B["i"] also.
"""

import numpy as np
import itertools

from collections import defaultdict
import time

from . import dict_tools
import scipy.sparse as sps

def get_linear_system(A, b, **kwargs):
    """ Get a linear system from multicomplex matrices A and b such that
    A x = b is decomposed for each component and stacked in a sparse CSC 
    matrix which can be given e.g. to spsolve.
    
    The equation system is, e.g. for "ij" component string:
    A x = b 
    => (A*x)[""]   = b[""]
       (A*x)["i"]  = b["i"]
       (A*x)["j"]  = b["j"]
       (A*x)["ij"] = b["ij"]
    
    Example:  
        # components '',i,j,ij
        A = 1 + 1*i - 1*j
        b = 10 + 10*i*j
        C, d = get_linear_system(A,b) 
        
        => C = array([[ 1,  1, -1,  0],
                      [-1,  1,  0, -1],
                      [ 1,  0,  1,  1],
                      [ 0,  1, -1,  1]])
        
           d = array([[10],
                      [0],
                      [0],
                      [10]])
        x = scipy.sparse.linalg.spsolve(C,d)
        
        => x = array([ 2.,  4., -4.,  2.])
    """
    order = list(A.components())
    order_lookup = dict(map(reversed, enumerate(order)))
    sysmat = {}
    bcol = len(order)
    shapes = []
    # assemble the system as single matrix and slice the last column off
    # to utilize the shape information of the matrix blocks also in b
    for c1, val in A.data.items():
        for c2, col in order_lookup.items():
            sign, comp = simplify(c1+c2)
            # row = order_lookup[c2]
            row = order_lookup[comp]
            old = sysmat.get((row,col), None)
            if old is None:
                sysmat[(row,col)] = sign*val
            else:
                sysmat[(row,col)] = old + sign*val
                
    for c, v in b.data.items():
        row = order_lookup[c]
        sysmat[(row, bcol)] = v
        shapes.append(np.shape(v))
                
    lst = dict_tools.tolist(sysmat)
    M = sps.bmat(lst,format='csc')
    if kwargs.get('get_shapes', False):
        return (M[:,:-1], M[:,-1]), shapes
    else:
        return (M[:,:-1], M[:,-1])

def to_mcmat(cstr, arr, shapes):
    start = 0
    components = []
    for i in range(0,len(shapes)):
        stop = start + shapes[i]
        if start == stop:
            components.append(None)
        else:
            vals = arr[start:stop]
            if stop-start == 1:
                components.append(vals[0])
            else:
                components.append(vals)
        start = stop
            
    data = dict(zip(all_components(cstr), components))
    return mcmat(cstr, data)    

def all_components(compstr):
    return itertools.chain([""], combinations(compstr))
        
def combinations(components):
    """ Return all possible 1..n combinations, order doesn't matter 
    e.g. "ij" -> "i", "j", "ij" """
    for i in range(1,len(components)+1):
        for x in itertools.combinations(components,i):
            yield "".join(x)

def unified_component_string(*mcmats):
    concat = "".join([m.component_str for m in mcmats])
    out = list(set(concat))
    out.sort()
    return "".join(out)

def simplify(lst):
    """ Given a component string 'lst' use simplification rules 
    (e.g. i*i = -1) to simplify it. Return the sign and the simplified 
    string. 
    
    Example: simplify('ijki') = (-1,'jk') """
    n = len(lst)

    # premature optimization
    if(n == 1):
        return 1, lst
    elif(n == 2):
        if lst[0] == lst[1]:
            return -1, ''
        else:
            return 1, "".join(sorted(lst))

    # general slow-ass algorithm for n > 2       
    d = defaultdict(lambda: 0)
    for t in lst:
        d[t] = d[t]+1
        
    terms_left = []
    sign = 1
    for t,v in d.items():
        if v % 2 == 0:
            sign = sign*(-1)**(int(v/2))
        else:
            terms_left.append(t)
            sign = sign*(-1)**int((v-1)/2)

    # keep in alphabetical order    
    terms_left.sort()
    return sign, "".join(terms_left)

def mcmat(components, values, ident=None):
    """Construct a MultiComplexMat object from values.
    
    Components is a string indicating which characters act as the components.
    Values is a dict: {component_string: object} where component_string
    are the components strings which can be formed using 'components'.
    
    The empty string "" represents the real component.
    
    Example: 
        mcmat("abc", {"": 1, "ab": 5, "ac": 15, "abc": 50}) // "bc" component 
                                                            // is zero
    """
    data = {}
    for k in itertools.chain([""], combinations(components)):      
        item = values.pop(k, None)
        if item is not None: 
            data[k] = item

    if values:
        components = list(itertools.chain([""], combinations(components)))
        raise ValueError(f"Extra components {list(values.keys())} given "
                         "which is not allowed. Only components "
                         f"{components} are needed.")
            
        
    return MultiComplexMat(components, data, ident, None)

def realmcmat(value, ident=None):
    """ Construct a MultiComplexMat object with only "" component """
    return MultiComplexMat("", {"": value}, ident, None)

def sub_with_none(v1,v2):
    """ Substract treating None as zero """
    p1 = v1 is None
    p2 = v2 is None
    
    if p1 and p2:
        return None
    elif p1:
        return -v2
    elif p2:
        return v1
    else:
        return v1 - v2
    
def sum_with_none(v1,v2):
    """ Sum treating None as zero """
    p1 = v1 is None
    p2 = v2 is None
    
    if p1 and p2:
        return None
    elif p1:
        return v2
    elif p2:
        return v1
    else:
        return v1 + v2    
    
def matmul_with_none(v1,v2):
    """ Matmul treating None as zero """
    if v1 is None:
        return None
    elif v2 is None:
        return None
    else:
        return v1 @ v2

def mul_with_none(v1,v2):
    """ Standard mul treating None with zero """
    if v1 is None:
        return None
    elif v2 is None:
        return None
    else:
        return v1 * v2


def dictzip(*dicts):
    """ Iterate over multiple dicts 'zipping' their elements with 
    matching keys. If some of the dicts are missing the entries, 
    they will be None."""
    keyset = set(itertools.chain(*dicts))
    return ((k, *[d.get(k,None) for d in dicts]) for k in keyset)

    
#%%
class MultiComplexMat():
    
    def __init__(self, component_str, data, ident, default_zero_constructor):
        self.component_str = component_str
        self.data = data   
        self.ident = ident
        self.default_zero_constructor = default_zero_constructor
        
    def _shallow_copy(self, *objs, **kwargs):
        """ Create shallow copy of the object, inheriting all possible data
        from 'self' and overriding the stuff passed in with 'kwargs' """
        
        cstr = kwargs.pop('component_str', None)
        if cstr == None:
            cstr = unified_component_string(self, *objs)
        data = kwargs.pop('data', self.data)
        ident = kwargs.pop('ident', self.ident)
        default_zero_constructor = kwargs.pop('default_zero_constructor',
                                              self.default_zero_constructor)
        # assert not (re is None or im is None), "re and im must be specified"
        return MultiComplexMat(cstr, data, ident, default_zero_constructor)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): 
        ufname = ufunc.__name__
        if ufname == 'matmul':            
            # matmul f @ B where f is a numpy array 
            # easy way, elevate inputs to mcmats and then multiply
            B = inputs[1]
            A = realmcmat(inputs[0])
            return A @ B
        elif ufname == 'multiply':
            B = inputs[1]
            A = realmcmat(inputs[0])
            return A * B
        # elif ufname == 'absolute':
        #     return inputs[0].abs()
        elif ufname == 'subtract':
            B = inputs[1]
            A = realmcmat(inputs[0])
            return A - B
        elif ufname == 'add':
            B = inputs[1]
            A = realmcmat(inputs[0])
            return A + B
        # elif ufname == 'conjugate':
        #     return inputs[0].conj()
        # elif ufname == 'sqrt':
        #     raise NotImplementedError()
        else: 
            from debug import debug
            debug()

    def _mul_generic(self, obj_in, op):
        """ Generic multiplication machinery, * and @ are implemented using 
        this
        """              
        
        if not isinstance(obj_in, MultiComplexMat): 
            # Wrap whateber obj is into a multicomplexmat
            obj = realmcmat(obj_in)
            from utils.debug import debug
            debug()            
        else:
            obj = obj_in

            
            
        d = dict()
        for k1, v1 in self.data.items():
            for k2, v2 in obj.data.items():
                newind = "".join([k1,k2])
                sign, left = simplify(newind)
                old = d.get(left, None)
                result = op(v1,v2)
                if old is None:
                    d[left] = sign*result if result is not None else result
                else: 
                    d[left] = old + sign*result if result is not None else 0
        return self._shallow_copy(obj, data=d)
    
    def __matmul__(self, obj):
        return self._mul_generic(obj, matmul_with_none)

    def __mul__(self, obj):
        return self._mul_generic(obj, mul_with_none)
    
    def set_default_constructor(self, fun):
        """ The default shape of the contained matrices, this affects
        the value returned by the indexing [] operator. If default shape is 
        None, indexing returns None for an empty component. 
        Otherwise it calls 'fun' and returns whatever it returns. """
        self.default_zero_constructor = fun
        
    def __rmatmul__(self, obj):
        # is called if f @ A where f is not a MultiComplexMat object
        # except when f is np.array, then an ufunc is called
        A = realmcmat(obj)
        return A.__matmul__(self)

    def components(self):
        """ Return all possible components of this MultiComplexMat object """
        #return [""] + list(combinations(self.component_str))
        return all_components(self.component_str)

    def __repr__(self):
        elems = {}
        for k in self.components():
            v = self.data.get(k,None)
            if v is None:
                continue
            elif np.shape(v) == ():
                elems[k] = v
            else:
                elems[k] = np.shape(v)
                
        return f"MultiComplexMat({elems})"

    def __delitem__(self, key):
        self.data.__delattr__(key)

    def __getitem__(self, key):
        """ The [] indexing. If the component is 'None' check if 
        default constructor is set. This helps with arithmetic when using 
        components explicitly, e.g. A["i"] @ A["j"] will produce an error
        if either one is None. """
        item = self.data.get(key,None)
        if item is None and self.default_zero_constructor != None:
            return self.default_zero_constructor()
        return item

    def __setitem__(self, key, value):
        self.data[key] = value

    # def toarray(self):
    #     return self.re + 1j*self.im

    # def __mul__(self, obj):
    #     # is called when A @ B where B is whatever
    #     if isinstance(obj, MultiComplexMat):
    #         re = self.re * obj.re - self.im * obj.im
    #         im = self.re * obj.im + self.im * obj.re            
    #     else:
    #         # multiplication with a numpy array or constant
    #         ore, oim = csplit(obj)
    #         re = self.re * ore - self.im * oim
    #         im = self.re * oim + self.im * ore       
    #     return self._shallow_copy(re=re, im=im)
        
    def __rmul__(self, obj):
        # is called if f @ A where f is not a MultiComplexMat object
        # except when f is np.array, then an ufunc is called
        A = realmcmat(obj)
        return A.__mul__(self)
    
    def __radd__(self, obj):
        # is called if f @ A where f is not a MultiComplexMat object
        # except when f is np.array, then an ufunc is called
        A = realmcmat(obj)
        return A.__add__(self)
        
    
    def _sum_generic(self, obj, op):
        """ Implements + and - operations """
        if not isinstance(obj, MultiComplexMat): 
            # Wrap whateber obj is into a multicomplexmat
            obj = realmcmat(obj)        

        data = {k: op(v1,v2) 
                for k,v1,v2 in dictzip(self.data, obj.data)}
        return self._shallow_copy(obj, data=data)      

    
    def __add__(self, obj):
        return self._sum_generic(obj, sum_with_none)
        

    def __sub__(self, obj):
        return self._sum_generic(obj, sub_with_none)   

    def __rsub__(self, obj):
        # is called if f @ A where f is not a MultiComplexMat object
        # except when f is np.array, then an ufunc is called
        A = realmcmat(obj)
        return A.__sub__(self)
        
    # def elementwise_inv(self):
    #     return self.conj()*(1/self.abs()**2)
    
    def abs(self):
        return np.sqrt(self.re**2 + self.im**2)
    
    # def angle(self):
    #     return np.arctan2(self.im, self.re)
    
    # def conj(self):
    #     return self._shallow_copy(im=-self.im)
    
    # @property
    # def real(self):
    #     return self.re
    # @property
    # def imag(self):
    #     return self.im
    
    # def __pow__(self,n):
    #     magn = self.abs()**n
    #     ang = self.angle()
        
    #     re = magn*np.cos(ang*n)
    #     im = magn*np.sin(ang*n)
        
    #     return self._shallow_copy(re=re, im=im)
    
    # def __truediv__(self, B):
    #     # is called when A @ B where B is whatever
    #     if not isinstance(B, MultiComplexMat):
    #         B = mcmat(B)

    #     return self*B.elementwise_inv()      
#%%

""" 
For convenience, define few arrays which can be easily used in expressions
"""
i = mcmat("i", {'i':1})
j = mcmat("j", {'j':1})
k = mcmat("k", {'k':1})


