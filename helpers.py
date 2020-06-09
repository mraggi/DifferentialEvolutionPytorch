import random

def randfloat(a,b):
    return a + (b-a)*random.random()

def tofunc(x):
    if isinstance(x,float) or isinstance(x,int): 
        return lambda : x
    if isinstance(x,slice):
        return lambda : randfloat(x.start,x.stop)
    if isinstance(x,tuple) or isinstance(x,list):
        return lambda : randfloat(x[0],x[1])
    return x