import numpy as np

def split2d(array, nAgents_v, nAgents_h=[]):
    if nAgents_h==[]:
        nAgents_h = nAgents_v

    array_split_v = np.array_split(array, nAgents_v, axis=0)
    array_split_hv = map(lambda x:  np.array_split(x, nAgents_h,axis=1), array_split_v)

    mystatelist = []

    for arr in array_split_hv:
        for subarr in arr:

            #print(subarr,'\n')
            mystatelist.append(subarr.reshape(-1,).tolist())

    return mystatelist


def pickcenter(array, Nx, Ny, nAgents_v, nAgents_h=[]):
    if nAgents_h==[]:
        nAgents_h = nAgents_v
    
    ix = np.linspace(0,Nx,nAgents_h,endpoint=False).astype('int')
    iy = np.linspace(0,Ny,nAgents_v,endpoint=False).astype('int')

    array_agents = array[ix,:][:,iy]
    
    mystatelist = array_agents.reshape(-1,1).tolist()

    return mystatelist
