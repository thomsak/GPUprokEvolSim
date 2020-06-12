import sys 
import numpy as np
import math
import numba
from numba import cuda, int64, jit
from numba.cuda import random as rnd
from timeit import default_timer as timer
from scipy.stats import binom, poisson
import pickle as pkl
#np.set_printoptions(threshold=np.nan)
import argparse
import os

@cuda.jit
def evolve(rng_states,gen,numberGenerations,genotypes,binomcdf,parentstree,rhocdf,recombination_marks,recomblen,recombmodel,recombpar):
    cellThreadID = cuda.grid(1)
    #row = cuda.blockIdx.y*cuda.blockDim.y+cuda.threadIdx.y
    #col = cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    #cellThreadID = row*(cuda.blockDim.y*cuda.blockDim.x)+col
    genoblocks = genotypes.shape[1]/2
    L=64*genoblocks
    N=genotypes.shape[0]
    numberLocalCells = 1
    for numbIt in range(numberGenerations):
        for localI in range(numberLocalCells):
            cellID = (cellThreadID*numberLocalCells) + localI
            if cellID < N:
               if (gen+numbIt) % 2 == 1:
                   targetoffset = 0
                   sourceoffset = int(genotypes.shape[1]/2)
               else:
                   targetoffset = int(genotypes.shape[1]/2)
                   sourceoffset = 0
               parent=-1
               if parentstree.shape[1]<(gen):
                   for index in range(N):
                      if parentstree[-(gen+2),index]==(cellID):
                          parent = parentstree[-(gen+1),index]
                          recordRec=True
                          break
               if parent == -1:
                   parent = int(math.floor(N*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)))
                   recordRec=False
               for i in range(genoblocks):
                   genotypes[cellID,i+targetoffset]=genotypes[parent,i+sourceoffset]
               #add recombinations
               r=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
               for k in range(100):
                   if r < rhocdf[k]:
                       break
               numberRecombinations=k
               for i in range(numberRecombinations):
                   recombine = True
                   parent2 = int(math.floor(N*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)))
                   #pos2 = int(math.floor(2*L*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)))
                   #posInt2 = pos2//64
                   posInt2 = 2*int(math.floor((genoblocks)/2*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)))
                   if recombmodel == 1 or recombmodel == 2:
                       diff=0
                       for ki in range(recomblen*2):
                           offset=int((posInt2+ki)%genoblocks)
                           delta = genotypes[cellID,offset+targetoffset] ^ genotypes[parent2,offset+sourceoffset]
                           for j in range(0,64,2):
                               if int64(delta) & 1 == 1 or int64(delta) & 10 == 10 or int64(delta) & 11 == 11:
                                   diff = diff + 1
                               delta = math.floor(delta / 4)
                               if delta == 0:
                                   break
                           rd=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
                           if recombmodel == 1:
                               maxsnps = math.floor(-recomblen*64*math.log(rd)*recombpar)
                           if recombmodel == 2:
                               alpha=recombpar
                               maxsnps = math.floor(recomblen*64/100 * (1-rd) ** (-1/(alpha-1)))
                           if diff > maxsnps:
                               recombine=False
                   #this might be quite slow:
                   if recombmodel == 3:
                       diff=0
                       for ki in range(genoblocks):
                           delta = genotypes[cellID,targetoffset+ki] ^ genotypes[parent2,sourceoffset+ki]
                           for j in range(0,64,2):
                               if int64(delta) & 1 == 1 or int64(delta) & 10 == 10 or int64(delta) & 11 == 11:
                                   diff = diff + 1
                               delta = math.floor(delta / 4)
                               if delta == 0:
                                   break
                           rd=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
                           alpha=recombpar
                           maxsnps = math.floor(genoblocks*64/100 * (1-rd) ** (-1/(alpha-1)))
                           if diff > maxsnps:
                               recombine=False

                   if recombine:
                       for ki in range(recomblen*2):
                           offset=int((posInt2+ki)%genoblocks)
                           genotypes[cellID,targetoffset+offset] = genotypes[parent2,sourceoffset+offset]
                       if recordRec:
                           for ki in range(recomblen*64):
                               offset=int((int(posInt2/2*64)+ki)%recombination_marks.shape[1])
                               #recombination_marks[cellID,offset]+=1
               #add mutations
               r=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
               for k in range(100):
                   if r < binomcdf[k]:
                       break
               numberMutations=k
               for i in range(numberMutations):
                   pos = 2*int(math.floor(L/2*rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)))
                   posInt = pos//64
                   posInInt = pos%64
                   r=rnd.xoroshiro128p_uniform_float32(rng_states, cellThreadID)
                   if r < 1/4:
                       #transverion (first bit with probability 1/2)
                       if r < 1/8:
                           genotypes[cellID,posInt+targetoffset]=genotypes[cellID,posInt+targetoffset] ^ 2**posInInt
                       #second bit has to be flipped
                       genotypes[cellID,posInt+targetoffset]=genotypes[cellID,posInt+targetoffset] ^ 2**(posInInt+1)
                   else:
                       #transition (00<->10 or 01<->11)
                       genotypes[cellID,posInt+targetoffset]=genotypes[cellID,posInt+targetoffset] ^ 2**posInInt

        cuda.syncthreads()

from scipy.optimize import fsolve
def mufunct(mu,N,S):
    return np.prod([1/(1+2*mu*N/(k-1)) for k in range(2,S+1)])-0.9



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="the seed of the simulation (default=1)", default=0)
parser.add_argument("-v", "--verbose", help="increase output verbosity", action='count')
parser.add_argument("--threadsperblock", type=int, help="the number of threads per block on the GPU; N=threadsperblock*blocks ", default=60)
parser.add_argument("--blocks", type=int, help="the number of blocks; N=threadsperblock*blocks", default=60)
parser.add_argument("-S", "--samplesize", type=int, help="the size of sample", default=50)
parser.add_argument("-p", "--fractionvariable", type=float, help="the fraction of positions which are expected to be variable", default=0.9)
parser.add_argument("-r", "--rhoovermu", type=float, help="rho/mu", default=1.0)
parser.add_argument("-t", "--guidetree", help="the path of the guide tree fille, empyt with create a new one")
parser.add_argument("-I", "--ID", help="the ID of the simulation run", required=True)
parser.add_argument("-l", "--recombinationlength", type=int, help="the length of the recombined stretches", default=200)
parser.add_argument("-m", "--recombinationmodel", type=int, help="the recombination model used, 0=random, 1=local exponential cut-off, 2=local powerlaw cut-off, 3=global powerlaw cut-off", default=0)
parser.add_argument("-x", "--recombinationparameter", type=float, help="the parameter for the recombination model; for exponential this the parameter of the exponential (like 0.01 or so), for powerlaws this is the powerlaw exponent (like 2 or 3 or so)", default=0.01)
args = parser.parse_args()

if args.verbose:
    print("number of cores: "+str(numba.config.NUMBA_DEFAULT_NUM_THREADS))

ID=args.ID
if not os.path.exists(ID):
    os.makedirs(ID)
L=40000
n=64*L
threadsperblock = args.threadsperblock
blocks = args.blocks
N=threadsperblock*blocks
S=args.samplesize
#p=-np.log(0.9)/(2*np.log(S)*N)
#print("p1: "+str(p))
p2=fsolve(mufunct,0.000001,args=(N,S))[0]
if args.verbose:
    print("p2: "+str(p2))
binomcdf = np.zeros((100),dtype=np.float32)
for k in range(100):
    binomcdf[k]=binom.cdf(k, n, p2)
rhofact=args.rhoovermu
rho=rhofact*p2*n
rhocdf = np.zeros((100),dtype=np.float32)
for k in range(100):
    rhocdf[k]=poisson.cdf(k,rho)
if args.verbose:
    print("rho: "+str(rho))
recomblen=args.recombinationlength
recombmodel=args.recombinationmodel
recombpar=args.recombinationparameter
SEED=args.seed
logfile=open(ID+"/"+ID+".log",'w')
logfile.write("L: "+str(L)+"\n")
logfile.write("n: "+str(n)+"\n")
logfile.write("N: "+str(N)+"\n")
logfile.write("S: "+str(S)+"\n")
logfile.write("p2: "+str(p2)+"\n")
logfile.write("rho: "+str(rho)+"\n")
logfile.write("recombinationlength: "+str(recomblen)+"\n")
logfile.write("recombinationmodel: "+str(recombmodel)+"\n")
logfile.write("recombination parameter: "+str(recombpar)+"\n")
logfile.write("seed: "+str(SEED)+"\n")
NUMCYCLES=8*N
subindx = np.random.choice(N,S,replace=False)
subindx=range(0,S)
parentstree=[]
parentstree.append(subindx)
parentsindx=subindx
active=S
if args.guidetree:
    guidefile=args.guidetree
    parentstree= np.loadtxt(guidefile,dtype=int)
    logfile.write(guidefile+"\n")
else:
    positionsactive=list(range(S))
    maxx=S
    for i in range(NUMCYCLES):
        parentsindx=np.random.choice(N,S,replace=True) 
        seen={}
        labelindx=[]
        #print(len(parentsindx))
        for pi,pa in enumerate(parentsindx):
            #maxlabel+=1
            if (not pa in seen) or (pi not in positionsactive):
                label=parentstree[-1][pi]#maxlabel
                if pi in positionsactive:
                    seen[pa]=label
            else:
                label=seen[pa]
            labelindx.append(label)
        #print(labelindx)
        alreadyseen=[]
        for indx in range(S):
            if not indx in positionsactive:
                labelindx[indx]=-1
        labelindx_up=labelindx.copy()
        for li, l in enumerate(labelindx):
            if labelindx.count(l) > 1 and not l in alreadyseen and l>=0:
                for lii in range(li,S):
                    if labelindx[lii]==l:
                        labelindx_up[lii]=maxx
                        if l in alreadyseen:
                            positionsactive.remove(lii)
                        else:
                            alreadyseen.append(l)
                maxx+=1
        parents=labelindx_up
        parentstree.append(parents)
    parentstree=np.array(parentstree)
    np.savetxt("guidetree_"+ID+".txt",parentstree,fmt="%d")
    parentstree= np.loadtxt("guidetree_"+ID+".txt",dtype=int)
    logfile.write("guidetree_"+ID+".txt\n")
maxnode = parentstree[-1,0]+1
subindx = parentstree[0,:]
genotypes = np.array(np.reshape(np.array(list(np.random.uniform(0,2**64, L*2))*2*N),(N,L*4)), dtype=np.uint64)
recombination_marks = np.zeros((maxnode,n),dtype=np.int)

devgenotypes = cuda.to_device(genotypes)
devparentstree = cuda.to_device(parentstree)
devbinomcdf = cuda.to_device(binomcdf)
devrhocdf = cuda.to_device(rhocdf)
devrecombination_marks = cuda.to_device(recombination_marks)
rng_states = rnd.create_xoroshiro128p_states(N, seed=SEED)

start = timer()
evolve[blocks, threadsperblock](rng_states,0,1,devgenotypes,devbinomcdf,devparentstree,devrhocdf,devrecombination_marks,recomblen,recombmodel,recombpar)
evolve_time=timer()-start
if args.verbose:
    print("first iteration: "+str(evolve_time))
logfile.write("first iteration: "+str(evolve_time)+"\n")

start = timer()
for i in range(1,1001):
    evolve[blocks, threadsperblock](rng_states,i,1,devgenotypes,devbinomcdf,devparentstree,devrhocdf,devrecombination_marks,recomblen,recombmodel,recombpar)
cuda.synchronize()
evolve_time=timer()-start
if args.verbose:
    print("1000 more iterations: "+str(evolve_time))
logfile.write("1000 more iterations: "+str(evolve_time)+"\n")

start = timer()
for i in range(1001,NUMCYCLES):
    evolve[blocks, threadsperblock](rng_states,i,1,devgenotypes,devbinomcdf,devparentstree,devrhocdf,devrecombination_marks,recomblen,recombmodel,recombpar)
evolve_time=timer()-start
cuda.synchronize()
if args.verbose:
    print("remaining "+str(NUMCYCLES-1000)+" cylcles: "+str(evolve_time)+"\n")
logfile.write("remaining "+str(NUMCYCLES-1000)+" iterations: "+str(evolve_time)+"\n")
start = timer()
genotypes = devgenotypes.copy_to_host()
recombination_marks = devrecombination_marks.copy_to_host()
evolve_time=timer()-start
if args.verbose:
    print("copying of data to memory: "+str(evolve_time)+"\n")
logfile.write("copying of data to memory: "+str(evolve_time)+"\n")

outfile=open(ID+"/genotypes4Letters_"+ID+".pkl",'wb')
pkl.dump(subindx, outfile)
pkl.dump(N, outfile)
pkl.dump(L, outfile)
subset = [(i in subindx) for i in range(N) ]
subgenotypes = genotypes[subset,:(2*L)]
pkl.dump(subgenotypes, outfile)
pkl.dump(recombination_marks, outfile)
outfile.close()
logfile.close()
