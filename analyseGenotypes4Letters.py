import numpy as np
import math
from numba import cuda, int64, jit
from timeit import default_timer as timer
import pickle as pkl
import sys

@jit
def transformAlignment(genotypes):
    N=int(genotypes.shape[0])
    L=int(genotypes.shape[1])
    alignment=[]
    for pos in range(0,L*64,2):
        posInt = pos//64
        posInInt = pos%64
        variant=[]
        for i in range(N):
            if int(genotypes[i,posInt]) & 2**posInInt == 0 and int(genotypes[i,posInt]) & 2**(posInInt+1) == 0:
                 variant.append("A")
            elif int(genotypes[i,posInt]) & 2**posInInt == 0 and int(genotypes[i,posInt]) & 2**(posInInt+1) > 0:
                 variant.append("C")
            elif int(genotypes[i,posInt]) & 2**posInInt > 0 and int(genotypes[i,posInt]) & 2**(posInInt+1) == 0:
                 variant.append("G")
            else:
                 variant.append("T")
        alignment.append(variant)
    return(alignment)

def writeAlignmentFile(alignment,subindx,filename,S):
    outfile=open(filename,'w')
    outfile.write(str(len(alignment[:][0]))+" "+str(len(alignment))+"\n")
    for i in range(S):
        r="".join([alignment[a][i] for a in range(len(alignment))])
        outfile.write("S"+str(subindx[i])+" "*(100-len("S"+str(subindx[i])))+r+"\n")
    outfile.close()

#@jit
def pairwiseClonal(recombination_marks,S,parentstree):
    marksarray=list()
    for i in range(S-1):
        for j in range(i+1,S):
            ip=i
            jp=j
            marks=np.zeros(recombination_marks.shape[1],dtype=np.int)
            ipOld=-2
            jpOld=-2
            for row in range(parentstree.shape[0]-1):
                ipindex=list(parentstree[row,:]).index(ip)
                jpindex=list(parentstree[row,:]).index(jp)
                ip=parentstree[row+1,ipindex]
                jp=parentstree[row+1,jpindex]
                if ip==jp:
                    break
                else:
                    if not ip==ipOld:
                        binmarksI=np.array([int(mark>0)for mark in recombination_marks[ip,:]])
                        marks=binmarksI|marks
                    if not jp==jpOld:
                        binmarksJ=np.array([int(mark>0)for mark in recombination_marks[jp,:]])
                        marks=binmarksJ|marks
                ipOld=ip
                jpOld=jp
            marksarray.append(1-sum(marks)/marks.shape[0])
    return(marksarray)

@jit
def pairwiseClonalQuant(recombination_marks,S,parentstree):
    marksarray=list()
    for i in range(S-1):
        for j in range(i+1,S):
            ip=i
            jp=j
            marks=np.zeros(recombination_marks.shape[1],dtype=np.int)
            ipOld=-2
            jpOld=-2
            for row in range(parentstree.shape[0]-1):
                ipindex=list(parentstree[row,:]).index(ip)
                jpindex=list(parentstree[row,:]).index(jp)
                ip=parentstree[row+1,ipindex]
                jp=parentstree[row+1,jpindex]
                if ip==jp:
                    break
                else:
                    if not ip==ipOld:
                        marks=recombination_marks[ip,:]+marks
                    if not jp==jpOld:
                        marks=recombination_marks[jp,:]+marks
                ipOld=ip
                jpOld=jp
            marksarray.append(marks)
    return(np.array(marksarray))


ID=sys.argv[1]

infile=open(ID+"/genotypes4Letters_"+ID+".pkl",'rb')
subindx=pkl.load(infile)
N=pkl.load(infile)
L=pkl.load(infile)
subgenotypes=pkl.load(infile)
recombination_marks=pkl.load(infile)
infile.close()
logfile=open(ID+"/"+ID+".log",'r')
logtable=logfile.readlines()
guidetreefilen=logtable[10].strip()
logfile.close()

rhofile=open('rhofile.txt','a')
rhofile.write(ID+' '+logtable[5].strip()+"\n")

S=int(subgenotypes.shape[0])
parentstree=np.loadtxt(guidetreefilen,dtype=int)

np.savetxt(ID+"/reco_"+ID+".txt", recombination_marks,fmt='%d', delimiter=',')
np.savetxt(ID+"/recoColSum_"+ID+".txt", np.sum(recombination_marks,axis=0)[None],fmt='%d', delimiter=',')

marksarray=pairwiseClonalQuant(recombination_marks,S,parentstree)
print("pairwiseClonalQuant done")
np.savetxt(ID+"/recoQuant_"+ID+".txt",marksarray,fmt='%d', delimiter=',')

marksarray=pairwiseClonal(recombination_marks,S,parentstree)
outfile=open(ID+"/clonal_"+ID+".txt",'w')
ind=0
for i in range(S-1):
    for j in range(i+1,S):
        outfile.write("S"+str(i)+" S"+str(j)+" "+str(marksarray[ind])+"\n")
        ind+=1
outfile.close()

alignment=transformAlignment(subgenotypes)
writeAlignmentFile(alignment,subindx,ID+"/sim_"+ID+".phy",S)

