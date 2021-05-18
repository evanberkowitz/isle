import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import isle
import isle.drivers

with h5.File(sys.argv[1],"r") as h5f:
    rawP = h5f["correlation_functions/single_particle/destruction_creation/"][()]
    weights = h5f["weights/actVal"][()]

metadata = isle.h5io.readMetadata(sys.argv[1])
lattice = metadata[0]
parameters = metadata[1]
BETA = parameters.beta
U = parameters.U
NBS = 100
BSLENGTH=rawP.shape[0]
NCORR = rawP.shape[1]
NT=rawP.shape[3]
DELTA = BETA/NT

print(f'U={U}  beta={BETA}  Nt={NT}')
print(f'# of corelators = {NCORR}')
# Ok, this next part is related to simulations that have a sign problem, which yours do not have at the moment.
# But you can still evaluate this part anyways, as it won't affect your results.  I add it here anyways in 
# case we decide in the near future to run on systems with a sign problem.  
theta = -np.imag(weights)
allWeights = np.exp(1j*theta)

# now the number of weights equals the total number of configurations
# but the number of measurements is not necessarily the same!
# I need to take this into account

measFreq = int(weights.shape[0]/rawP.shape[0])
print("# measurements were done every {}th configuration (that was stored)".format(measFreq))

weights = np.array([allWeights[i] for i in range(0,allWeights.shape[0],measFreq)])
# now multiple correlators by their weights
for itr in range(rawP.shape[0]):
    rawP[itr] *= weights[itr]

# All data are stored in these arrays
corrRe = [[] for i in range(NCORR)]
errRe  = [[] for i in range(NCORR)]
corrIm = [[] for i in range(NCORR)]
errIm  = [[] for i in range(NCORR)]
tau = [ t*DELTA for t in range(NT)]


#bining the data
NBIN = 100
NBS = 100

for nc in range(NCORR):
    weightsBinned = []
    rawPBinned = []
    for i in range(int(rawP.shape[0]/NBIN)):
        weightsBinned.append(np.mean(np.array([weights[i*NBIN+j] for j in range(NBIN)])))
        temp = np.array([0+0j for _ in range(NT)])
        for j in range(NBIN):
            temp += rawP[i*NBIN+j,nc,nc]
        temp /= NBIN
        rawPBinned.append(temp)

    weightsBinned = np.array(weightsBinned)
    rawPBinned = np.array(rawPBinned)

    avgCorrP = np.mean(rawPBinned,axis=0)/np.mean(weightsBinned)
    
    BSLENGTH = weightsBinned.shape[0]
    bootstrapIndices = np.random.randint(0, BSLENGTH, [NBS, BSLENGTH])

    
    # now get error on correlators
    RerrPxx = np.std(np.real(np.array([np.mean(np.array([rawPBinned[cfg] for cfg in bootstrapIndices[sample]])/np.mean(np.array([weightsBinned[cfg] for cfg in bootstrapIndices[sample]])), axis=0) for sample in range(NBS) ] )), axis=0)

    IerrPxx = np.std(np.imag(np.array([np.mean(np.array([rawPBinned[cfg] for cfg in bootstrapIndices[sample]])/np.mean(np.array([weightsBinned[cfg] for cfg in bootstrapIndices[sample]])), axis=0) for sample in range(NBS) ] )), axis=0)
    
    for t in range(NT):
        corrRe[nc].append(avgCorrP[t].real)
        errRe[nc].append(RerrPxx[t])
        corrIm[nc].append(avgCorrP[t].imag)
        errIm[nc].append(IerrPxx[t])
        #print(t*DELTA,avgCorrP[t].real,RerrPxx[t],avgCorrP[t].imag,IerrPxx[t])


dataG = open('ExactFiles/U3B4G.dat','r').readlines()
dataM = open('ExactFiles/U3B4M.dat','r').readlines()
exact_t_gamma = []
exact_gamma_plus = []
exact_gamma_minus = []
exact_t_m = []
exact_m_plus = []
exact_m_minus = []
for dat in dataG:
    ss = dat.split()
    exact_t_gamma.append(float(ss[0]))
    exact_gamma_plus.append(float(ss[1]))
    exact_gamma_minus.append(float(ss[2]))
for dat in dataM:
    ss = dat.split()
    exact_t_m.append(float(ss[0]))
    exact_m_plus.append(float(ss[1]))
    exact_m_minus.append(float(ss[2]))

#Exact co-relators data 
# exactData = open("ExactFiles/U4B6.dat").readlines()
# exT = []
# exBonding = []
# exAntiBonding = []
# exAA = []
# exAB = []
# exBA = []
# exBB = []
# for i in range(len(exactData)):
#     split = exactData[i].split()
#     exT.append(float(split[0]))   # tau
#     exAntiBonding.append(float(split[1]))  # anti-bonding
#     exBonding.append(float(split[2]))      # bonding
#     exAA.append(float(split[3]))  # cAA
#     exAB.append(float(split[4]))  # cAB
#     exBA.append(float(split[5]))  # cBA
#     exBB.append(float(split[6]))  # cBB

# Now I just plot our the real parts of the correlators (the imaginary part should be MUCH smaller)
fig, ax = plt.subplots(1,1,figsize=(10,7.5))
ax.set_yscale('log')
for nc in range(NCORR):
    #print("error",nc,"=",errRe[nc])
    ax.errorbar(tau,corrRe[nc],yerr=errRe[nc],marker='o',label=f'corr{nc}')

# ax.plot(exT,exBonding,'k--',label='exact')
# ax.plot(exT,exAntiBonding,'k--')
ax.plot(exact_t_gamma,exact_gamma_plus,'k--')
ax.plot(exact_t_gamma,exact_gamma_minus,'k--')
ax.plot(exact_t_m,exact_m_plus,'r--')
ax.plot(exact_t_m,exact_m_minus,'r--')

ax.grid()
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$C(\tau)$')
ax.set_title(rf' Four sites $U={U}$  $\beta={BETA}$,  $N_t={NT}$')
ax.legend()
plt.savefig('results/Exact4sitesNmd3_logU{}B{}NT{}.pdf'.format(U,BETA,NT))






