import numpy as np 
import h5py as h5
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers


# if len(sys.argv) > 1:
#     #HMC data file 
#     HMC_data_file = str(sys.argv[1])
# else:
#     print("Please enter the h5 file")
#     sys.exit()

#parameters =list(map(float, input('Enter U,beta and Nt seperated by space: ').split(' ')))


HMC_data_file = "FoursiteU3B3Nt16.h5"
hf = h5.File(HMC_data_file, 'r')

LATTICE = "four_sites"
lat = isle.LATTICES[LATTICE]
Nt = 16 # number of time steps
lat.nt(Nt)



PARAMS = isle.util.parameters(
    beta=3.0,         # inverse temperature
    U=3.0,            # on-site coupling
    mu=0,           # chemical potential
    sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                    # (+1 only allowed for bipartite lattices)

    # Those three control which implementation of the action gets used.
    # The values given here are the defaults.
    # See documentation in docs/algorithm.
    hopping=isle.action.HFAHopping.EXP,
    basis=isle.action.HFABasis.PARTICLE_HOLE,
    algorithm=isle.action.HFAAlgorithm.DIRECT_SINGLE
)

def makeAction(lat, params):
    # Import everything this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lat)) \
        + isle.action.makeHubbardFermiAction(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping,
                                             params.basis,
                                             params.algorithm)
#calculates the action
action = makeAction(lat,PARAMS)


#loading the trajectory indexes from actual HMC run
traj_index = [int(index) for index in np.array((hf['configuration']))]


############intialization##################
training_actualHMC = np.zeros((len(traj_index),lat.lattSize()))+ 0j
gradient_actualHMC = np.zeros((len(traj_index),lat.lattSize()))+ 0j
action_HMC = np.zeros(len(traj_index)) + 0.j

# training data from Gaussian distribution
num_samples = 10000
training_gaus = np.zeros((num_samples,lat.lattSize())) + 0j
gradient_gaus = np.zeros((num_samples,lat.lattSize())) + 0j

#stores data from Actual HMC and Gaussian distributions
xx = np.zeros((len(traj_index)+ num_samples,lat.lattSize())) + 0j
yy = np.zeros((len(traj_index)+ num_samples,lat.lattSize())) + 0j

#loading actual HMC phi's and calculating the force
with h5.File(HMC_data_file, "r") as h5f:
    for i,index in  tqdm(enumerate(traj_index)):
        configuration, act = isle.h5io.loadConfiguration(h5f,index)
        training_actualHMC[i,:] = configuration
        action_HMC[i] = act
        gradient_actualHMC[i,:] = -action.force(isle.Vector(training_actualHMC[i,:]+0j))


# phi_r = isle.Vector(np.random.normal(0,
#                                  PARAMS.tilde("U", lat)**(1/2),
#                                  lat.lattSize())
#                       +0j)
# gradient_gaus = -action.force(phi_r)

#creating gaussian samples
for i in tqdm(range(num_samples)): 
    training_gaus[i,:] = np.random.normal(0,PARAMS.tilde("U", lat)**(1/2),lat.lattSize())
    gradient_gaus[i,:] = -action.force(isle.Vector(training_gaus[i,:]+0j))
    
# joining actual HMC and gaussian data
xx[:num_samples,:] = training_gaus[:num_samples,:]
xx[num_samples:,:] = training_actualHMC[:,:]
yy[:num_samples,:] = gradient_gaus[:num_samples,:]
yy[num_samples:,:] = gradient_actualHMC[:,:]

#saving the training_data
np.save('inputs_4sites_U3B3Nt16',xx)
np.save('targets_4sites_U3B3Nt16',yy)

#plotting histogram of training_data
plt.hist(np.real(xx).flatten(),density=True,bins=30,label='training_gaus + HMC')
plt.legend()
plt.savefig('trainingData_4sites_U3B3Nt16.pdf')