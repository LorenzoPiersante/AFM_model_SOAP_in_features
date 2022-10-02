import torch
import numpy as np
import ase
import dscribe
from dscribe.descriptors import SOAP

# species class definition
chem_dict = {
    0: "H",
    1: "C",
    2: "N",
    3: "O",
    4: "F"
}
# number of classes
num_classes = len(chem_dict)

# SOAP hyperparameters
species_in_mol = {"H", "C", "N", "O", "F"}
r_cut = 4
n_max = 3
l_max = 2
SOAP_size = (l_max+1)*(n_max)**2

# initilaise SOAP object
soap = SOAP(
    species=species_in_mol,
    periodic=False,
    rcut=r_cut,
    nmax=n_max,
    lmax=l_max
)


# SOAP output is ndarray of shape (num_atoms, num_SOAP_entries)

# SELF-SOAP RESHAPING FUNCTION
# In the same-species SOAP, the soap.create() method generates a vector with only non-repeated elements,
# thus we need a function to reshape it into the complete symmetric power vector
# The general SOAP for a given chemical env and l-component is arrange as:
# for n from 1 to n_max
#   for n' from 1 to n_max
# if the chemical environment involves like species, symmetric components are stored once only
# so taking a (H, H) env the components will be 1-1, 1-2, 1-3, 2-2, 2-3, 3-3
# (this is the output of soap.create() where soap = SOAP object of dscribe library of molecular descriptors)
# the target vector is then:
# 1-1, 1-2, 1-3, 1-4, 1-2, 2-2, 2-3, 1-3, 2-3, 3-3
# this will ensure that the label SOAP power vectors all have the same size

# reshape SS vector from (num_atoms, num_SS_entries) to (num_atoms, num_IS_entries) adding the
# symmetric components as explained above
def SSOAP_reshaping(SS_vector):
    '''
    Casts the self-SOAP vector to the same shape of the inter-SOAP vector.
    It adds the symmetric components that are otherwise missing.

    Arg:
    - SS_vector: ndarray of dim(N, 16) (for n_max = 3 and l_max = 2) where N is the number of atoms in the
    molecule or in the batch of molecule, for a molecule composed of a single atom N = 1, the input must
    always be a 2D array.
    '''
    num_atoms, SOAP_entries = np.shape(SS_vector)
    reshape_tuple = (num_atoms, 1)
    axis = 1

    # split the vector into the l components
    l_components = np.split(SS_vector, int(l_max + 1), axis=axis)

    # identify the n-value split locations
    split_locations = []
    loc = 0
    for n in reversed(range(1, n_max + 1)):
        loc += n
        split_locations.append(loc)

    # reshape all the components one at a time
    for l in range(l_max + 1):
        # divide the l-component according to n
        current_l = np.split(l_components[l], split_locations, axis=axis)
        for i in range(1, n_max):
            for j in reversed(range(i)):
                # add the missing nn' values to the given n
                current_l[i] = np.concatenate((current_l[j][:, i].reshape(reshape_tuple), current_l[i]), axis=axis)

        # reform the l-component
        reshaped_l_component = np.concatenate(current_l, axis=axis)
        # add it to original list
        l_components[l] = reshaped_l_component

    # generate the new SS_vector
    new_SS_vector = np.concatenate(l_components, axis=axis)

    return new_SS_vector


# CHEM ENV MANAGER CLASS

# atom dictionary
atomic_dict = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9
}

# reverse atom dictionary
reverse_atomic_dict = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F"
}


class env_manager():
    '''
    Initialisation args:
    - species: set of all the atomic sts(symbols) in the molecule.
    '''

    def __init__(self, species):

        # This function sorts the input species according to atomic number
        def sorter(species):
            species_list = [atomic_dict[j] for j in species]  # list of atomic numbers as given by the input
            species_list.sort()  # sort them
            sorted_species_list = [reverse_atomic_dict[j] for j in species_list]  # convert back to symbols

            return sorted_species_list

        self.species = sorter(species)

    # This function creates a list of environment tuples according to the standard SOAP ordering
    def env_tuples(self):

        envs = []

        n = 0
        for i in self.species:
            a = i

            for j in range(n, len(self.species)):
                b = self.species[j]
                SOAP_env = (a, b)
                envs.append(SOAP_env)

            n += 1

        return envs


# define chem environments and chem environment dictionaries
# define env tuples (H, H), (H, C) ...
env_tuples = env_manager(species_in_mol).env_tuples()
# define SOAP dictionary env:slice in SOAP vector
SOAP_dict = {env: soap.get_location(env) for env in env_tuples}


def reshape_SOAP_array(SOAP_feature):
    '''
    This function divides up the SOAP array according to the different chem envs, reshapes the SS and
    then stacks the diffrent chem env SOAP vectors in an array of dim(num_env, N, 27).

    Args:
    - SOAP_feature: nd array of dimension dim(N, SOAP_vector_size) output of soap.create.
    N can either be the number of atoms in a molecule, or the total number of nodes in the batch (in this
    case the soap.create output is concatenated along dimension 0 before being input to the function,
    the concatenation step is skipped when the batch contains a single molecule).
    '''

    reshaped_SOAP_feat = []
    for env in env_tuples:
        SOAP_env_entries = SOAP_feature[:, SOAP_dict[env]]
        # reshape in case of self environment
        if env[0] == env[1]:
            SOAP_env_entries = SSOAP_reshaping(SOAP_env_entries)
        # append to list
        reshaped_SOAP_feat.append(SOAP_env_entries)

    # stack SOAP_env_entries along new 0 dimension
    reshaped_SOAP_feat = np.stack(reshaped_SOAP_feat, axis=0)

    return reshaped_SOAP_feat


# this function can totally accommodate an empty input

# GRAPH encoding input

# this part of the code must run on cpu since ase molecules require numpy arrays and because SOAP is
# optimised for parallel processing across multiple workers
def add_SOAP2node_input_Genc(node_inputs, Ns, SOAP_comp_size, device):
    '''
    This function calculates the SOAP descriptor for each input molecule in the batch and generates the
    input for the MPNN used for the graph encoding.

    Args:
    - node_inputs: tensor of dim(N, 3+num_classes), where 3 is the node position and num_classes is the
    number of different chemical species/classes that are present in the molecules.
    - Ns: list of number of atoms per molecule in the batch.
    - SOAP_comp_size: int, number of entries in SOAP vector chem env component.

    N can either be the number of atoms in a molecule or the total number of nodes in a batch made of
    multiple molecule.
    For a molecule with one atom only, N = 1, the array must always be 2D.

    len(Ns) gives the number of atoms in the molecule.

    Returns.
    - new_node inputs: tensor of dim(num_SOAP_environments, N, 27+3+num_classes) where for the case of 5 different
    classes is dim(15, N, 35) where 27 is the SOAP vector for the chem environment of the given node, 3 is the position
    and 5 is the one-hot class encoding of the node.
    - class_chunks: list of dtype=torch.int tensors which indicate the classes of each node in the molecules
    in the batch, tensor dim is dim(N_i), where i is a molecule in the batch.
    - position_chunks: list of dtype=torch.float tensors containing the node positions of each molecule
    in the batch, tensor dim is dim(N_i, 3), where i is a molecule in the batch.

    The chemical environments are indexed according to the standard SOAP ordering
    (see: https://singroup.github.io/dscribe/1.0.x/tutorials/descriptors/soap.html)
    and the size of the chem env SOAP vector is determined by the hyperparameters n_max and
    l_max (n_max = 3, l_max = 2 in this case).

    The tensors in class_chunks and position_chunks are stored in cpu since the SOAP creator object is optimised
    for parallel processing in CPUs.
    '''

    if len(node_inputs) > 1:
        # convert one-hot vector part of node_inputs into atomic class
        # create 1D input tensor of integer class labels for each node
        class_weight_batch = node_inputs[:, 3:]
        atomic_classes_inputs = torch.tensor([torch.argmax(one_hot).item() for one_hot in class_weight_batch],
                                             dtype=torch.int)

        # create positions input tensor for each node
        position_inputs = node_inputs[:, 0:3]

        # transfer to CPU
        atomic_classes_inputs = atomic_classes_inputs.to("cpu")
        position_inputs = position_inputs.to("cpu")

        # divide tensor into chunks corresponding to sub-graphs
        class_chunks = list(torch.split(atomic_classes_inputs, Ns, dim=0))
        position_chunks = list(torch.split(position_inputs, Ns, dim=0))

        # define a list of ase molecule objects
        mol_samples = [ase.Atoms("".join([chem_dict[l.item()] for l in c]), np.array(p)) for c, p in
                       zip(class_chunks, position_chunks)]

        # calculate SOAP
        SOAP_feature = soap.create(mol_samples, n_jobs=8)
        # if batch contains multiple molecules, concatenate the SOAP arrays,
        # otherwise the input is ready to be further processed
        if len(Ns) == 1:
            pass
        else:
            SOAP_feature = np.concatenate(SOAP_feature, axis=0)

        # cast the SOAP array to the shape dim(num_env, N, 27)
        reshaped_SOAP_feat = reshape_SOAP_array(SOAP_feature)

        # convert to tensor
        reshaped_SOAP_feat = torch.tensor(reshaped_SOAP_feat, requires_grad=False, dtype=torch.float, device=device)

        # new_nodes_input
        new_nodes_input = torch.zeros(
            (reshaped_SOAP_feat.size(0), reshaped_SOAP_feat.size(1), reshaped_SOAP_feat.size(2) + 3 + num_classes),
            requires_grad=False, dtype=torch.float, device=device)
        # assign entries
        new_nodes_input[:, :, 0:reshaped_SOAP_feat.size(2)] = reshaped_SOAP_feat
        new_nodes_input[:, :, reshaped_SOAP_feat.size(2):] = node_inputs

    else:
        new_nodes_input = torch.empty((len(env_tuples), 0, SOAP_comp_size + 3 + num_classes), requires_grad=False,
                                      dtype=torch.float, device=device)
        class_chunks = [torch.empty((0), requires_grad=False, dtype=torch.int, device="cpu") for n in Ns]
        position_chunks = [torch.empty((0), requires_grad=False, dtype=torch.float, device="cpu") for n in Ns]

    return new_nodes_input, class_chunks, position_chunks


# This function will operate only when there are other atoms in the graph - see workflow in complete model
# After all in order to have an edge connection you need at least two atoms in the subgraph

# EDGE identification encoding input

# must modify the above function to include the information coming from the classification of the new nodes
# and to generate the SOAP_node_inputs for the new nodes
def generate_Eenc_input(class_chunks, position_chunks, node_inputs, new_nodes, device):
    '''
    This function updates the node_inputs accounting for the newly classified node, and creates
    the new_nodes input for the edge classification block of the model.

    Args:
    - class_chunks and position_chunks: outputs of add_SOAP2node_input_Genc(), they are lists of tensors
    the former one contains tensors of dtype=torch.int which indicate the classes of each node in the
    molecules in the batch, the latter one contains tensors of dtype=torch.float that store the node
    positions of each molecule in the batch.
    - node_inputs: tensor of dim(N, 3+num_classes), where 3 is the node position and num_classes is the
    number of different chemical species/classes that are present in the molecules.
    - new_nodes: tensor of dim(N_new, 3+num_classes), where N_new == number of molecules in the batch,
    the dim=1 entries are the same as node_inputs.

    N_new = 1 when there is a single molecule in the batch.

    Returns:
    - nodes_input2: tensor of dim(num_SOAP_environments, N, 27+3+num_classes), where the SOAP entries have
    been calculated accounting for the classification of the new node.
    - new_nodes2: tensor of dim(num_SOAP_environments, N_new, 27+3+num_classes), where the SOAP entries are
    the SOAP vectors for the new nodes.
    '''
    new_class_weights = new_nodes[:, 3:]
    new_positions = new_nodes[:, 0:3]

    # transfer to CPU
    new_class_weights = new_class_weights.to("cpu")
    new_positions = new_positions.to("cpu")

    # turn weights into atomic classes
    new_atomic_classes = torch.tensor([torch.argmax(one_hot).item() for one_hot in new_class_weights], dtype=torch.int)

    # append new nodes to chunks
    for m, n in enumerate(zip(new_atomic_classes, new_positions)):
        n_c, n_p = n
        extended_class = torch.cat((class_chunks[m], torch.unsqueeze(n_c, dim=0)), dim=0)
        extended_class = extended_class.detach()
        class_chunks[m] = extended_class
        extended_position = torch.cat((position_chunks[m], torch.unsqueeze(n_p, dim=0)), dim=0)
        extended_position = extended_position.detach()
        position_chunks[m] = extended_position

    # define a list of ase molecule objects
    mol_samples = [ase.Atoms("".join([chem_dict[l.item()] for l in c]), np.array(p)) for c, p in
                   zip(class_chunks, position_chunks)]

    # calculate SOAP
    SOAP_features = soap.create(mol_samples, n_jobs=4)

    # when there is a single molecule in the batch, turn the output above into a len = 1 list
    # in order to execute correctly the operations in the for loop
    if new_nodes.size(0) == 1:
        SOAP_features = [SOAP_features]

    # divide the ndarrays into the previously classified nodes and the new nodes SOAP vectors
    SOAP_features_N_1 = []
    SOAP_features_N = []
    for s in SOAP_features:
        SOAP_feat_N_1 = s[0:-1, :]
        SOAP_feat_N = s[-1, :]
        SOAP_features_N_1.append(SOAP_feat_N_1)
        SOAP_features_N.append(SOAP_feat_N)
    # stack them along axis = 0
    SOAP_features_N_1 = np.vstack(SOAP_features_N_1)
    SOAP_features_N = np.vstack(SOAP_features_N)

    # reshape them to dim(num_env, N or N_new, 27)
    reshaped_SOAP_features_N_1 = reshape_SOAP_array(SOAP_features_N_1)
    reshaped_SOAP_features_N = reshape_SOAP_array(SOAP_features_N)

    # convert to tensors, create output and assign entries
    # N-1
    reshaped_SOAP_features_N_1 = torch.tensor(reshaped_SOAP_features_N_1, requires_grad=False, dtype=torch.float,
                                              device=device)
    nodes_input2 = torch.zeros((reshaped_SOAP_features_N_1.size(0), reshaped_SOAP_features_N_1.size(1),
                                reshaped_SOAP_features_N_1.size(2) + 3 + num_classes),
                               requires_grad=False, dtype=torch.float, device=device)

    nodes_input2[:, :, 0:reshaped_SOAP_features_N_1.size(2)] = reshaped_SOAP_features_N_1
    nodes_input2[:, :, reshaped_SOAP_features_N_1.size(2):] = node_inputs
    # N
    reshaped_SOAP_features_N = torch.tensor(reshaped_SOAP_features_N, requires_grad=False, dtype=torch.float,
                                            device=device)
    new_nodes2 = torch.zeros((reshaped_SOAP_features_N.size(0), reshaped_SOAP_features_N.size(1),
                              reshaped_SOAP_features_N.size(2) + 3 + num_classes),
                             requires_grad=False, dtype=torch.float, device=device)

    new_nodes2[:, :, 0:reshaped_SOAP_features_N.size(2)] = reshaped_SOAP_features_N
    new_nodes2[:, :, reshaped_SOAP_features_N.size(2):] = new_nodes

    return nodes_input2, new_nodes2

