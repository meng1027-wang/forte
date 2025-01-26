import numpy as np
from functools import reduce

def get_mo_casdm1(nact):
    casdm1_mo = np.zeros((nact, nact))
    with open('1rdm_mo', 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            index1, ndex2, value_str = line.lstrip().split()[1:]
            i, j, value = int(index1), int(ndex2), float(value_str)
            casdm1_mo[i, j] += value
    return casdm1_mo

def get_ao_rdm1(casdm1_mo, ncore, cas_list):
    mo_coeff = np.load('mo_coeff.npy')
    mocore = mo_coeff[:,:ncore] #ncore
    mocas = mo_coeff[:,cas_list] #cas_list
    dm1 = np.dot(mocore, mocore.conj().T) * 2
    dm1 = dm1 + reduce(np.dot, (mocas, casdm1_mo, mocas.conj().T))
    return dm1

casdm1_mo = get_mo_casdm1(4)
cas_list = [i for i in range(1,5)]
dm1 = get_ao_rdm1(casdm1_mo, 1, cas_list)


