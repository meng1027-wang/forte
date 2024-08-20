#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import psi4
import forte
import numpy as np
from itertools import combinations


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_wavefunction(data):
    Ca = data.psi_wfn.Ca().to_array()
    with open("coeff.json", "w+") as f:
        json.dump({"Ca": Ca}, f, cls=NumpyEncoder)


def read_wavefunction(data):
    with open("coeff.json") as f:
        coeff_data = json.load(f)
        C_read = coeff_data["Ca"]

    C_list = []
    for i in range(len(C_read)):
        if not C_read[i]:
            C_list.append(None)
        else:
            C_list.append(np.asarray(C_read[i]))

    if data.psi_wfn.nirrep() != 1:
        C_mat = psi4.core.Matrix.from_array(C_list)
    else:  # C1 no spatial symmetry, input is list(np.ndarray)
        C_mat = psi4.core.Matrix.from_array([np.asarray(C_list)])
    data.psi_wfn.Ca().copy(C_mat)
    data.psi_wfn.Cb().copy(C_mat)
def get_pairs_inac(nactel, nact):
    #获取活性空间内所有的电子对，inac:in active space
    # nact: 活性轨道个数； nactel: 活性电子个数
    if nactel <= nact:
        min_spin_e, max_spin_e = 0, nactel
    if nactel > nact:
        min_spin_e, max_spin_e = nactel - nact, nact
    pairs_e = []
    for na in range(min_spin_e, max_spin_e + 1):
        nb = nactel - na
        pairs_e.append((na, nb))
    return  pairs_e

def dets_inac(nactel, nact):
    #获取涉及活性空间内活性电子激发的所有行列式，inac:in active space
    pairs_e = get_pairs_inac(nactel, nact)
    actorbs = range(nact)
    dets = []
    dets_dict = {}
    for na, nb in pairs_e:
        s = 0.5 * (na - nb)
        type_dets_num = 0
        alfa_locs = list(combinations(actorbs, na))
        beta_locs = list(combinations(actorbs, nb))

        if na == 0:
            for beta_loc in beta_locs:      
                det = forte.Determinant()                          
#                 for m in range(ndocc):
#                     det.create_alfa_bit(m)
#                     det.create_beta_bit(m)    
                for i in beta_loc: 
                    det.create_beta_bit(i)
                dets.append(det) 
                type_dets_num += 1
        elif nb == 0:
            for alfa_loc in alfa_locs:                      
                det = forte.Determinant()                          
#                 for m in range(ndocc):
#                     det.create_alfa_bit(m)
#                     det.create_beta_bit(m) 
                for j in alfa_loc: 
                    det.create_alfa_bit(j) 
                dets.append(det) 
                type_dets_num += 1
        else:
            for alfa_loc in alfa_locs: 
                for beta_loc in beta_locs:                          
                    det = forte.Determinant()                          
#                     for m in range(ndocc):
#                         det.create_alfa_bit(m)
#                         det.create_beta_bit(m) 
                    for j in alfa_loc: 
                        det.create_alfa_bit(j)
                    for i in beta_loc:
                        det.create_beta_bit(i)                    
                    dets.append(det) 
                    type_dets_num += 1
        dets_dict[s] = type_dets_num
    return dets, dets_dict



def write_external_active_space_file(as_ints, state_map, mo_space_info, json_file="forte_ints.json"):
    ndocc = mo_space_info.size("INACTIVE_DOCC")

    for state, nroots in state_map.items():
        file = {}

        nmo = as_ints.nmo()

        file["state_symmetry"] = {"data": state.irrep(), "description": "Symmetry of the state"}

        file["na"] = {"data": state.na() - ndocc, "description": "number of alpha electrons in the active space"}

        file["nb"] = {"data": state.nb() - ndocc, "description": "number of beta electrons in the active space"}

        file["nso"] = {"data": 2 * nmo, "description": "number of active spin orbitals"}

        file["symmetry"] = {
            "data": [i for i in as_ints.mo_symmetry() for j in range(2)],
            "description": "symmetry of each spin orbital (Cotton ordering)",
        }

        file["spin"] = {
            "data": [j for i in range(nmo) for j in range(2)],
            "description": "spin of each spin orbital (0 = alpha, 1 = beta)",
        }

        scalar_energy = as_ints.frozen_core_energy() + as_ints.scalar_energy() + as_ints.nuclear_repulsion_energy()
        file["scalar_energy"] = {
            "data": scalar_energy,
            "description": "scalar energy (sum of nuclear repulsion, frozen core, and scalar contributions",
        }

        oei_a = [(i * 2, j * 2, as_ints.oei_a(i, j)) for i in range(nmo) for j in range(nmo)]
        oei_b = [(i * 2 + 1, j * 2 + 1, as_ints.oei_b(i, j)) for i in range(nmo) for j in range(nmo)]

        file["oei"] = {"data": oei_a + oei_b, "description": "one-electron integrals as a list of tuples (i,j,<i|h|j>)"}

        tei = []
        for i in range(nmo):
            for j in range(nmo):
                for k in range(nmo):
                    for l in range(nmo):
                        tei.append((i * 2, j * 2, k * 2, l * 2, as_ints.tei_aa(i, j, k, l)))  # aaaa
                        tei.append((i * 2, j * 2 + 1, k * 2, l * 2 + 1, +as_ints.tei_ab(i, j, k, l)))  # abab
                        tei.append((i * 2, j * 2 + 1, l * 2 + 1, k * 2, -as_ints.tei_ab(i, j, k, l)))  # abba
                        tei.append((j * 2 + 1, i * 2, k * 2, l * 2 + 1, -as_ints.tei_ab(i, j, k, l)))  # baab
                        tei.append((j * 2 + 1, i * 2, l * 2 + 1, k * 2, +as_ints.tei_ab(i, j, k, l)))  # baba
                        tei.append((i * 2 + 1, j * 2 + 1, k * 2 + 1, l * 2 + 1, as_ints.tei_bb(i, j, k, l)))  # bbbb

        file["tei"] = {
            "data": tei,
            "description": "antisymmetrized two-electron integrals as a list of tuples (i,j,k,l,<ij||kl>)",
        }
        print('-----------oei_a-----------------')
        oei_a, oei_b = np.zeros((nmo, nmo)), np.zeros((nmo, nmo))
        with open('oei_a.txt', 'w')as a:
            for i in range(nmo):
                for j in range(nmo):
                    print(i, j, as_ints.oei_a(i,j), file=a)
                    oei_a[i, j] = as_ints.oei_a(i,j)
        np.save('oei_a_psi4.npy', oei_a)
        

        print('-----------oei_b----------------')
        with open('oei_b.txt', 'w')as b:
            for i in range(nmo):
                for j in range(nmo):
                    print(i, j, as_ints.oei_b(i,j), file=b)
                    oei_b[i, j] = as_ints.oei_b(i,j)
        np.save('oei_b_psi4.npy', oei_b)


        tei_aa, tei_bb, tei_ab = np.zeros((nmo, nmo, nmo, nmo)), np.zeros((nmo, nmo, nmo, nmo)), np.zeros((nmo, nmo, nmo, nmo))

        print('------------tei_aa---------------')
        with open('tei_aa.txt', 'w')as c:
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            print(i, j, k, l, as_ints.tei_aa(i, j, k, l), file=c)
                            tei_aa[i,j,k,l] = as_ints.tei_aa(i,j,k,l)
        np.save('tei_aa_psi4.npy', tei_aa)
    
        print('------------tei_bb---------------')
        with open('tei_bb.txt', 'w')as d:
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            print(i, j, k, l, as_ints.tei_bb(i, j, k, l), file=d)
                            tei_bb[i,j,k,l] = as_ints.tei_bb(i,j,k,l)
        np.save('tei_bb_psi4.npy', tei_bb)

        print('------------tei_ab---------------')
        with open('tei_ab.txt', 'w')as e:
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            print(i, j, k, l, as_ints.tei_ab(i, j, k, l), file=e)
                            tei_ab[i,j,k,l] = as_ints.tei_ab(i,j,k,l)
        np.save('tei_ab_psi4.npy', tei_ab)

        dets = dets_inac(7, 4)[0]
        ndets = len(dets)
        H = np.ndarray((ndets,ndets))
        for I, detI in enumerate(dets):
            for J, detJ in enumerate(dets):
                H[I][J] = as_ints.slater_rules(detI,detJ)
        with open('H.txt', 'w')as g:
            print(H, file=g)
        
        eigval, eigvec = np.linalg.eigh(H)
        print((eigval[0] + scalar_energy))

        with open(json_file, "w+") as f:
            json.dump(file, f, sort_keys=True, indent=2)

        # make_hamiltonian(as_ints,state)


def make_hamiltonian(as_ints, state_map):
    import itertools

    for state, nroots in state_map.items():
        print(f"\nstate: {state}, nroots: {nroots}")
        dets = []

        na = state.na()
        nb = state.nb()

        orbs = [i for i in range(as_ints.nmo())]

        # generate all the alpha strings
        for astr in itertools.combinations(orbs, na):
            # generate all the beta strings
            for bstr in itertools.combinations(orbs, nb):
                d = forte.Determinant()
                for i in astr:
                    d.set_alfa_bit(i, True)
                for i in bstr:
                    d.set_beta_bit(i, True)
                dets.append(d)

        print(f"\n\n==> List of FCI determinants <==")
        for d in dets:
            print(f"{d.str(4)}")

        scalar_e = as_ints.scalar_energy() + as_ints.nuclear_repulsion_energy() + as_ints.frozen_core_energy()
        print(f"scalar_e = {scalar_e}")

        import numpy as np

        ndets = len(dets)
        H = np.ndarray((ndets, ndets))
        for I, detI in enumerate(dets):
            for J, detJ in enumerate(dets):
                H[I][J] = as_ints.slater_rules(detI, detJ)
                # if I == J:
                #     H[I][J] += scalar_e

        print(f"\n==> Active Space Hamiltonian <==\n")
        print(f"\n{H}")


def write_external_rdm_file(rdm, active_space_solver):
    g1a = rdm.g1a()
    g1b = rdm.g1b()

    g2aa = rdm.g2aa()
    g2ab = rdm.g2ab()
    g2bb = rdm.g2bb()

    nact = g1a.shape[0]

    gamma1_a = [(i * 2, j * 2, g1a[i][j]) for i in range(nact) for j in range(nact)]
    gamma1_b = [(i * 2 + 1, j * 2 + 1, g1b[i][j]) for i in range(nact) for j in range(nact)]

    file = {}

    file["nso"] = {"data": 2 * nact, "description": "number of active spin orbitals"}

    state_energies_map = active_space_solver.state_energies_map()
    #print('---------------maybe---wrong-------------')
    #print(state_energies_map)
    for state, energies in state_energies_map.items():
       # print('******************')
       # print(energies)
       # print(type(energies))
        file["energy"] = {"data": energies[0], "description": "energy"}

    file["gamma1"] = {
        "data": gamma1_a + gamma1_b,
        "description": "one-body density matrix as a list of tuples (i,j,<i^ j>)",
    }

    gamma2 = []
    for i in range(nact):
        for j in range(nact):
            for k in range(nact):
                for l in range(nact):
                    gamma2.append((i * 2, j * 2, k * 2, l * 2, g2aa[i, j, k, l]))  # aaaa
                    gamma2.append((i * 2, j * 2 + 1, k * 2, l * 2 + 1, +g2ab[i, j, k, l]))  # abab
                    gamma2.append((i * 2, j * 2 + 1, l * 2 + 1, k * 2, -g2ab[i, j, k, l]))  # abba
                    gamma2.append((j * 2 + 1, i * 2, k * 2, l * 2 + 1, -g2ab[i, j, k, l]))  # baab
                    gamma2.append((j * 2 + 1, i * 2, l * 2 + 1, k * 2, +g2ab[i, j, k, l]))  # baba
                    gamma2.append((i * 2 + 1, j * 2 + 1, k * 2 + 1, l * 2 + 1, g2bb[i, j, k, l]))  # bbbb

    file["gamma2"] = {
        "data": gamma2,
        "description": "two-body density matrix as a list of tuples (i,j,k,l,<i^ j^ l k>)",
    }
   # print('------test------')
   # print(rdm.max_rdm_level())
   # print('------test------')
    if rdm.max_rdm_level() == 3:
        g3aaa = rdm.g3aaa()
        g3aab = rdm.g3aab()
        g3abb = rdm.g3abb()
        g3bbb = rdm.g3bbb()
        gamma3 = []
        for i in range(nact):
            for j in range(nact):
                for k in range(nact):
                    for l in range(nact):
                        for m in range(nact):
                            for n in range(nact):
                                # aaa case
                                gamma3.append((i * 2, j * 2, k * 2, l * 2, m * 2, n * 2, g3aaa[i, j, k, l, m, n]))

                                # aab case
                                gamma3.append(
                                    (i * 2, j * 2, k * 2 + 1, l * 2, m * 2, n * 2 + 1, g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2, k * 2 + 1, l * 2, n * 2 + 1, m * 2, -g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2, k * 2 + 1, n * 2 + 1, l * 2, m * 2, g3aab[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (i * 2, k * 2 + 1, j * 2, l * 2, m * 2, n * 2 + 1, -g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, k * 2 + 1, j * 2, l * 2, n * 2 + 1, m * 2, g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, k * 2 + 1, j * 2, n * 2 + 1, l * 2, m * 2, -g3aab[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (k * 2 + 1, i * 2, j * 2, l * 2, m * 2, n * 2 + 1, g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (k * 2 + 1, i * 2, j * 2, l * 2, n * 2 + 1, m * 2, -g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (k * 2 + 1, i * 2, j * 2, n * 2 + 1, l * 2, m * 2, g3aab[i, j, k, l, m, n])
                                )

                                # abb case
                                gamma3.append(
                                    (i * 2, j * 2 + 1, k * 2 + 1, l * 2, m * 2 + 1, n * 2 + 1, g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2 + 1, k * 2 + 1, m * 2 + 1, l * 2, n * 2 + 1, -g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2 + 1, k * 2 + 1, m * 2 + 1, n * 2 + 1, l * 2, g3abb[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (j * 2 + 1, i * 2, k * 2 + 1, l * 2, m * 2 + 1, n * 2 + 1, -g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, i * 2, k * 2 + 1, m * 2 + 1, l * 2, n * 2 + 1, g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, i * 2, k * 2 + 1, m * 2 + 1, n * 2 + 1, l * 2, -g3abb[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (j * 2 + 1, k * 2 + 1, i * 2, l * 2, m * 2 + 1, n * 2 + 1, g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, k * 2 + 1, i * 2, m * 2 + 1, l * 2, n * 2 + 1, -g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, k * 2 + 1, i * 2, m * 2 + 1, n * 2 + 1, l * 2, g3abb[i, j, k, l, m, n])
                                )

                                # bbb case
                                gamma3.append(
                                    (
                                        i * 2 + 1,
                                        j * 2 + 1,
                                        k * 2 + 1,
                                        l * 2 + 1,
                                        m * 2 + 1,
                                        n * 2 + 1,
                                        g3bbb[i, j, k, l, m, n],
                                    )
                                )

        file["gamma3"] = {
            "data": gamma3,
            "description": "three-body density matrix as a list of tuples (i,j,k,l,m,n <i^ j^ k^ n m l>)",
        }

    with open("ref_rdms.json", "w+") as f:
        json.dump(file, f, sort_keys=True, indent=2)

def write_external_pdm_file(rdm, active_space_solver, indice):
    g1a = rdm.g1a()
    g1b = rdm.g1b()

    g2aa = rdm.g2aa()
    g2ab = rdm.g2ab()
    g2bb = rdm.g2bb()

    nact = g1a.shape[0]

    gamma1_a = [(i * 2, j * 2, g1a[i][j]) for i in range(nact) for j in range(nact)]
    gamma1_b = [(i * 2 + 1, j * 2 + 1, g1b[i][j]) for i in range(nact) for j in range(nact)]

    file = {}

    file["nso"] = {"data": 2 * nact, "description": "number of active spin orbitals"}

    state_energies_map = active_space_solver.state_energies_map()
   # print('---------------maybe---wrong-------------')
   # print(state_energies_map)
    for state, energies in state_energies_map.items():
       # print('******************')
       # print(energies)
       # print(type(energies))
        file["energy"] = {"data": energies[0], "description": "energy"}

    file["gamma1"] = {
        "data": gamma1_a + gamma1_b,
        "description": "one-body density matrix as a list of tuples (i,j,<i^ j>)",
    }

    gamma2 = []
    for i in range(nact):
        for j in range(nact):
            for k in range(nact):
                for l in range(nact):
                    gamma2.append((i * 2, j * 2, k * 2, l * 2, g2aa[i, j, k, l]))  # aaaa
                    gamma2.append((i * 2, j * 2 + 1, k * 2, l * 2 + 1, +g2ab[i, j, k, l]))  # abab
                    gamma2.append((i * 2, j * 2 + 1, l * 2 + 1, k * 2, -g2ab[i, j, k, l]))  # abba
                    gamma2.append((j * 2 + 1, i * 2, k * 2, l * 2 + 1, -g2ab[i, j, k, l]))  # baab
                    gamma2.append((j * 2 + 1, i * 2, l * 2 + 1, k * 2, +g2ab[i, j, k, l]))  # baba
                    gamma2.append((i * 2 + 1, j * 2 + 1, k * 2 + 1, l * 2 + 1, g2bb[i, j, k, l]))  # bbbb

    file["gamma2"] = {
        "data": gamma2,
        "description": "two-body density matrix as a list of tuples (i,j,k,l,<i^ j^ l k>)",
    }
   # print('------test------')
   # print(rdm.max_rdm_level())
   # print('------test------')
    if rdm.max_rdm_level() == 3:
        g3aaa = rdm.g3aaa()
        g3aab = rdm.g3aab()
        g3abb = rdm.g3abb()
        g3bbb = rdm.g3bbb()
        gamma3 = []
        for i in range(nact):
            for j in range(nact):
                for k in range(nact):
                    for l in range(nact):
                        for m in range(nact):
                            for n in range(nact):
                                # aaa case
                                gamma3.append((i * 2, j * 2, k * 2, l * 2, m * 2, n * 2, g3aaa[i, j, k, l, m, n]))

                                # aab case
                                gamma3.append(
                                    (i * 2, j * 2, k * 2 + 1, l * 2, m * 2, n * 2 + 1, g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2, k * 2 + 1, l * 2, n * 2 + 1, m * 2, -g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2, k * 2 + 1, n * 2 + 1, l * 2, m * 2, g3aab[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (i * 2, k * 2 + 1, j * 2, l * 2, m * 2, n * 2 + 1, -g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, k * 2 + 1, j * 2, l * 2, n * 2 + 1, m * 2, g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, k * 2 + 1, j * 2, n * 2 + 1, l * 2, m * 2, -g3aab[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (k * 2 + 1, i * 2, j * 2, l * 2, m * 2, n * 2 + 1, g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (k * 2 + 1, i * 2, j * 2, l * 2, n * 2 + 1, m * 2, -g3aab[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (k * 2 + 1, i * 2, j * 2, n * 2 + 1, l * 2, m * 2, g3aab[i, j, k, l, m, n])
                                )

                                # abb case
                                gamma3.append(
                                    (i * 2, j * 2 + 1, k * 2 + 1, l * 2, m * 2 + 1, n * 2 + 1, g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2 + 1, k * 2 + 1, m * 2 + 1, l * 2, n * 2 + 1, -g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (i * 2, j * 2 + 1, k * 2 + 1, m * 2 + 1, n * 2 + 1, l * 2, g3abb[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (j * 2 + 1, i * 2, k * 2 + 1, l * 2, m * 2 + 1, n * 2 + 1, -g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, i * 2, k * 2 + 1, m * 2 + 1, l * 2, n * 2 + 1, g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, i * 2, k * 2 + 1, m * 2 + 1, n * 2 + 1, l * 2, -g3abb[i, j, k, l, m, n])
                                )

                                gamma3.append(
                                    (j * 2 + 1, k * 2 + 1, i * 2, l * 2, m * 2 + 1, n * 2 + 1, g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, k * 2 + 1, i * 2, m * 2 + 1, l * 2, n * 2 + 1, -g3abb[i, j, k, l, m, n])
                                )
                                gamma3.append(
                                    (j * 2 + 1, k * 2 + 1, i * 2, m * 2 + 1, n * 2 + 1, l * 2, g3abb[i, j, k, l, m, n])
                                )

                                # bbb case
                                gamma3.append(
                                    (
                                        i * 2 + 1,
                                        j * 2 + 1,
                                        k * 2 + 1,
                                        l * 2 + 1,
                                        m * 2 + 1,
                                        n * 2 + 1,
                                        g3bbb[i, j, k, l, m, n],
                                    )
                                )

        file["gamma3"] = {
            "data": gamma3,
            "description": "three-body density matrix as a list of tuples (i,j,k,l,m,n <i^ j^ k^ n m l>)",
        }

    with open(f"{indice}_pdms.json", "w+") as f:
        json.dump(file, f, sort_keys=True, indent=2)

def read_external_active_space_file(as_ints, state_map):
    return False
