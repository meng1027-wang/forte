import numpy as np
from functools import reduce
import re, copy
from pyscf import gto, scf
from pyscf.scf import jk
from pyscf.data import nist


c2 = nist.LIGHT_SPEED ** 2
#mo_coeff = np.load('mo_coeff.npy') # psi4 SA-CASSCF

# construct molecule via Pyscf
def make_mol(geom, basis, charge=0, symm=False, s2=0, unit='angstrom'):
    mol = gto.Mole()
    mol.atom = geom
    mol.symmetry = symm
    mol.spin = s2
    mol.charge = charge
    mol.verbose = 6
    mol.basis = basis
    mol.unit = unit
    mol.build()
    return mol

def run_scf(mol, e_cov=1e-6, g_cov=1e-6, autoaux_basis=False):
    if autoaux_basis == False:
        mf = scf.RHF(mol).density_fit().sfx2c1e().run(conv_tol=e_cov, conv_grad_tol=g_cov)
    else:
        autoaux_basis = autoaux_basis
        mf = scf.RHF(mol).density_fit(autoaux=autoaux_basis).sfx2c1e().run(conv_tol=e_cov, conv_grad_tol=g_cov)
    return mf

def h1_soc_ao(mol):
    mat = 0
    for atm_id in range(mol.natm):
        mol.set_rinv_orig(mol.atom_coord(atm_id))
        chg = mol.atom_charge(atm_id)
        mat -= chg * mol.intor('int1e_prinvxp_sph')  # right
    h1_xyz_ao = mat * 1 / (2 * c2) * (1j)  # right
    return h1_xyz_ao

def h2_somf_soc_ao(mol, rdm1, amfi=True): #rdm1: origin sa-casscf via Psi4; need to rotate to pyscf ao-order
    if amfi == False:
        h2_1 = jk.get_jk(mol, rdm1, 'ijkl,kl->ij', intor='int2e_p1vxp1', aosym='s1',comp=3)
        h2_2 = jk.get_jk(mol, rdm1, 'ijkl,jk->il', intor='int2e_p1vxp1', aosym='s1',comp=3)
        h2_3 = jk.get_jk(mol, rdm1, 'ijkl,li->kj', intor='int2e_p1vxp1', aosym='s1',comp=3)
        h2_ao = (h2_1 - 1.5 * h2_2 - 1.5 * h2_3) * 1 / (2 * c2) * (1j)
    else: # amfi == True:
        ao_loc = mol.ao_loc_nr()
        nao = ao_loc[-1]
        amfi_h2_1 = np.zeros((3, nao, nao),dtype=complex)
        amfi_h2_2 = np.zeros((3, nao, nao),dtype=complex)
        atom = copy.copy(mol)
        aoslice = mol.aoslice_by_atom(ao_loc)
        for ia in range(mol.natm):
            b0, b1, p0, p1 = aoslice[ia]
            atom._bas = mol._bas[b0:b1]
            atm_h2_1 = jk.get_jk(atom, rdm1[p0:p1, p0:p1], 'ijkl,kl->ij', intor='int2e_p1vxp1', aosym='s1',comp=3)
            atm_h2_2 = jk.get_jk(atom, rdm1[p0:p1, p0:p1], 'ijkl,jk->il', intor='int2e_p1vxp1', aosym='s1',comp=3)
            atm_h2_3 = jk.get_jk(atom, rdm1[p0:p1, p0:p1], 'ijkl,li->kj', intor='int2e_p1vxp1', aosym='s1',comp=3)
            amfi_h2_1[:, p0:p1, p0:p1] = atm_h2_1
            amfi_h2_2[:, p0:p1, p0:p1] = (atm_h2_2 + atm_h2_3)
        h2_ao = (amfi_h2_1 - 1.5 * amfi_h2_2) * 1 / (2 * c2) * (1j)
    return h2_ao

def f_soc_mo(h1_soc_ao, h2_somf_soc_ao, order_pyscf, order_psi4, mo_coeff, cas_list):
    h_soc_ao_pyscf = h1_soc_ao + h2_somf_soc_ao
    h_soc_ao_psi4 = trans_matrix_to_psi4_order(order_pyscf, order_psi4, h_soc_ao_pyscf)
    f_soc_mo_psi4 = np.einsum('rij,ip,jq->rpq', h_soc_ao_psi4, mo_coeff[:, cas_list], mo_coeff[:, cas_list])
    return f_soc_mo_psi4

def get_ao_order_psi4(filename): #修改了psi4的c++代码获得
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
    return lines

dic_am = {'s':'s+0',
          'py':'p-1', 'px':'p+1','pz':'p+0',
          'dxy':'d-2', 'dyz':'d-1', 'dz^2':'d+0','dxz':'d+1', 'dx2-y2':'d+2'} # forte内部的代码已证实

def replace_keys_with_values(string, dic):
    # 定义一个正则表达式模式，用于匹配字典中的键
    pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))

    # 使用替换函数
    def replace_match(match):
        return dic[match.group(0)]

    # 对每个字符串进行替换
    replaced_strings = pattern.sub(replace_match, string)
    return replaced_strings

def get_ao_order_pyscf(mol):
    ao_labels = mol.ao_labels() # a list
    new_ao_labels = []
    for label in ao_labels:
        label = label.rstrip()
        parts = label.split()
        parts[1] = parts[1].upper()
        parts[2] = replace_keys_with_values(parts[2], dic_am)
        new_label = ' '.join(parts) #到这一步是对的
        new_ao_labels.append(new_label)
    return new_ao_labels

def get_trans_matrix_pyscf_to_psi4(pyscf_ao_order, psi4_ao_order):
    size = len(pyscf_ao_order)
    # 构建一个字典来快速查找 psi4_ao_order 中的索引
    order_map = {val: j for j, val in enumerate(psi4_ao_order)}
    # P:permutation_matrix
    P = np.zeros((size, size))
    # 使用 order_map 获取 psi4_ao_order 中的索引以构建 P
    for i, val in enumerate(pyscf_ao_order):
        j = order_map[val]
        P[j, i] = 1
    return P

def trans_matrix_to_psi4_order(pyscf_ao_order, psi4_ao_order, pyscf_ao_matrix):
    P = get_trans_matrix_pyscf_to_psi4(pyscf_ao_order, psi4_ao_order)
    new_matrix = np.zeros_like(pyscf_ao_matrix)
    for i in range(3):
        new_matrix[i, :, :] = P @ pyscf_ao_matrix[i, :, :] @ P.T #@==np.dot
    return new_matrix

# def get_coeff_sa_casscf_psi4():
#     # 首先将mo_coeff保存到npy文件，在input.dat里添加的代码如下：
#     # ca = wfn.Ca_subset('AO', 'ALL').to_array()
#     # np.save('mo_coeff.npy', ca)
#     # TODO:如果可以最好不存到文件，直接能自己从wfn直接获得 -> 如果soc的py代码直接能通过一个关键词加入，那么pymodule.py里说不定可以直接加
#     # 读取npy文件
#     coeff_psi4 = np.load('mo_coeff.npy') #shape.[nmo.nmo] nmo=nao
#     # TODO：确认一下，mo的顺序，是否是纯按分子轨道能量排序，和irrep无关，psi4的hf是这样，待确认forte
#     return coeff_psi4

# def get_1rdm_mo_sa_casscf_psi4(): #only for active space
#     mo_1rdm = np.load('sa-casscf-1rdm.npy')
#     #好像是从dsrg计算可以获得这个的npy文件，是不是计算新的forte使用这个比较有利，新的forte在dsrg计算自动计算casscf
#     #shape:[nact, nact]
#     return mo_1rdm

# def get_moeff_active_forte(actorbs_indices, Ca_name='mo_coeff.npy'):
#     ca = np.load(Ca_name) # mo-coeff for all orbitals in forte
#     ca_active = ca[:, actorbs_indices]
#     return ca_active

def get_1rdm_ao_pyscf(P, ao_rdm1_psi4): # pyscf format
    """get a matrix of 1rdm in pyscf AO order, which is transformed from the 1rdm of AO 1rdm about psi4.
    the returned value will be give to the function "h2_somf_soc_ao" as the parameter rdm1

    Args:
        P (np.array): trans_ao_order_from_pyscf_to_psi4_matrix
        ao_rdm1_psi4: the AO 1-particle rdm in psi4 AO order
    """
    ao_1rdm_pyscf = P.T @ ao_rdm1_psi4 @ P
    return ao_1rdm_pyscf


def get_mo_casdm1(nact):
    casdm1_mo = np.zeros((nact, nact))
    with open('1rdm_mo', 'r') as f:
        lines = f.readlines()[3:]
        for line in lines:
            index1, index2, value_str = line.lstrip().split()[1:]
            i, j, value = int(index1), int(index2), float(value_str)
            casdm1_mo[i, j] += value
    return casdm1_mo

def get_ao_rdm1(ncore, nact, mo_coeff, cas_list): # psi4 format
    casdm1_mo = get_mo_casdm1(nact)
    # mo_coeff = np.load('mo_coeff.npy') #获得sa-casscf的分子轨道系数在许多函数下都用到，可以写成外部变量
    mocore = mo_coeff[:,:ncore] #ncore
    mocas = mo_coeff[:,cas_list] #cas_list
    dm1 = np.dot(mocore, mocore.conj().T) * 2
    dm1 = dm1 + reduce(np.dot, (mocas, casdm1_mo, mocas.conj().T))
    return dm1

def get_substates_energies(): # passed
    with open('state_energy.txt', 'r')as file:
        lines = file.readlines()
        substates = []
        energies = []
        for line in lines:
            last_space_index = line.rfind(" ")
            # 将最后一个空格前的内容和最后的小数分开
            substate = line[:last_space_index]
            energy = float(line[last_space_index + 1:])
            substates.append(substate)
            energies.append(energy)
    return substates, energies























