from pyscf import gto, scf, mcscf, symm
from pyscf.data import nist
import numpy as np
import json
import psi4
import forte.utils
import copy
from collections import Counter
np.set_printoptions(threshold=np.inf)
from itertools import combinations
from scipy.sparse.linalg import eigsh

c2 = nist.LIGHT_SPEED ** 2

def get_mol(geom, basis_set, spin, charge, symmetry, unit): #是否给其他信息，例如symmetry, point_group
    mol = gto.M(atom=geom,            
                symmetry=symmetry, #point group or False -> point group是否不适合于casci，可能导致某些自旋多重度的不可约下root为1，fcisiso在state_average_时错           
                basis=basis_set, spin=spin, charge=charge, unit=unit)
    return mol

def get_hf(geom, basis_set, spin, charge, symmetry, unit):
    mol = get_mol(geom, basis_set, spin, charge, symmetry, unit)
    mf = scf.RHF(mol).sfx2c1e().run() if (mol.nelectron) % 2 == 0 else scf.ROHF(mol).sfx2c1e().run()#rhf或者rohf是否也要选择呢,可以写成reference
    return mol, mf
    
def h1_soc_ao(mol): 
    mat = 0
    for atm_id in range(mol.natm):
        mol.set_rinv_orig(mol.atom_coord(atm_id))
        chg = mol.atom_charge(atm_id)
        mat -= chg * mol.intor('int1e_prinvxp_sph')  # right
    # print(mat)
    h1_xyz_ao = mat * 1 / (2 * c2) * (1j)  # right
    #h1_xyz_mo = np.einsum('xpq, pi, qj -> xij', h1_xyz_ao, ca, ca) 
    return h1_xyz_ao

def h2_soc_ao(mol): 
    h2_xyz_1fold = mol.intor('int2e_p1vxp1', 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao) # right
    # print(h2_xyz_1fold)
    h2_xyz_1fold = h2_xyz_1fold * 1 / (2 * c2) * (1j) 
    return h2_xyz_1fold

def get_soc_fpq_pyscf(nact, ndocc, nactel, mol=None, myhf=None, geom=None, basis_set=None, spin=None, charge=None, symmetry=None, unit=None,   cas_mos=None, ca=None, ao_rdm1=None, amfi=False):
    # nact: number of active orbitals; 
    # ndocc: number of core orbitals plus to number of restricted_docc orbitals
    # nactel: number of active electrons, or such a tuple:(nelec_alpha, nelec_beta)
    # cas_mos: a list contains selected active orbitals, which the orbital indices start with 1 not 0, for example [3,6];
    #          which is passed to mycas by "sort_mo()"
    
    # ca和ao_rdm1一般一起给 # 是sa-casscf的
    if mol == None and myhf == None:
        mol, myhf = get_hf(geom, basis_set, spin, charge, symmetry, unit)
    
    h1_ao_soc = h1_soc_ao(mol)
    # start --待修改完善
    if ao_rdm1 is None: 
        ca = myhf.mo_coeff
        mycas = mcscf.CASCI(myhf, nact, nactel) #casci还是casscf要做区别--> TODO
        if cas_mos is not None:
            ca = mycas.sort_mo(cas_mos)
    # "sort_mo" returns: An reoreded mo_coeff, which put the orbitals given by caslst in the CAS space 
    # https://pyscf.org/_modules/pyscf/mcscf/addons.html#sort_mo
    # 应该就是把挑选出的活性轨道转到了活性空间里（即分子轨道顺序为core-active-virtual)，所以对应的分子轨道系数也重新排序
        mycas.kernel(ca)
        norb = ca.shape[1]
        ao_rdm1 = mycas.make_rdm1() # for CASCI or CASSCF, make_rdm1 return rdm1 in AO    #??mycas是否使用态平均  -> 和直接给的rdm不一样
        print('----------------')
    # end --待修改完善
    # effective one-electron operator
    # two-particle -> one-particle eq27等号右边第二部分
    if amfi == False:
        h2_ao_soc = h2_soc_ao(mol)
        array1 = h2_ao_soc[:, :, :, :, :]
        array2 = np.einsum('rs, dpqrs -> dpq', ao_rdm1, array1) #right  
        array3 = np.einsum('rs, dprsq -> dpq', ao_rdm1, array1) #right
        array4 = np.einsum('rs, dsqpr -> dpq', ao_rdm1, array1) #right
    
        new_h2_ao_soc = array2 - 3 / 2 * array3 - 3 / 2 * array4

    else: # using AMFI
        ao_loc = mol.ao_loc_nr()
        nao = ao_loc[-1]
        array_amfi_1 = np.zeros((3, nao, nao),dtype=complex)
        array_amfi_2 = np.zeros((3, nao, nao),dtype=complex)
        atom = copy.copy(mol)
        aoslice = mol.aoslice_by_atom(ao_loc)
        for ia in range(mol.natm):
            b0, b1, p0, p1 = aoslice[ia]
            atom._bas = mol._bas[b0:b1]
            h2_ao_soc = h2_soc_ao(atom)
            array1 = h2_ao_soc[:, :, :, :, :]
            array2 = np.einsum('rs, dpqrs -> dpq', ao_rdm1[p0:p1, p0:p1], array1)
            array3 = np.einsum('rs, dprsq -> dpq', ao_rdm1[p0:p1, p0:p1], array1)
            array4 = np.einsum('rs, dsqpr -> dpq', ao_rdm1[p0:p1, p0:p1], array1)
            array_amfi_1[:, p0:p1, p0:p1] = array2
            array_amfi_2[:, p0:p1, p0:p1] = (array3 + array4) 
            
        new_h2_ao_soc = array_amfi_1 - 3 / 2 * array_amfi_2
    # eq 27
    #print(new_h2_ao_soc.shape)
    f_pq_ao = h1_ao_soc + new_h2_ao_soc
    f_pq = np.einsum('rij,ip,jq->rpq', f_pq_ao, ca[:, ndocc:ndocc + nact], ca[:, ndocc:ndocc + nact]) # MO right

    return f_pq

def trans_f_pq_from_pyscf_to_psi4(f_pq_pyscf, pyscf_orbs, psi4_orbs): # geom, basis_set, spin, charge, symmetry, nact, ndocc, nactel, ca=None, ao_rdm1=None
    #f_pq_pyscf = get_soc_fqp_pyscf(geom, basis_set, spin, charge, symmetry, nact, ndocc, nactel)
    # pyscf_orbs = get_pyscf_orbs_list(geom, basis_set, spin, charge, symmetry)
    # psi4_orbs = get_psi4_orbs_list(geom, spin, charge, psi4_wfn)
    
    length = len(psi4_orbs)
    indices = [i for i in range(length)]
    dic = {}
    for key, value in zip(indices, psi4_orbs):
        dic[key] = value
    f_soc_mo_psi4 = np.zeros((3, length, length),dtype=complex)
    
    for dim in range(3):
        for key1, value1 in dic.items():
            for key2, value2 in dic.items():
                index1_mo_pyscf = pyscf_orbs.index(value1)
                index2_mo_pyscf = pyscf_orbs.index(value2)
                value = f_pq_pyscf[dim,index1_mo_pyscf,index2_mo_pyscf]
                f_soc_mo_psi4[dim,key1,key2] += value
    return f_soc_mo_psi4

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

def dets_inac(nactel, nact): # det里应该也包含
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
                
                # for m in range(ndocc):
                #     det.create_alfa_bit(m)
                #     det.create_beta_bit(m)  
                    
                for i in beta_loc: 
                    det.create_beta_bit(i)
                dets.append(det) 
                type_dets_num += 1
        elif nb == 0:
            for alfa_loc in alfa_locs:                      
                det = forte.Determinant()    
                
                # for m in range(ndocc):
                #     det.create_alfa_bit(m)
                #     det.create_beta_bit(m) 
                    
                for j in alfa_loc: 
                    det.create_alfa_bit(j) 
                dets.append(det) 
                type_dets_num += 1
        else:
            for alfa_loc in alfa_locs: 
                for beta_loc in beta_locs:                          
                    det = forte.Determinant()   
                    
                    # for m in range(ndocc):
                    #     det.create_alfa_bit(m)
                    #     det.create_beta_bit(m) 
                        
                    for j in alfa_loc: 
                        det.create_alfa_bit(j)
                    for i in beta_loc:
                        det.create_beta_bit(i)                    
                    dets.append(det) 
                    type_dets_num += 1
        dets_dict[s] = type_dets_num
    return dets, dets_dict
    
def sign_soc(det1, det2, p, q, situation_num):
    if situation_num == 1:
        num = 0
        for i in range(p):
            if det2.get_alfa_bit(i) == True:
                num += 1
        for i in range(q):
            if det2.get_beta_bit(i) == True:
                num += 1
            
    if situation_num == 2:
        num = 0
        for i in range(p):
            if det2.get_beta_bit(i) == True:
                num += 1
        for i in range(q):
            if det2.get_alfa_bit(i) == True:
                num += 1
            
    if situation_num == 3:
        num = 0
        for i in range(q):
            if det2.get_alfa_bit(i) == True:
                num += 1
        for i in range(p):
            if det2.get_alfa_bit(i) == True:
                num += 1
        if q < p:
            num -= 1
            
    if situation_num == 4:
        num = 0
        for i in range(q):
            if det2.get_beta_bit(i) == True:
                num += 1
        for i in range(p):
            if det2.get_beta_bit(i) == True:
                num += 1
        if q < p:
            num -= 1      
            
    sign = -1 if num % 2 == 1 else +1
#     print(f'num = {num};sign = {sign}')
    return sign

def soc_rules(det1, det2, f_matrix_psi4, nact):
    int = 0
    
    for p in range(nact):
        for q in range(nact):
            other_orbs = [i for i in range(nact)]
            other_orbs.remove(p)
            if p != q:
                other_orbs.remove(q)
            li = []
            for j in other_orbs:
                if det1.get_alfa_bit(j) == det2.get_alfa_bit(j) and det1.get_beta_bit(j) == det2.get_beta_bit(j):
                    li.append(True)
            if len(li) == len(other_orbs):#起码mo的占据情况均相同
#                 print('------------------------')
#                 print(p,q)
                if  p == q:        
                    # situation 1
                    if det1.get_alfa_bit(p) == True and det1.get_beta_bit(q) == False:
                        if det2.get_alfa_bit(p) == False and det2.get_beta_bit(q) == True: 
#                             print('situation 1, p == q')
                            sign = sign_soc(det1, det2, p, q, 1)
                            int += sign * 0.5 * f_matrix_psi4[0, p, q]
                            int -= sign * 0.5 * (1j) * f_matrix_psi4[1, p, q]

                    # situation2
                    if det1.get_beta_bit(p) == True and det1.get_alfa_bit(q) == False:
                        if det2.get_beta_bit(p) == False and det2.get_alfa_bit(q) == True: 
                            sign = sign_soc(det1, det2, p, q, 2)
#                             print('situation 2, p == q')
#                             print(det1,det2,p,q,sign)
                            int += sign * 0.5 * f_matrix_psi4[0, p, q]
                            int += sign * 0.5 * (1j) * f_matrix_psi4[1, p, q]

                    # situation3
                    if det1.get_alfa_bit(p) == True and det2.get_alfa_bit(q) == True:
                        if det1.get_beta_bit(p) == det2.get_beta_bit(p) and det1.get_beta_bit(q) == det2.get_beta_bit(q):                                                  
                            sign = sign_soc(det1, det2, p, q, 3)
#                             print('situation 3, p == q')
#                             print(det1,det2,p,q,sign)
                            int += sign * 0.5 * f_matrix_psi4[2, p, q]

                    # situation4
                    if det1.get_beta_bit(p) == True and det2.get_beta_bit(q) == True:
                        if det1.get_alfa_bit(p) == det2.get_alfa_bit(p) and det1.get_alfa_bit(q) == det2.get_alfa_bit(q):                                            
                            sign = sign_soc(det1, det2, p, q, 4)
#                             print('situation 4, p == q')
#                             print(det1,det2,p,q,sign)

                            int -= sign * 0.5 * f_matrix_psi4[2, p, q]
                if p != q:
                    # situation1
                    if det1.get_alfa_bit(p) == True and det1.get_beta_bit(q) == False:
                        if det2.get_alfa_bit(p) == False and det2.get_beta_bit(q) == True: 
                            if det1.get_beta_bit(p) == det2.get_beta_bit(p) and det1.get_alfa_bit(q) == det2.get_alfa_bit(q):                                     
                                sign = sign_soc(det1, det2, p, q, 1)
#                                     print('situation 1, p!= q')
                                int += sign * 0.5 * f_matrix_psi4[0, p, q]
                                int -= sign * 0.5 * (1j) * f_matrix_psi4[1, p, q]
                    # situation2
                    if det1.get_beta_bit(p) == True and det1.get_alfa_bit(q) == False:
                        if det2.get_beta_bit(p) == False and det2.get_alfa_bit(q) == True: 
                            if det1.get_alfa_bit(p) == det2.get_alfa_bit(p) and det1.get_beta_bit(q) == det2.get_beta_bit(q): 
                                sign = sign_soc(det1, det2, p, q, 2)
#                                 print('situation 2, p != q')
#                                 print(det1,det2,p,q,sign)
                                int += sign * 0.5 * f_matrix_psi4[0, p, q]
                                int += sign * 0.5 * (1j) * f_matrix_psi4[1, p, q]
                    # situation3
                    if det1.get_alfa_bit(p) == True and det1.get_alfa_bit(q) == False:
                        if det2.get_alfa_bit(p) == False and det2.get_alfa_bit(q) == True: 
                            if det1.get_beta_bit(p) == det2.get_beta_bit(p) and det1.get_beta_bit(q) == det2.get_beta_bit(q): 
                                sign = sign_soc(det1, det2, p, q, 3)
#                                     print('situation 3, p != q')
#                                     print(det1,det2,p,q,sign)
                                int += sign * 0.5 * f_matrix_psi4[2, p, q]
                    # situation4
                    if det1.get_beta_bit(p) == True and det1.get_beta_bit(q) == False:
                        if det2.get_beta_bit(p) == False and det2.get_beta_bit(q) == True:   
                            if det1.get_alfa_bit(p) == det2.get_alfa_bit(p) and det1.get_alfa_bit(q) == det2.get_alfa_bit(q): 
                                sign = sign_soc(det1, det2, p, q, 4)
#                                     print('situation 4, p != q')
#                                     print(det1,det2,p,q,sign)
                                int -= sign * 0.5 * f_matrix_psi4[2, p, q]     

    return int   

def judge_sign(det1, det2, diff_pairs_det1_det2, type):
    #首先根据diff_pairs的长度判断是rule 2还是rule 3
    if len(diff_pairs_det1_det2) == 2: # rule 2
        n_sum = 0
        i, j = diff_pairs_det1_det2
        if i + 1 != j:
#             print('************')
            if type == 'aa':
                for m in range(i + 1, j, 1):
#                     print(m)
                    if det2.get_alfa_bit(m) == True:
                        n_sum += 1
            else: #type == 'bb'
                for m in range(i + 1, j, 1):
                    if det2.get_beta_bit(m) == True:
                        n_sum += 1
#         print(n_sum)
        sign = +1 if (n_sum) % 2 == 0 else -1      
        
    if len(diff_pairs_det1_det2) == 4: # rule 3
        i, j, k, l = diff_pairs_det1_det2
#         print(i,j,k,l)
        n_sum1, n_sum2 = 0, 0
        if type == 'aaaa':
#             print('---aaaaaa')
            if i + 1 != k:            
                for m in range(i + 1, k, 1):
                    if det2.get_alfa_bit(m) == True:
                        n_sum1 += 1
            if j + 1 != l:
                for n in range(j+1, l, 1):
                    if det2.get_alfa_bit(n) == True:
                        n_sum2 += 1
                        
                if j < k:
                    if det1.get_alfa_bit(k) == True:
                        n_sum2 += 1
                    if det1.get_alfa_bit(k) == False:
                        n_sum2 -= 1
#             print(n_sum2)
                    
        if type == 'bbbb':
#             print('bbbbbbbbbbb')
            if i + 1 != k:   
                for m in range(i+1, k, 1):
                    if det2.get_beta_bit(m) == True:
                        n_sum1 += 1
            if j + 1 != l:
                for n in range(j+1, l, 1):
                    if det2.get_beta_bit(n) == True:
                        n_sum2 += 1
                if j < k:
                    n_sum2 -= 1
                    
        if type == 'abab':
#             print('ababab')
            if i < k:
                if i + 1 != k:   
                    for m in range(i+1, k, 1):
                        if det2.get_alfa_bit(m) == True:
                            n_sum1 += 1
            if i > k:
                if k + 1 != i:   
                    for m in range(k+1, i, 1):
                        if det2.get_alfa_bit(m) == True:
                            n_sum1 += 1
            if j < l:
                if j + 1 != l:
                    for n in range(j+1, l, 1):
                        if det2.get_beta_bit(n) == True:
                            n_sum2 += 1   
            if j > l:
                if l + 1 != j:
                    for n in range(l+1, j, 1):
                        if det2.get_beta_bit(n) == True:
                            n_sum2 += 1 
        sign = +1 if (n_sum1 + n_sum2) % 2 == 0 else -1        
                
            
                
            
    return sign

def slater_rules(det1, det2, oei, tei, nact):   
    det1_nalpha, det2_nalpha, orbs_different_num = 0, 0, 0
    diff_pairs_aa, diff_pairs_bb = [], [] #里面是mo的序号，不是so
    for i in range(nact): 
        if det1.get_alfa_bit(i) == True:
            det1_nalpha += 1
        if det2.get_alfa_bit(i) == True:
            det2_nalpha += 1
        if det1.get_alfa_bit(i) != det2.get_alfa_bit(i):
            #无soc中，要想单双电子积分不为0，det1和det2必须alpha和ebta电子个数对应相等
            orbs_different_num += 1
            diff_pairs_aa.append(i) 
        if det1.get_beta_bit(i) != det2.get_beta_bit(i):
            orbs_different_num += 1
            diff_pairs_bb.append(i)               
            
    if det1_nalpha != det2_nalpha: # <alpha|beta> == 0
        return 0
    if orbs_different_num > 4: # 同一自旋的，存在两对轨道以上对应不同，积分为0; orbs_different_num/2 = 不同的轨道对个数
        return 0
    
    int = 0
    # rule 1:
    if orbs_different_num == 0:  
        for i in range(nact):
            # oei
            if det1.get_alfa_bit(i) == True:
                int += oei[(2 * i, 2 * i)]
            if det1.get_beta_bit(i) == True:
                int += oei[(2 * i + 1, 2 * i + 1)]
            # tei
            for j in range(nact):
                if det1.get_alfa_bit(i) == True and det1.get_alfa_bit(j) == True:
                    int += 0.5 * tei[(2 * i, 2 * j, 2 * i, 2 * j)]
                if det1.get_beta_bit(i) == True and det1.get_beta_bit(j) == True: 
                    int += 0.5 * tei[(2 * i + 1, 2 * j + 1, 2 * i + 1, 2 * j + 1)]
                if det1.get_alfa_bit(i) == True and det1.get_beta_bit(j) == True: 
                    int += 0.5 * tei[(2 * i, 2 * j + 1, 2 * i, 2 * j + 1)]
                if det1.get_beta_bit(i) == True and det1.get_alfa_bit(j) == True: 
                    int += 0.5 * tei[(2 * i + 1, 2 * j, 2 * i + 1, 2 * j)]

    # rule 2: 
    if orbs_different_num == 2: #仅有一对轨道不同。。还得判断是什么类型的轨道不同，是aa还是bb,借用diff_pairs_aa/bb存了下来
        #判断类型
        if len(diff_pairs_aa) == 2: #不同的轨道中占据的是alpha电子
            i, j = diff_pairs_aa
            sign = judge_sign(det1, det2, diff_pairs_aa, 'aa')
            
            # oei
            int += sign * oei[(2 * i, 2 * j)] 
            # tei
            for k in range(nact):
                if det1.get_alfa_bit(k) == True and det2.get_alfa_bit(k) == True: #应该是forte里面没有存<ab||aa>这样子的积分
                    #貌似上面用and连接的两个判断条件只写前面一个就可以。
                    #因为在执行这部之前，在前面已经判断了除了diff_pairs_aa里存的轨道的占据情况不同，其他轨道情况都相同
                    #所以，貌似只需要确定其中一个det里k这个位置有alfa电子，也就相应推出另一个det里k这个位置也应该有alfa电子
                    #下面判断beta电子的情况同理--------->>>> 所以这两处待修改
                    int += sign * tei[(2 * i, 2 * k, 2 * j, 2 * k)]
                if det1.get_beta_bit(k) == True and det2.get_beta_bit(k) == True:
                    int += sign * tei[(2 * i, 2 * k + 1, 2 * j, 2 * k + 1)]
                    
        if len(diff_pairs_bb) == 2: #不同的轨道中占据的是beta电子
            i, j = diff_pairs_bb          
            sign = judge_sign(det1, det2, diff_pairs_bb, 'bb')            
            # oei
            int += sign * oei[(2 * i + 1, 2 * j + 1)] 
            # tei
            for k in range(nact):
                if det1.get_beta_bit(k) == True and det2.get_beta_bit(k) == True: 
                    int += sign * tei[(2 * i + 1, 2 * k + 1, 2 * j + 1, 2 * k + 1)]
                if det1.get_alfa_bit(k) == True and det2.get_alfa_bit(k) == True:
                    int += sign * tei[(2 * i + 1, 2 * k , 2 * j + 1, 2 * k)]
                    
     # rule 3: 
    if orbs_different_num == 4: # 即仅有两对轨道不同
        if len(diff_pairs_aa) == 4: #两对的轨道中占据的是alpha电子
            # 貌似这里还得判断哪两个是匹配的，即哪两个在一个det上。下面通过对det1的四个位置哪两个位置get_alfa_bit的结果相同来进行判断
            for loc in diff_pairs_aa[1:]:
                if det1.get_alfa_bit(diff_pairs_aa[0]) == det1.get_alfa_bit(loc):
                    i, j = diff_pairs_aa[0], loc
                    det2_locs = diff_pairs_aa[1:]
                    det2_locs.remove(loc)
                    k ,l = det2_locs 
                    break
            sign = judge_sign(det1, det2, [i, j, k, l], 'aaaa')            
            int += sign * tei[(2 * i, 2 * j, 2 * k, 2 * l)]
        if len(diff_pairs_aa) == 2 and len(diff_pairs_bb) == 2:
            # i, k = diff_pairs_aa
            # j, l = diff_pairs_bb
            # sign = judge_sign(det1, det2, [i, j, k, l], 'abab')            
            # int += sign * tei[(2 * i, 2 * j + 1, 2 * k, 2 * l + 1)]
            
            # 如下test -->>经测试，正确，适用于DSRG的双电子积分
            if det1.get_alfa_bit(diff_pairs_aa[0]) == True:
                i, k = diff_pairs_aa 
            else:
                k, i = diff_pairs_aa
            if det1.get_beta_bit(diff_pairs_bb[0]) == True:
                j, l = diff_pairs_bb 
            else:
                l, j = diff_pairs_bb
            sign = judge_sign(det1, det2, [i, j, k, l], 'abab')            
            int += sign * tei[(2 * i, 2 * j + 1, 2 * k, 2 * l + 1)]


    
        if len(diff_pairs_bb) == 4:
            for loc in diff_pairs_bb[1:]:
                if det1.get_beta_bit(diff_pairs_bb[0]) == det1.get_beta_bit(loc):
                    i, j = diff_pairs_bb[0], loc
                    det2_locs = diff_pairs_bb[1:]
                    det2_locs.remove(loc)
                    k ,l = det2_locs 
                    break
            sign = judge_sign(det1, det2, [i, j, k, l], 'bbbb')           
            int += sign * tei[(2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1)]
           
    return int    

def get_ints_from_json(file_name):
    with open(file_name, 'r')as a:
        info = json.load(a)
    oei_li = info["oei"]["data"] # 获得一个嵌套列表
    tei_li = info["tei"]["data"]
    oei, tei = {}, {}
    for i in oei_li:
        oei[tuple(i[:2])] = i[-1]
    for j in tei_li:
        tei[tuple(j[:4])] = j[-1]
    scalar_e = info["scalar_energy"]["data"]
    return oei, tei, scalar_e

def h_spin_free(dets, dets_dict, nact, oei, tei):
    hamiltonian = np.zeros((len(dets), len(dets)))
    start_indice = 0
    end_indice = 0
    for s, type_dets_num in dets_dict.items():
        end_indice += type_dets_num
        for i in range(start_indice, end_indice, 1):
            for j in range(start_indice, i + 1, 1):
                hamiltonian[j, i] = hamiltonian[i, j] = slater_rules(dets[i], dets[j], oei, tei, nact)
        start_indice += type_dets_num
    return hamiltonian

def h_soc(dets, dets_dict, f_pq_psi4, nact):
    hamiltonian_soc = np.zeros((len(dets), len(dets)), dtype=complex)
    start1_indice, end1_indice = 0, 0
    for s1, type1_dets_num in dets_dict.items():
        end1_indice += type1_dets_num
        start2_indice, end2_indice = 0, 0
        for s2, type2_dets_num in dets_dict.items():
            end2_indice += type2_dets_num
            if abs(s1 - s2) <= 1:
#             print('true')
                for i in range(start1_indice, end1_indice, 1):
                    for j in range(start2_indice, i + 1, 1):
#                     print(start1_indice, end1_indice, start2_indice, end2_indice)
                        hamiltonian_soc[i, j] = soc_rules(dets[i], dets[j], f_pq_psi4, nact)
                        hamiltonian_soc[j, i] = np.conj(hamiltonian_soc[i,j])
            if j == end1_indice:
                continue
            start2_indice += type2_dets_num
        start1_indice += type1_dets_num
    return hamiltonian_soc
    