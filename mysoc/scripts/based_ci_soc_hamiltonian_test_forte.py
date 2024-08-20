import forte                
import json         
import numpy as np
import re
from create_psi4_cas_hamiltonian import *

def get_ci_and_coeff(file_name):
    _, root_order, s2, irrep, _, ms = file_name.split('_')
    forte_ci_dict = {}
    with open(file_name, 'r')as r:
        lines = r.read().strip().split('\n')
        for line in lines:
            values = re.split(r'\s+', line)  # 使用\s+匹配一个或多个空格
            # for value in values:
            if np.absolute(float(values[-1])) < 1e-10:
                continue
            forte_ci_dict[''.join(values[:-1])] = float(values[-1]) # 连接values中除了最后一个之外的所有字符串作为键，最后一个字符串作为值 #之前的' '.join
    return forte_ci_dict

def dets_coeffs(forte_ci_dict): #实际获得的是str和对应的coeffs
    dets = []
    coeffs = []
    for str, coeff in forte_ci_dict.items():
        dets.append(str)
        coeffs.append(coeff)
    return dets, coeffs 
    
    
def tdm_1_test(strs1, coeffs1, strs2, coeffs2, p, q): ##a^(+)_(pα)a_(qβ) using SparseOperato and StateVector
    op = forte.SparseOperator(antihermitian=False)
    op.add_term_from_str(f'[{p}a+ {q}b-]',1.0)
    val = 0 
    for i, coeff_i in enumerate(coeffs1):
        for j, coeff_j in enumerate(coeffs2):
            det_i = forte.StateVector({ forte.det(strs1[i]): coeff_i})
            det_j = forte.StateVector({ forte.det(strs2[j]): coeff_j})
            # print(det_j)
            c = forte.get_projection(op, det_j, det_i)            
            val += c[0]                    
    return val

def tdm_2_test(strs1, coeffs1, strs2, coeffs2, p, q): #a^(+)_(pβ)a_(qα) using SparseOperato and StateVector
    op = forte.SparseOperator(antihermitian=False)
    op.add_term_from_str(f'[{p}b+ {q}a-]',1.0)
    val = 0 
    for i, coeff_i in enumerate(coeffs1):       
        for j, coeff_j in enumerate(coeffs2):
            det_i = forte.StateVector({ forte.det(strs1[i]): coeff_i})
            det_j = forte.StateVector({ forte.det(strs2[j]): coeff_j})
            c = forte.get_projection(op, det_j, det_i)
            val += c[0]                    
    return val

def tdm_3_test(strs1, coeffs1, strs2, coeffs2, p, q): ##a^(+)_(pα)a_(qα) using SparseOperato and StateVector
    op = forte.SparseOperator(antihermitian=False)
    op.add_term_from_str(f'[{p}a+ {q}a-]',1.0)
    val = 0 
    for i, coeff_i in enumerate(coeffs1):        
        for j, coeff_j in enumerate(coeffs2):
            det_i = forte.StateVector({ forte.det(strs1[i]): coeff_i})
            det_j = forte.StateVector({ forte.det(strs2[j]): coeff_j})
            c = forte.get_projection(op, det_j, det_i)
            val += c[0]                    
    return val

def tdm_4_test(strs1, coeffs1, strs2, coeffs2, p, q): ##a^(+)_(pβ)a_(qβ) using SparseOperato and StateVector
    op = forte.SparseOperator(antihermitian=False)
    op.add_term_from_str(f'[{p}b+ {q}b-]',1.0)
    val = 0 
    for i, coeff_i in enumerate(coeffs1):
        for j, coeff_j in enumerate(coeffs2):
            det_i = forte.StateVector({ forte.det(strs1[i]): coeff_i})
            det_j = forte.StateVector({ forte.det(strs2[j]): coeff_j})
            c = forte.get_projection(op, det_j, det_i)
            val += c[0]                    
    return val

def tdm_matrix(nact, forte_ci_dict1, forte_ci_dict2):# irrep_order, mo_irrep
    strs1, coeffs1 = dets_coeffs(forte_ci_dict1)
    strs2, coeffs2 = dets_coeffs(forte_ci_dict2)
    tdm_matrix_x,  tdm_matrix_y, tdm_matrix_z= [np.zeros((nact, nact), dtype=complex) for _ in range(3)]
    
    for p in range(nact):
        for q in range(nact):

            # x direction
            tdm_matrix_x[p, q] += 0.5 * tdm_1_test(strs1, coeffs1, strs2, coeffs2, p, q)
            tdm_matrix_x[p, q] += 0.5 * tdm_2_test(strs1, coeffs1, strs2, coeffs2, p, q)
            # y direction
            tdm_matrix_y[p, q] += 0.5 * (1j) * tdm_2_test(strs1, coeffs1, strs2, coeffs2, p, q)
            tdm_matrix_y[p, q] -= 0.5 * (1j) * tdm_1_test(strs1, coeffs1, strs2, coeffs2, p, q)
            # z direction
            tdm_matrix_z[p, q] += 0.5 * tdm_3_test(strs1, coeffs1, strs2, coeffs2, p, q)
            tdm_matrix_z[p, q] -= 0.5 * tdm_4_test(strs1, coeffs1, strs2, coeffs2, p, q)
            # print(f'---------------------{tdm_matrix_z[p, q]}')
    tdm_matrix = np.array([tdm_matrix_x,  tdm_matrix_y, tdm_matrix_z])     
    # print(tdm_matrix_z)
    # print(tdm_matrix)
    # print(np.linalg.norm(tdm_matrix))
    # print(np.linalg.eigh(tdm_matrix)[0])
    return tdm_matrix

def h_soc_element(nact, forte_ci_dict1, forte_ci_dict2, f_pq): # irrep_order, mo_irrep, 
    # only get matrix element of hamiltion_soc
    tdm = tdm_matrix(nact, forte_ci_dict1, forte_ci_dict2)
    # print(tdm)
    # print(f'tdm:{tdm}')
    h_element = np.einsum('xij,xij->', f_pq, tdm)
    return h_element

# def h_soc_based_ci(nact, ci_list, ireep_order, mo_irrep, f_pq):
#     h_soc = np.zeros((len(ci_list), len(ci_list)), dtype=complex)
#     # print(len(ci_list))
#     for i_ci in ci_list:
#         for j_ci in ci_list:
#             # if i_ci == ci_list[0] and j_ci == ci_list[1]:
#             element = h_soc_element(nact, i_ci, j_ci, ireep_order, mo_irrep, f_pq)
#             # print(f'element: {element}')
#             indice1 = ci_list.index(i_ci)
#             indice2 = ci_list.index(j_ci)
#             h_soc[indice1, indice2] += element        
    return h_soc
def h_soc_based_ci(nact, ci_list, f_pq, info_list): # irrep_order, mo_irrep, 
    h_soc = np.zeros((len(ci_list), len(ci_list)), dtype=complex)
    # print(len(ci_list))
    for indice1, i_ci in enumerate(ci_list):
        for indice2, j_ci in enumerate(ci_list):
            if indice2 > indice1:
                continue
            indice2 = ci_list.index(j_ci)
            # print(indice1, indice2)
            i_info = info_list[indice1]
            _, multi_i, _, ms2_i = i_info.split(' ')
            
            j_info = info_list[indice2]
            _, multi_j, _, ms2_j = j_info.split(' ')
       
            if abs(int(multi_i) - int(multi_j)) <= 2:
                if abs(int(ms2_i) - int(ms2_j)) <= 2:
                    element = h_soc_element(nact, i_ci, j_ci, f_pq)
                    # print(f'element: {element}')
                    h_soc[indice1, indice2] = element
                    if indice1 != indice2:
                        h_soc[indice2, indice1] = element.conj()

    # print('----------------')
    return h_soc

def get_ci_list(info_list): # 按顺序产生生成SOC矩阵的基，将他们存在一个List里
    ci_list = []
    for info in info_list:
        root_order, multi, irrep, ms2 = info.split(' ')
        s = (int(multi) - 1) / 2
        file_name = 'output_root' + root_order + '_' + str(s * (s + 1)) + '_' + irrep + '_' + 'ms_' + str(int(ms2) / 2) + '.txt'
        ci = get_ci_and_coeff(file_name)
        ci_list.append(ci)
    return ci_list