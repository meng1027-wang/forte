import json         
import numpy as np

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

def get_rdms_from_json(file_name):
    with open(file_name, 'r')as a:
        info = json.load(a)
    gamma1 = info["gamma1"]["data"] # 获得一个嵌套列表
    gamma2 = info["gamma2"]["data"]
    rdm1, rdm2 = {}, {}
    for i in gamma1:
        rdm1[tuple(i[:2])] = i[-1]
    for j in gamma2:
        rdm2[tuple(j[:4])] = j[-1]
    return rdm1, rdm2 

def get_element_from_pdm(oei, tei, pdm_file, nact): #根据能量的计算公式，貌似不需要知道有哪些行列式，只需要知道活性空间内轨道对应的单双电子积分和约化密度矩阵即可
    rdm1, rdm2 = get_rdms_from_json(pdm_file)
    e1, e2 = 0, 0
    for i in range(nact):
        for j in range(nact):
            # 1-electron part
            #------------------------------------------------------修改
            e1 += rdm1[(2 * i, 2 * j)] * oei[(2 * j, 2 * i)]
            e1 += rdm1[(2 * i + 1, 2 * j + 1)] * oei[(2 * j + 1, 2 * i + 1)]            
    
            for k in range(nact):
                for l in range(nact):
                    #2-electron part： 
                    #------------------------------------------------------修改2
                    e2 += rdm2[(2 * i, 2 * j, 2 * k, 2 * l)] * tei[(2 * k, 2 * l, 2 * i, 2 * j)]
                    # print(e2)
                    e2 += rdm2[(2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1)] * tei[(2 * k + 1, 2 * l + 1, 2 * i + 1, 2 * j + 1)]
                    # print(e2)
                    e2 += rdm2[(2 * i, 2 * j + 1, 2 * k, 2 * l + 1)] * tei[(2 * k, 2 * l + 1, 2 * i, 2 * j + 1)]
                    # print(e2)                    
                    e2 += rdm2[(2 * i, 2 * j + 1, 2 * k + 1, 2 * l)] * tei[(2 * k + 1, 2 * l, 2 * i, 2 * j + 1)] #?
                    # print(e2)                    
                    e2 += rdm2[(2 * i + 1, 2 * j, 2 * k + 1, 2 * l)] * tei[(2 * k + 1, 2 * l, 2 * i + 1, 2 * j)]
                    # print(e2)                    
                    e2 += rdm2[(2 * i + 1, 2 * j, 2 * k, 2 * l + 1)] * tei[(2 * k, 2 * l + 1, 2 * i + 1, 2 * j)] #?
    # print(e1, e2)
                    # print(e2)
    e = e1 + 0.25 * e2 #这样获得的是unrelaxed的能量。想获得relaxed的能量，需要多个态，多个态构成哈密顿矩阵然后对角化
    return e

def trans_matrix_order(old_basis, new_basis, old_matrix):
    size = len(old_basis)
    # P:permutation_matrix
    P = np.zeros((size, size))
    for i, val in enumerate(old_basis):
        j = new_basis.index(val)
        P[j, i] = 1 
    new_matrix = P @ old_matrix @ P.T   #@==np.dot
    return new_matrix
    