import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from scipy.sparse.linalg import eigs
from scipy.linalg import expm
from openfermion.utils import get_ground_state
from openfermion.transforms import get_sparse_operator


def get_Energy(bond_length):
    #パウリ演算子の準備
    nqubits = 4
    pI = np.array([[1+0.0j,0+0.0j],[0+0.0j,1+0.0j]])
    pX = np.array([[0+0.0j,1+0.0j],[1+0.0j,0+0.0j]])
    pZ = np.array([[1+0.0j,0+0.0j],[0+0.0j,-1+0.0j]])
    pY = np.array([[0+0.0j,-1.0j],[0.0+1.0j,0.0+0.0j]])
    pHad = (pX+pZ)/np.sqrt(2)
    pP0 = (pI+pZ)/2
    pP1 = (pI-pZ)/2
    
    #任意の状態に演算できるように準備
    X=[1]*(nqubits)
    Y=[1]*(nqubits)
    Z=[1]*(nqubits)
    H=[1]*(nqubits)
    P0=[1]*(nqubits)
    P1=[1]*(nqubits)
    for i in range(nqubits):
        for j in range(nqubits):
            if(i != j):
                X[i] = np.kron(pI,X[i])
                Y[i] = np.kron(pI,Y[i])
                Z[i] = np.kron(pI,Z[i])
                H[i] = np.kron(pI,H[i])
                P0[i] = np.kron(pI,P0[i])
                P1[i] = np.kron(pI,P1[i])
            else:
                X[i] = np.kron(pX,X[i])
                Y[i] = np.kron(pY,Y[i])
                Z[i] = np.kron(pZ,Z[i])
                H[i] = np.kron(pHad,H[i])
                P0[i] = np.kron(pP0,P0[i])
                P1[i] = np.kron(pP1,P1[i])
    Ide = np.eye(2**nqubits)
    
    #2量子ゲートの準備
    CZ = [[0 for i in range(nqubits)] for j in range(nqubits)]
    CX = [[0 for i in range(nqubits)] for j in range(nqubits)]
    for i in range(nqubits):
        for j in range(nqubits):
            CZ[i][j]= (P0[i]+np.dot(P1[i],Z[j]))
            CX[i][j]= (P0[i]+np.dot(P1[i],X[j]))
        
    #変分量子ゲートの準備
    def iSWAP(target1,target2,angle):
        return expm(-0.5*angle*1.j*CX[target1][target2])
    def iCPHASE(target1,target2,angle):
        return expm(-0.5*angle*1.j*CZ[target1][target2])
    def RX(target,angle):
        return expm(-0.5*angle*1.j*X[target])
    def RY(target,angle):
        return expm(-0.5*angle*1.j*Y[target])
    def RZ(target,angle):
        return expm(-0.5*angle*1.j*Z[target])

    #初期状態の準備
    def StateZeros(nqubits):
        State = np.zeros(2**nqubits)
        State[9]=1
        return State

    #求めるハミルトニアンのデータ
    geometry = [["H", [0,0,0]],
                ["H", [0,0,bond_length]]]
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    description = "test" #str()

    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    
    #求めるハミルトニアンのJW変換
    molecule = run_psi4(molecule)

    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))

    jw_matrix = get_sparse_operator(jw_hamiltonian)
    
    #量子回路
    n_param = 12
    def QubitPQC(phi):
        state = StateZeros(4)
        state = np.dot(iSWAP(0,1,phi[0]),state)
        state = np.dot(iCPHASE(0,1,phi[1]),state)
        state = np.dot(iSWAP(2,3,phi[2]),state)
        state = np.dot(iCPHASE(2,3,phi[3]),state)
        state = np.dot(iSWAP(1,2,phi[4]),state)
        state = np.dot(iCPHASE(1,2,phi[5]),state)
        state = np.dot(iSWAP(0,1,phi[6]),state)
        state = np.dot(iCPHASE(0,1,phi[7]),state)
        state = np.dot(iSWAP(2,3,phi[8]),state)
        state = np.dot(iCPHASE(2,3,phi[9]),state)
        state = np.dot(iSWAP(1,2,phi[10]),state)
        state = np.dot(iCPHASE(1,2,phi[11]),state)
        return state

    #エネルギーの期待値を求める関数
    def ExpectVal(Operator,State):
        BraState = np.conjugate(State.T) #列ベクトルを行ベクトルへ変換
        tmp = BraState.dot(Operator.dot(State)) #行列を列ベクトルと行ベクトルではさむ
        return np.real(tmp) #要素の実部を取り出す

    #VQEの実行
    def cost(phi):
        return ExpectVal(jw_matrix, QubitPQC(phi))

    init = np.random.rand(n_param)
    res = scipy.optimize.minimize(cost, init,
                              method='Powell')
    
    molecule = run_psi4(molecule,run_scf=1,run_fci=1)
    
    eigenenergies, eigenvecs = eigs(jw_matrix)
    
    return cost(res.x)




def FCI_get_Energy(bond_length):
    #求めるハミルトニアンのデータ
    geometry = [["H", [0,0,0]],
                ["H", [0,0,bond_length]]]
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    description = "test" #str()

    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    
    #求めるハミルトニアンのJW変換
    molecule = run_psi4(molecule)
    jw_hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    jw_matrix = get_sparse_operator(jw_hamiltonian)
    
    #FCI計算
    molecule = run_psi4(molecule,run_scf=1,run_fci=1)
    eigenenergies, eigenvecs = eigs(jw_matrix)
    
    return molecule.fci_energy



def HF_get_Energy(bond_length):
        
    #求めるハミルトニアンのデータ
    geometry = [["H", [0,0,0]],
                ["H", [0,0,bond_length]]]
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    description = "test" #str()

    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_psi4(molecule)
    
    return molecule.hf_energy




initial = 0.20
step = 0.025
number = 28*4+1
data1 = []
data2 = []
data3 = []
data4 = []
for i in range(number):
    bond_length = initial + i*step 
    data1.append(bond_length)
    ENERGY = get_Energy(bond_length)
    data2.append(ENERGY)
    FCIENERGY = FCI_get_Energy(bond_length)
    data3.append(FCIENERGY)
    HFENERGY = HF_get_Energy(bond_length)
    data4.append(HFENERGY)
    
    
# 図にプロット
plt.plot(data1,data2,"red")
plt.plot(data1,data3,"blue",linestyle="dashed")
plt.plot(data1,data4,"green",linestyle="dashed")
plt.xlabel("R (Å)")
plt.ylabel("Ground Energy (hartree)")
plt.xlim(0.20,3.00)
plt.ylim(-1.3,0.2)
plt.show                  

