# ==============================================================================
# [가산 B] 스코어 기반 구조 생성을 실행한다 (MPI 분산 환경을 지원한다).
# ==============================================================================

import os
import sys
import math
import time
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger

sys.stdout.reconfigure(encoding='utf-8')
RDLogger.DisableLog('rdApp.*')

# ---- 공통 스코어링 모듈을 임포트한다 ----
try:
    import pesticide_scorer
except ImportError:
    print("❌ 에러: pesticide_scorer.py 모듈을 찾을 수 없다.")
    print("   노트북(finalV7.ipynb)을 먼저 전체 실행하여 공통 모듈을 생성해야 한다.")
    sys.exit(1)

# ---- MPI를 초기화한다 ----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print_lock = True
DONE = None

# ---- 분자 생성 하이퍼파라미터 및 탐색을 설정한다 ----
START_SMI = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'    # 이부프로펜 (시작 구조)
K = 5                                       # 탐색 반복 깊이
BEAM = 100                                  # 빔 크기
NEIGHBOR_ELEMENTS = [6, 7, 8, 9, 15, 16, 17] # 이웃 치환 및 추가 원소(C, N, O, F, P, S, Cl)를 설정한다.

# 결과 시각화 차트 경로를 설정한다.
HISTOGRAM_PNG = 'taskB_score_distribution.png'
SCATTER_PNG = 'taskB_score_scatter.png'

# ---- 이웃 분자 생성 함수를 정의한다 ----
def generate_neighbors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    result = []
    # 1) 원자를 치환한다.
    for i in range(mol.GetNumAtoms()):
        for atom_num in NEIGHBOR_ELEMENTS:
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(i).SetAtomicNum(atom_num)
            try:
                Chem.SanitizeMol(rw)
                result.append(Chem.MolToSmiles(rw))
            except:
                pass
    # 2) 원자를 추가한다.
    for i in range(mol.GetNumAtoms()):
        for atom_num in NEIGHBOR_ELEMENTS:
            rw = Chem.RWMol(mol)
            new_idx = rw.AddAtom(Chem.Atom(atom_num))
            rw.AddBond(i, new_idx, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                result.append(Chem.MolToSmiles(rw))
            except:
                pass
    return list(set(result))

# ==============================================================================
# 메인 MPI 실행 구간이다.
# ==============================================================================
if rank == 0:
    print(f"=== MPI Score 기반 구조 생성 탐색을 시작한다 (프로세스 개수: {size}) ===")
    print(f"  - 시작 SMILES: {START_SMI}")
    print(f"  - 탐색 깊이 K: {K} | 빔 크기 BEAM: {BEAM}")
    print(f"  - 최적 가중치(JSON 로드): Property={pesticide_scorer.w_p:.2f}, Structure={pesticide_scorer.w_s:.2f}")
    print(f"  - 구조 가중치(JSON 로드): Scaffold={pesticide_scorer.best_w_scf:.2f}, Residue={1.0-pesticide_scorer.best_w_scf:.2f}")
    print("==================================================================")
    
    start_time = time.time()
    
    # 빔을 초기화한다.
    beam = [(pesticide_scorer.calculate_reward_score(START_SMI), START_SMI)]
    history = {START_SMI: beam[0][0]}
    
    for step in range(1, K + 1):
        step_start = time.time()
        
        # 1. 빔에 있는 분자들로부터 모든 이웃 후보군을 생성한다.
        candidates = set()
        for _, smi in beam:
            for nbr in generate_neighbors(smi):
                if nbr not in history:
                    candidates.add(nbr)
                    
        candidates_list = list(candidates)
        total_candidates = len(candidates_list)
        print(f"\n[Step {step}] 생성된 고유 후보군 개수: {total_candidates}개")
        
        if total_candidates == 0:
            print("더 이상 탐색할 수 있는 후보군이 없다. 조기 종료한다.")
            break
            
        # 2. Worker들에게 연산을 분배한다.
        chunks = np.array_split(candidates_list, size)
        for r in range(1, size):
            comm.send(list(chunks[r]), dest=r, tag=1)
            
        master_chunk = list(chunks[0])
        master_results = []
        for smi in master_chunk:
            score = pesticide_scorer.calculate_reward_score(smi)
            master_results.append((score, smi))
            
        # 3. Worker들의 계산 결과를 수집한다.
        all_results = list(master_results)
        for r in range(1, size):
            worker_res = comm.recv(source=r, tag=2)
            all_results.extend(worker_res)
            
        # 4. 히스토리를 업데이트하고 다음 빔을 구성한다.
        for score, smi in all_results:
            history[smi] = score
            
        unique_results = {smi: sc for sc, smi in all_results}
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        beam = [(sc, smi) for smi, sc in sorted_results[:BEAM]]
        
        step_elapsed = time.time() - step_start
        print(f"  - Step {step}을 완료하였다. 소요 시간: {step_elapsed:.2f}초")
        print(f"  - 현재 Step 내 최고 스코어: {beam[0][0]:.4f} | 구조: {beam[0][1]}")
        
    # 모든 탐색 종료 알림을 전송한다.
    for r in range(1, size):
        comm.send(DONE, dest=r, tag=1)
        
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 66)
    print("[RESULT] 탐색 완료 종합 리포트이다.")
    print(f"  - 총 탐색 시간: {total_elapsed:.2f}초")
    print(f"  - 방문한 총 고유 구조 개수: {len(history):,}개")
    print(f"  - 최종 최고 스코어 분자: {beam[0][1]}")
    print(f"  - 최종 최고 스코어: {beam[0][0]:.4f}")
    print("=" * 66)
    
    # 최종 시각화를 저장한다.
    scores = list(history.values())
    
    # (1) 히스토그램을 저장한다.
    plt.figure(figsize=(7, 4.5))
    plt.hist(scores, bins=40, color='#2563EB', alpha=0.75, edgecolor='black', linewidth=0.3)
    plt.title('Score Distribution of Explored Molecules', fontweight='bold', fontsize=12)
    plt.xlabel('Pesticide-likeness Score', fontsize=10)
    plt.ylabel('Molecule Count', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PNG, dpi=180)
    plt.close()
    
    # (2) 방문 순서 대비 스코어 변화 차트를 생성한다.
    plt.figure(figsize=(7, 4.5))
    plt.plot(scores, color='#EF4444', alpha=0.6, lw=0.8)
    plt.title('Exploration Trajectory', fontweight='bold', fontsize=12)
    plt.xlabel('Visited Order', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=180)
    plt.close()
    
    print(f"결과 시각화 이미지 저장을 완료하였다: {HISTOGRAM_PNG}, {SCATTER_PNG}")

else:
    # Worker 프로세스 루프이다.
    while True:
        task = comm.recv(source=0, tag=1)
        if task is DONE:
            break
            
        results = []
        for smi in task:
            score = pesticide_scorer.calculate_reward_score(smi)
            results.append((score, smi))
            
        comm.send(results, dest=0, tag=2)
