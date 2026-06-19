# negative_data_kisti.py
# KISTI 슈퍼컴퓨터 환경에서 MPI를 활용하여 ZINC 전체 데이터베이스로부터 표준화 및 Tanimoto 유사도 비교를 통해 음성 데이터셋을 병렬 선별하는 스크립트

from mpi4py import MPI
import pandas as pd
import numpy as np
import glob
import math
import os
import time
import random
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경용 백엔드 설정
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
# RDKit 내부 경고 메시지 출력 차단 수행
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

# 1. MPI 통신망 초기화 및 프로세스 정보 취득 처리
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
DONE = -1

if rank == 0:
    print("-> [마스터] MPI 병렬 처리 대조군 선별 프로세스 시작")

# 2. 0번 일꾼(Rank 0)이 마스터로서 양성 데이터 로딩 및 브로드캐스트 준비 수행
if rank == 0:
    print("-> [마스터] PubChem 농약 데이터셋 로드 및 표준화 개시")
    pos_df = pd.read_csv('PubChem_Agrochemical.csv')
    pos_df = pos_df.rename(columns={'SMILES': 'smiles', 'Molecular_Weight': 'mw', 'XLogP': 'xlogp'})
    pos_df = pos_df.dropna(subset=['smiles']).copy()
    
    # RDKit Canonical SMILES 표준화 및 중복 제거 수행
    pos_df['mol'] = [Chem.MolFromSmiles(s) for s in pos_df['smiles']]
    pos_df = pos_df[pos_df['mol'].notna()].copy()
    pos_df['standardized_smi'] = [Chem.MolToSmiles(m) for m in pos_df['mol']]
    pos_df = pos_df.drop_duplicates(subset='standardized_smi').reset_index(drop=True)
    pos_df['smiles'] = pos_df['standardized_smi']
    
    pos_smiles = pos_df['smiles'].tolist()
    
    # 분위수 및 통계 계산용 변수 준비 수행
    mw_low = pos_df['mw'].quantile(0.05)
    mw_high = pos_df['mw'].quantile(0.95)
    xlogp_low = pos_df['xlogp'].quantile(0.05)
    xlogp_high = pos_df['xlogp'].quantile(0.95)
    
    broadcast_data = {
        'pos_smiles': pos_smiles
    }
else:
    broadcast_data = None

# 3. 마스터 일꾼의 표준화된 양성 데이터를 다른 모든 일꾼 프로세스로 브로드캐스트 전송 처리
broadcast_data = comm.bcast(broadcast_data, root=0)

# 4. 모든 일꾼 프로세스가 수신받은 데이터를 기반으로 로컬 변수 설정 수행
pos_smiles = broadcast_data['pos_smiles']

# 5. 각 일꾼 프로세스 내부에서 수신한 SMILES를 RDKit Mol로 로컬 변환 및 핑거프린트 연산 수행
pos_mols = [Chem.MolFromSmiles(s) for s in pos_smiles]
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ref_fps = [fp_gen.GetFingerprint(m) for m in pos_mols if m is not None]
pos_smi_set = set(pos_smiles)

# 6. ZINC 데이터 파일 경로 확보 및 정렬 수행
all_files = sorted(glob.glob('./zinc_db/*.txt'))
combined_list = []

# 7. 멀티 프로세스 환경에서의 마스터-워커 동적 할당(Dynamic Queue) 수행
if rank == 0:
    start_time = time.time()
    print(f"-> [마스터] 동적 할당 모드로 병렬 처리 개시 (총 파일: {len(all_files)}개, 워커: {size - 1}명)")
    print("=" * 60)
    
    task_queue = list(range(len(all_files)))
    finished_workers = 0
    
    # 워커들로부터 요청 및 결과를 대기하여 동적 분배 처리
    while finished_workers < size - 1:
        status = MPI.Status()
        local_res = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
        worker = status.Get_source()
        
        # 수신된 결과 데이터 취합 처리
        if local_res:
            combined_list.extend(local_res)
            print(f"[마스터] 워커 {worker} -> 결과 {len(local_res)}개 취합 완료 (누적: {len(combined_list)}개)")
        
        # 남은 일감이 있을 경우 작업 배정 수행
        if task_queue:
            file_idx = task_queue.pop(0)
            comm.send(file_idx, dest=worker, tag=1)
            print(f"[마스터] 워커 {worker} <- 작업 배정 완료: {os.path.basename(all_files[file_idx])}")
        else:
            comm.send(DONE, dest=worker, tag=1)
            finished_workers += 1
            print(f"[마스터] 워커 {worker} <- 종료 신호 전송 완료 ({finished_workers}/{size - 1} 완료)")
            
    print("=" * 60)
    print(f"[마스터] 전체 워커 결과 수합 완료 (총 {len(combined_list)}개 후보군, 소요시간: {time.time() - start_time:.2f}초)")
    
else:
    # 워커(Rank > 0) 프로세스의 초기 작업 요청 수행
    comm.send([], dest=0, tag=0)
    
    while True:
        file_idx = comm.recv(source=0, tag=1)
        if file_idx == DONE:
            print(f"  [워커 {rank}] 종료 신호 수신 완료")
            break
            
        file_path = all_files[file_idx]
        print(f"  [워커 {rank}] {os.path.basename(file_path)} 처리 개시")
        
        local_candidates = []
        try:
            # Pandas pyarrow 엔진 적용 시도 및 예외 시 폴백 처리
            try:
                df = pd.read_csv(file_path, sep='\t', usecols=['smiles', 'zinc_id', 'mwt', 'logp'], engine='pyarrow')
            except Exception:
                df = pd.read_csv(file_path, sep='\t', usecols=['smiles', 'zinc_id', 'mwt', 'logp'])
            
            df = df.rename(columns={'mwt': 'mw', 'logp': 'xlogp'})
            df = df.dropna(subset=['smiles']).copy()
            
            for _, row in df.iterrows():
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol is None:
                    continue
                std_smi = Chem.MolToSmiles(mol)
                # 라벨 불일치 필터링(교집합 제외) 수행
                if std_smi in pos_smi_set:
                    continue
                
                # Morgan Fingerprint Tanimoto 유사도 계산 수행
                fp = fp_gen.GetFingerprint(mol)
                sims = BulkTanimotoSimilarity(fp, ref_fps)
                max_sim = max(sims) if sims else 0.0
                
                local_candidates.append({
                    'smiles': std_smi,
                    'zinc_id': row['zinc_id'],
                    'mw': float(row['mw']),
                    'xlogp': float(row['xlogp']),
                    'max_similarity_to_positive': max_sim
                })
            print(f"  [워커 {rank}] {os.path.basename(file_path)} 완료 (선별 수: {len(local_candidates)}개)")
        except Exception as e:
            print(f"-> [일꾼 {rank}] 파일 처리 오류 ({os.path.basename(file_path)}): {str(e)}")
        
        # 결과 데이터 전송 수행
        comm.send(local_candidates, dest=0, tag=0)

# 8. 마스터 일꾼(Rank 0)이 취합 완료된 전체 데이터를 기반으로 최종 필터링 및 시각화 수행
if rank == 0:
    combined_df = pd.DataFrame(combined_list)
    # 중복 화합물 최종 제거 수행
    combined_df = combined_df.drop_duplicates('smiles').reset_index(drop=True)
    
    # 컷오프 필터 적용 전 전체 취합 데이터 저장 수행
    combined_df[['smiles', 'zinc_id', 'mw', 'xlogp', 'max_similarity_to_positive']].to_csv('zinc_candidates_all_calculated.csv', index=False)
    print("-> [마스터] 컷오프 적용 전 전체 데이터 zinc_candidates_all_calculated.csv 저장 성공")
    
    # --- Task 1-3. 양성의 양성 대비 최대 유사도 vs 필터 적용 전 ZINC 후보군의 양성 대비 최대 유사도 분포 시각화 수행 ---
    print("-> [마스터] 양성군 내부 최대 유사도 분포 계산 개시")
    pos_max_sims = []
    for i, fp in enumerate(ref_fps):
        other_fps = ref_fps[:i] + ref_fps[i+1:]
        sims = BulkTanimotoSimilarity(fp, other_fps)
        pos_max_sims.append(max(sims) if sims else 0.0)
        
    zinc_all_max_sims = combined_df['max_similarity_to_positive'].tolist()
    
    print("-> [마스터] 히스토그램 시각화 이미지 생성 및 저장 수행")
    plt.figure(figsize=(10, 5.5))
    plt.rcParams['font.family'] = 'DejaVu Sans'  # KISTI 리눅스 범용 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지 설정
    
    # 양성 분포 시각화 수행
    plt.hist(
        pos_max_sims, 
        bins=50, 
        alpha=0.6, 
        label='Positive to positive similarity (Max Tanimoto)', 
        color='#1f77b4', 
        edgecolor='none', 
        density=True
    )
    
    # ZINC 후보군 분포 시각화 수행
    plt.hist(
        zinc_all_max_sims, 
        bins=50, 
        alpha=0.6, 
        label='ZINC candidates to positive similarity (Max Tanimoto)', 
        color='#ff7f0e', 
        edgecolor='none', 
        density=True
    )
    
    plt.title('Tanimoto Similarity Distribution: Positive vs ZINC', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Tanimoto Similarity', fontsize=11, labelpad=10)
    plt.ylabel('Density', fontsize=11, labelpad=10)
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    # 컷오프 경계선과 평균치 텍스트가 없는 순수 원시 히스토그램 먼저 저장 수행
    plt.savefig('similarity_distribution_comparison_raw.png', dpi=300)
    
    # 두 분포(양성 vs 필터 전 ZINC)의 밀도 함수가 교차하여 갈라지는 최적의 경계점(Cutoff) 자동 계산 수행
    bins = np.linspace(0.0, 1.0, 101)
    pos_hist, _ = np.histogram(pos_max_sims, bins=bins, density=True)
    zinc_hist, _ = np.histogram(zinc_all_max_sims, bins=bins, density=True)
    
    # 0.20에서 0.70 사이의 구간에서 두 분포의 밀도가 교차하는 지점 탐색 수행
    cuts = [float(bins[i]) for i in range(20, 70) if pos_hist[i] > zinc_hist[i] and pos_hist[i-1] <= zinc_hist[i-1]]
    optimal_cutoff = cuts[0] if cuts else 0.40
    print(f"-> [마스터] 동적으로 계산된 최적의 유사도 컷오프 경계선: {optimal_cutoff:.2f}")
    
    # 유사도 컷오프 경계선 추가 수행
    plt.axvline(
        x=optimal_cutoff, 
        color='#d62728', 
        linestyle='--', 
        linewidth=2, 
        label=f'Negative screening upper bound (Cutoff = {optimal_cutoff:.2f})'
    )

    # 컷오프 평균치 텍스트가 없는 순수 원시 히스토그램 먼저 저장 수행
    plt.savefig('similarity_distribution_comparison_cutoff.png', dpi=300)
    
    # 평균치 텍스트 표기 처리
    pos_mean_val = np.mean(pos_max_sims)
    neg_mean_val = np.mean(zinc_all_max_sims)
    plt.text(pos_mean_val + 0.02, plt.gca().get_ylim()[1]*0.6, f'Positive Max Sim\n(Mean ~{pos_mean_val:.2f})', color='#1f77b4', fontweight='bold')
    plt.text(neg_mean_val - 0.22, plt.gca().get_ylim()[1]*0.6, f'ZINC Max Sim\n(Mean ~{neg_mean_val:.2f})', color='#ff7f0e', fontweight='bold')
    
    # 범례 갱신 및 최종 히스토그램 저장 수행
    plt.legend(fontsize=10, loc='upper right')
    plt.savefig('similarity_distribution_comparison.png', dpi=300)
    plt.close()
    
    # 컷오프 이하인 음성 데이터셋 선별 수행
    final_neg = combined_df[combined_df['max_similarity_to_positive'] <= optimal_cutoff].copy()
    
    # 1:1 무작위 샘플링 수행 전의 선별 데이터 저장 수행
    final_neg[['smiles', 'zinc_id', 'mw', 'xlogp', 'max_similarity_to_positive']].to_csv('negative_candidates_before_sampling.csv', index=False)
    print("-> [마스터] 샘플링 전 전체 후보군 negative_candidates_before_sampling.csv 저장 성공")
    
    # 1:1 무작위 샘플링(Random Sampling) 수행
    seed_val = random.randint(1, 100000)
    sample_size = min(len(pos_df), len(final_neg))
    final_neg = final_neg.sample(n=sample_size, random_state=seed_val).reset_index(drop=True)
    
    # 최종 negative list csv 저장 수행
    final_neg[['smiles', 'zinc_id', 'mw', 'xlogp', 'max_similarity_to_positive']].to_csv('negative_agro_vs_zinc.csv', index=False)
    print("-> [마스터] 통합 negative_agro_vs_zinc.csv 저장 성공")
    
    # 메타데이터 파일 기록 및 저장 수행
    metadata_content = f"""[Task 1 Execution Metadata - KISTI MPI (All Candidates Version)]
Tanimoto Cutoff: {optimal_cutoff:.4f}
Positive MW 5-95% Range: {mw_low:.4f} ~ {mw_high:.4f}
Positive LogP 5-95% Range: {xlogp_low:.4f} ~ {xlogp_high:.4f}
Total Agrochemical Positives: {len(pos_df)}
Total Selected Negatives: {len(final_neg)}
Random Sampling Seed: {seed_val}
"""
    with open('final_v3_negative_metadata.txt', 'w', encoding='utf-8') as meta_file:
        meta_file.write(metadata_content)
        
    print("\n" + "="*50)
    print("      --- KISTI MPI 병렬 처리 완료 보고 ---")
    print(f"-> 전체 선별된 후보군 수: {len(combined_df)}개")
    print(f"-> 최종 저장된 음성 수: {len(final_neg)}개")
    print("="*50)
