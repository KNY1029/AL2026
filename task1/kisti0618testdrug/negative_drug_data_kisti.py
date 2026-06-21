# negative_drug_data_kisti.py
# KISTI 슈퍼컴퓨터 환경에서 MPI를 활용하여 ZINC 전체 데이터베이스로부터 표준화 및 Tanimoto 유사도 비교를 통해 음성 데이터셋을 병렬 선별하는 스크립트 (대용량 2패스)

# 테스트 모드 활성화 여부 설정 (1: 테스트 모드 가동 및 10개 파일 제한, 0: 전체 데이터 대상 정식 가동)
test = 1

# 시스템 연산, 병렬 통신, 분자 구조식 제어, 데이터 가공을 위한 외부 라이브러리
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

# 테스트 모드 활성화 여부에 따라 저장할 파일들의 이름 맨 앞에 test_ 접두사를 추가
prefix = "dtest_" if test == 1 else "d"
raw_img_file = f"{prefix}distribution_comparison_raw.png"
cutoff_img_file = f"{prefix}distribution_comparison_cutoff.png"
final_img_file = f"{prefix}distribution_comparison.png"
neg_before_file = f"{prefix}negative_candidates_before_sampling.csv"
final_neg_file = f"{prefix}negative_drug_vs_zinc.csv"
metadata_file = f"{prefix}negative_metadata.txt"

# MPI 분산 병렬 통신 네트워크 환경의 초기화 선언 및 현재 프로세스의 개별 ID 및 전체 코어 개수 취득
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
DONE = -1

# 0번 마스터 노드
if rank == 0:
    print("-> [마스터] MPI 병렬 처리 대조군 선별 프로세스 시작")

# 0번 마스터 프로세스가 로컬 디스크로부터 기준점이 될 양성(농약) 데이터셋을 로드하고 필수 변수명을 표준화 처리
if rank == 0:
    print("-> [마스터] PubChem 약 데이터셋 로드 및 표준화 개시")
    pos_df = pd.read_csv('PubChem_Drugs.csv')
    pos_df = pos_df.rename(columns={'SMILES': 'smiles', 'Molecular_Weight': 'mw', 'XLogP': 'xlogp'})
    pos_df = pos_df.dropna(subset=['smiles']).copy()
    
    # RDKit 분자 객체 생성을 통하여 데이터의 이상 유무를 검증, 중복이 제거된 표준 정규화 SMILES 도출
    pos_df['mol'] = [Chem.MolFromSmiles(s) for s in pos_df['smiles']]
    pos_df = pos_df[pos_df['mol'].notna()].copy()
    pos_df['standardized_smi'] = [Chem.MolToSmiles(m) for m in pos_df['mol']]
    pos_df = pos_df.drop_duplicates(subset='standardized_smi').reset_index(drop=True)
    pos_df['smiles'] = pos_df['standardized_smi']
    
    pos_smiles = pos_df['smiles'].tolist()
    
    broadcast_data = {
        'pos_smiles': pos_smiles
    }
else:
    broadcast_data = None

# 마스터가 정제 완료한 표준 양성 SMILES 리스트를 분산 네트워크에 연결된 모든 워커 프로세스로 브로드캐스트 전송
broadcast_data = comm.bcast(broadcast_data, root=0)

# 모든 일꾼 노드들이 마스터로부터 공유받은 딕셔너리 전송 객체 내부에서 동기화된 양성 SMILES 리스트를 개별 추출
pos_smiles = broadcast_data['pos_smiles']

# 각 일꾼 노드 내부에서 할당받은 SMILES 구조식을 RDKit 분자 객체로 복원하고 분자 유사도 대조용 Morgan Fingerprint 연산
pos_mols = [Chem.MolFromSmiles(s) for s in pos_smiles]
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ref_fps = [fp_gen.GetFingerprint(m) for m in pos_mols if m is not None]
pos_smi_set = set(pos_smiles)

# 분산 연산 대상이 될 대용량 ZINC 데이터베이스 파일들을 정해진 디렉토리 내부에서 탐색 후 정렬
all_files = sorted(glob.glob('./zinc_db/*.txt'))
combined_list = []

# 테스트 모드가 활성화된 경우 전체 ZINC 파일 목록 중 앞의 10개 파일만 슬라이싱하여 제한 처리 수행
if test == 1:
    all_files = all_files[:10]

# 유사도 매칭 점수를 0.0부터 1.0까지 100개의 구간으로 쪼개어 카운트하기 위한 전역 통계용 bins 인덱스 배열 생성
bins = np.linspace(0.0, 1.0, 101)


# ===================================================================================
# [대용량 최적화] 1패스(Pass 1): 디스크 쓰기 없이 전역 유사도 히스토그램 도출 연산
# ===================================================================================

# 0번 마스터 노드가 1패스 연산의 시작을 로그에 기록하고 동적 작업 큐 및 빈도수 카운트 배열 초기화
if rank == 0:
    start_time = time.time()
    print(f"-> [마스터] 1패스: 전역 유사도 히스토그램 연산 개시 (총 파일: {len(all_files)}개)")
    print("=" * 60)
    
    task_queue = list(range(len(all_files)))
    finished_workers = 0
    zinc_hist_counts = np.zeros(100, dtype=np.int64)
    total_candidates_count = 0
    
    # 동적 큐 알고리즘을 가동하여 개별 파일의 처리를 끝낸 워커에게 다음 파일 인덱스를 비동기식으로 배정
    while finished_workers < size - 1:
        status = MPI.Status()
        local_count_data, local_hist = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
        worker = status.Get_source()
        
        if local_count_data > 0:
            total_candidates_count += local_count_data
        if local_hist is not None:
            zinc_hist_counts += local_hist
            
        if task_queue:
            file_idx = task_queue.pop(0)
            comm.send(file_idx, dest=worker, tag=1)
        else:
            comm.send(DONE, dest=worker, tag=1)
            finished_workers += 1
    print(f"[마스터] 1패스 완료 (총 후보군 수: {total_candidates_count}개, 소요시간: {time.time() - start_time:.2f}초)")

# 워커 노드들이 마스터에게 첫 신호를 보내어 일감을 요청하고 대기하는 루프 개시
else:
    comm.send((0, None), dest=0, tag=0)
    while True:
        file_idx = comm.recv(source=0, tag=1)
        if file_idx == DONE:
            break
            
        file_path = all_files[file_idx]
        local_count = 0
        max_sims_local_file = []
        
        # [메모리 과부하 방지] 워커 노드가 배정받은 대용량 파일을 5만 행씩 파일 분할 인덱싱 처리 수행
        try:
            for chunk in pd.read_csv(file_path, sep='\t', usecols=['smiles', 'zinc_id', 'mwt', 'logp'], chunksize=50000):
                chunk = chunk.rename(columns={'mwt': 'mw', 'logp': 'xlogp'}).dropna(subset=['smiles'])
                smi_arr = chunk['smiles'].tolist()
                
                # 분할 로드된 화합물 구조 정보들을 순회하며 RDKit 분자 변환 및 양성 그룹과의 Tanimoto 유사도 행렬 계산
                for s in smi_arr:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None: continue
                    std_smi = Chem.MolToSmiles(mol)
                    if std_smi in pos_smi_set: continue
                    
                    fp = fp_gen.GetFingerprint(mol)
                    sims = BulkTanimotoSimilarity(fp, ref_fps)
                    max_sim = max(sims) if sims else 0.0
                    max_sims_local_file.append(max_sim)
                    local_count += 1
                    
            # 1패스 고속 연산을 위해 분자 문자열을 버리고 오직 해당 파일 내 유사도 빈도수(히스토그램) 배열만 집계
            local_counts, _ = np.histogram(max_sims_local_file, bins=bins)
        except Exception as e:
            local_counts = np.zeros(100, dtype=np.int64)
            local_count = 0
            
        # 해당 ZINC 파일에서 도출된 정수형 빈도수 배열만을 마스터 노드로 송신하여 네트워크 부하 최소화
        comm.send((local_count, local_counts), dest=0, tag=0)


# ===================================================================================
# 9. 마스터 일꾼(Rank 0)의 동적 최적 컷오프 경계선 계산 및 브로드캐스트 처리 수행
# ===================================================================================

# 0번 마스터 노드가 기준이 될 양성 데이터셋 자기 자신들 간의 상호 교차 유사도를 전수 계산 개시
if rank == 0:
    print("-> [마스터] 양성군 내부 최대 유사도 분포 계산 및 시각화 개시")
    pos_max_sims = []
    for i, fp in enumerate(ref_fps):
        other_fps = ref_fps[:i] + ref_fps[i+1:]
        sims = BulkTanimotoSimilarity(fp, other_fps)
        pos_max_sims.append(max(sims) if sims else 0.0)
        
    pos_hist_counts, _ = np.histogram(pos_max_sims, bins=bins)
    
    # 매트플롯립 엔진의 그래픽 캔버스를 초기화하고 폰트 깨짐 및 음수 기호 마스킹 오류 방지 설정 적용
    plt.figure(figsize=(10, 5.5))
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    
    # 1패스 통신으로 누적 완료된 전역 카운트 데이터를 총 화합물 수로 나누어 정규 밀도 함수 플롯 정보 산출
    bin_width = bins[1] - bins[0]
    total_pos_count = len(pos_max_sims)
    pos_density = pos_hist_counts / (total_pos_count * bin_width) if total_pos_count > 0 else np.zeros(100)
    zinc_density = zinc_hist_counts / (total_candidates_count * bin_width) if total_candidates_count > 0 else np.zeros(100)
    
    # 양성 화합물의 자가 유사도 곡선과 ZINC 전체 후보군의 유사도 곡선을 오버레이하여 바 플롯 형태로 시각화
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    plt.bar(bin_centers, pos_density, width=bin_width, alpha=0.6, label="Positive self-similarity", color="#1f77b4", edgecolor="none")
    plt.bar(bin_centers, zinc_density, width=bin_width, alpha=0.6, label="ZINC candidates to positive", color="#ff7f0e", edgecolor="none")
    
    # 그래프의 기본 설정 후 이미지 파일로 1차 저장 수행
    plt.title("Tanimoto Similarity Distribution: Positive vs Pre-filtered ZINC", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Tanimoto Similarity", fontsize=11, labelpad=10)
    plt.ylabel("Density", fontsize=11, labelpad=10)
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(raw_img_file, dpi=300)
    
    # 양성 밀도가 전체 후보 밀도를 역전하는 첫 번째 교차 영역을 연산하여 최적의 동적 유사도 임계점(컷오프) 도출
    cuts = [float(bins[i]) for i in range(20, 70) if pos_density[i] > zinc_density[i] and pos_density[i-1] <= zinc_density[i-1]]
    optimal_cutoff = cuts[0] if cuts else 0.40
    print(f"-> [마스터] 동적으로 계산된 최적의 유사도 컷오프 경계선: {optimal_cutoff:.2f}")
    
    # 계산 완료된 동적 최적 유사도 임계 영역에 수직 경계 판정선을 차트에 추가 수행
    plt.axvline(x=optimal_cutoff, color="#d62728", linestyle="--", linewidth=2, label=f"Cutoff = {optimal_cutoff:.2f}")
    
    # 평균치 텍스트가 마스킹되기 전 상태의 컷오프 경계선 포함 그래프 이미지를 디스크에 저장
    plt.legend(fontsize=10, loc="upper right")
    plt.savefig(cutoff_img_file, dpi=300)
    
    # 양성군 및 후보군의 유사도 정량적 분포 수준을 텍스트로 가시화하기 위한 가중 평균값 계산
    pos_mean_val = np.mean(pos_max_sims)
    zinc_mean_val = np.sum(bin_centers * zinc_hist_counts) / total_candidates_count if total_candidates_count > 0 else 0.0
    
    # 차트 내의 지정 좌표 구역에 계산 완료된 각 집단별 평균유사도 수치 텍스트 삽입
    plt.text(pos_mean_val + 0.02, plt.gca().get_ylim()[1]*0.6, f"Positive Mean\n(~{pos_mean_val:.2f})", color="#1f77b4", fontweight="bold")
    plt.text(zinc_mean_val - 0.22, plt.gca().get_ylim()[1]*0.6, f"ZINC Mean\n(~{zinc_mean_val:.2f})", color="#ff7f0e", fontweight="bold")
    plt.legend(fontsize=10, loc="upper right")
    plt.savefig(final_img_file, dpi=300)
    plt.close()
else:
    optimal_cutoff = None

# 마스터가 단독 계산 완료한 최종 동적 컷오프 한계치 실수값을 분산 컴퓨팅 환경 내 모든 워커 노드로 동기화 브로드캐스트
optimal_cutoff = comm.bcast(optimal_cutoff, root=0)


# ===================================================================================
# 2패스(Pass 2): 컷오프 이하의 음성 후보군만 필터링하여 저장
# ===================================================================================

# 2패스 선별 데이터 수합을 위해 마스터 노드가 출력용 타겟 CSV 파일의 껍데기를 초기화하고 동적 일감 큐 재설정 개시
if rank == 0:
    print("-> [마스터] 2패스: 동적 컷오프 기반 음성 데이터셋 추출 및 스트리밍 개시")
    if os.path.exists(neg_before_file):
        os.remove(neg_before_file)
        
    task_queue = list(range(len(all_files)))
    finished_workers = 0
    write_buffer = []
    first_chunk = True
    
    # 각 워커가 2패스 내부에서 필터링하여 보내온 데이터(튜플)를 마스터의 메모리 버퍼에 병합
    while finished_workers < size - 1:
        status = MPI.Status()
        local_res = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker = status.Get_source()
        
        if local_res:
            for row in local_res:
                write_buffer.append({"smiles": row[0], "zinc_id": row[1], "mw": row[2], "xlogp": row[3], "max_similarity_to_positive": row[4]})
                
            # 마스터의 메모리 누수를 방지하기 위해 버퍼에 누적 데이터가 5만 개가 쌓일 때마다 디스크 쓰기를 수행 후 비우기 처리
            if len(write_buffer) >= 50000:
                pd.DataFrame(write_buffer).to_csv(neg_before_file, mode="a", header=first_chunk, index=False)
                write_buffer = []
                first_chunk = False
                
        if task_queue:
            file_idx = task_queue.pop(0)
            comm.send(file_idx, dest=worker, tag=3)
        else:
            comm.send(DONE, dest=worker, tag=3)
            finished_workers += 1
            
    # 전체 파일 순회가 완료된 뒤 잔여 버퍼 데이터 처리
    if write_buffer:
        pd.DataFrame(write_buffer).to_csv(neg_before_file, mode="a", header=first_chunk, index=False)
    print("-> [마스터] 필터링 후보군 파일 생성 완료 및 샘플링 단계 진입")

# 워커 노드들이 2패스 구동을 위해 마스터에 첫 번째 파일 배정 요청
else:
    comm.send([], dest=0, tag=2)
    while True:
        file_idx = comm.recv(source=0, tag=3)
        if file_idx == DONE:
            break
            
        file_path = all_files[file_idx]
        local_negatives = []
        
        # [메모리 및 연산 고속화] 원본 ZINC 파일을 다시 청크 단위로 스캔하여 각 컬럼을 Python 기본 리스트 배열로 언패킹
        try:
            for chunk in pd.read_csv(file_path, sep="\t", usecols=["smiles", "zinc_id", "mwt", "logp"], chunksize=50000):
                chunk = chunk.rename(columns={"mwt": "mw", "logp": "xlogp"}).dropna(subset=["smiles"])
                smi_arr = chunk["smiles"].tolist()
                id_arr = chunk["zinc_id"].tolist()
                mw_arr = chunk["mw"].tolist()
                logp_arr = chunk["xlogp"].tolist()
                
                # zip 루프 내에서 대조 유사도를 한 번 더 계산하되, 마스터로부터 전달받은 최적 컷오프 판정
                for s, z_id, m, x in zip(smi_arr, id_arr, mw_arr, logp_arr):
                    mol = Chem.MolFromSmiles(s)
                    if mol is None: continue
                    std_smi = Chem.MolToSmiles(mol)
                    if std_smi in pos_smi_set: continue
                    
                    fp = fp_gen.GetFingerprint(mol)
                    sims = BulkTanimotoSimilarity(fp, ref_fps)
                    max_sim = max(sims) if sims else 0.0
                    
                    # 동적 임계 조건 이하를 통과한 음성 대조군 분자 정보만 원시 튜플 형태로 수집
                    if max_sim <= optimal_cutoff:
                        local_negatives.append((std_smi, z_id, float(m), float(x), max_sim))
                        
        except Exception:
            pass
            
        # 마스터 노드로 역송수신
        comm.send(local_negatives, dest=0, tag=2)


# ===================================================================================
# 11. 최종 균형 샘플링 및 메타데이터 저장 수행
# ===================================================================================

# 0번 마스터 노드가 1:1 학습 밸런스를 맞추기 위해 생성된 1차 음성 파일 전체의 텍스트 원시 라인 개수 카운트
if rank == 0:
    print("-> [마스터] 1:1 균형 샘플링을 위한 스트리밍 샘플러 기동")
    with open(neg_before_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f) - 1
        
    sample_size = min(len(pos_df), total_lines)
    
    # 난수화 시드 키를 무작위 생성하여 저장하고, 추출해야 할 특정 행 번호 인덱스 세트를 메모리 내부에서 무작위 타겟팅 처리
    if sample_size > 0:
        seed_val = random.randint(1, 100000)
        random.seed(seed_val)
        selected_rows = set(sorted(random.sample(range(1, total_lines + 1), sample_size)))
        
        final_neg_list = []
        # 저장해 둔 중간본 파일을 스트리밍 방식으로 읽어 내려가며 선택된 고유 행 번호의 화합물만 로드
        with open(neg_before_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for idx, line in enumerate(f, 1):
                if idx in selected_rows:
                    parts = line.strip().split(",")
                    final_neg_list.append({
                        "smiles": parts[0], "zinc_id": parts[1], "mw": float(parts[2]), "xlogp": float(parts[3]), "max_similarity_to_positive": float(parts[4])
                    })
                    
        # 최종 음성 데이터셋 파일 저장
        final_neg = pd.DataFrame(final_neg_list)
        final_neg.to_csv(final_neg_file, index=False)
        print(f"-> [마스터] 통합 {final_neg_file} 저장 성공")
    else:
        print("-> [마스터] 경고: 샘플링할 수 있는 음성 대조군 후보가 없음")
        seed_val = -1
        
    # 메타데이터 파일 작성 및 저장
    metadata_content = f"""[Task 1 Execution Metadata - KISTI MPI (Two-Pass Optimized Version)]
Tanimoto Cutoff: {optimal_cutoff:.4f}
Total Agrochemical Positives: {len(pos_df)}
Total Selected Negatives: {len(final_neg) if sample_size > 0 else 0}
Random Sampling Seed: {seed_val}
Testing Mode Enabled: {test == 1}
"""
    with open(metadata_file, "w", encoding="utf-8") as meta_file:
        meta_file.write(metadata_content)
        
    # 완결 보고 문구 출력 처리
    print("\n" + "="*50)
    print("      --- KISTI MPI 병렬 처리 완료 보고 ---")
    print(f"-> 전체 선별된 후보군 수: {total_candidates_count}개")
    print(f"-> 최종 저장된 음성 수: {len(final_neg) if sample_size > 0 else 0}개")
    print("="*50)
