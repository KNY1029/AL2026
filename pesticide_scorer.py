# -*- coding: utf-8 -*-
# pesticide_scorer.py - 공통 스코어링 계산 모듈을 포함하는 파일이다 (노트북에서 자동 생성한다).
import os
import sys
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

CONFIG_FILE = "best_score_model_config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"{CONFIG_FILE} 최적 설정 파일을 찾을 수 없다. 노트북을 먼저 실행해야 한다.")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

# 최적 가중치 바인딩
w_p = config["w_Property"]
w_s = config["w_Structure"]
best_w_scf = config["w_Scaffold"]

# 패턴 데이터 컴파일
SCAFFOLD_PATTERNS = []
for smi in config["scaffold_smiles"]:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        smarts = Chem.MolToSmarts(mol)
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            SCAFFOLD_PATTERNS.append(pat)
    except:
        pass

RESIDUE_PATTERNS = []
for smi in config["residue_smarts"]:
    try:
        pat = Chem.MolFromSmarts(smi)
        if pat is not None:
            RESIDUE_PATTERNS.append(pat)
    except:
        pass

SCAFFOLD_WEIGHTS = config["scaffold_weights"]
RESIDUE_WEIGHTS = config["residue_weights"]
HIST_MODELS = config["hist_models"]

# 추가 전역 변수 정의
SELECTED_PROPS = ["mw", "xlogp", "rotbonds", "aromatic_rings"]
NUM_BINS = 30
hist_models = HIST_MODELS

PROPERTY_FUNCS = {
    'mw': lambda m: Descriptors.MolWt(m),
    'xlogp': lambda m: Crippen.MolLogP(m),
    'rotbonds': lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
    'aromatic_rings': lambda m: rdMolDescriptors.CalcNumAromaticRings(m)
}

def get_property_score_hist(mol):
    try:
        props = {
            "mw": Descriptors.MolWt(mol),
            "xlogp": Descriptors.MolLogP(mol),
            "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
    except Exception:
        return 0.0  # 손상된 분자는 0점 처리

    total_score = 0.0
    for col in SELECTED_PROPS:
        val = props[col]
        bins = hist_models["bins"][col]

        # 입력된 분자의 물성값이 30개 막대 중 몇 번째 막대에 속하는지 인덱스 탐색
        bin_idx = np.digitize(val, bins) - 1
        bin_idx = max(0, min(bin_idx, NUM_BINS - 1))  # 인덱스 범위 초과 방지

        p_pos = hist_models["pos"][col][bin_idx]
        p_neg = hist_models["neg"][col][bin_idx]

        # 베이즈 확률(PPV) 수식: P(Pos) / [ P(Pos) + P(Neg) ]
        # (데이터가 아예 없는 빈 구간을 지날 때 0으로 나누어지는 에러를 방지하기 위해 분모 하한선 1e-10 설정)
        if (p_pos + p_neg) > 1e-10:
            score = p_pos / (p_pos + p_neg)
        else:
            score = 0.0
        total_score += score

    # 4개 속성 PPV 점수의 평균값을 최종 물성 점수로 반환 (0.0 ~ 1.0)
    return total_score / len(SELECTED_PROPS)

def get_scaffold_score_only(mol, patterns=SCAFFOLD_PATTERNS, weights=SCAFFOLD_WEIGHTS):
    scf = MurckoScaffold.GetScaffoldForMol(mol)
    if scf is None:
        return 0.05

    scf_atoms = scf.GetNumHeavyAtoms()
    valid_weights = []

    for p, w in zip(patterns, weights):
        try:
            p_atoms = p.GetNumHeavyAtoms()
            # 부분 일치를 만족하되, '본체 뼈대와 족보 패턴의 중원자 수 차이가 5개 이하'일 때만 승인
            # (거대 ZINC 분자가 작은 농약 고리를 기생 매칭으로 털어가는 현상 차단)
            if scf.HasSubstructMatch(p) and abs(scf_atoms - p_atoms) <= 5:
                valid_weights.append(w)
        except Exception:
            continue

    # 체급 잠금 필터를 통과한 족보 패턴이 하나도 없으면 소프트 바닥값(0.10) 부여
    if not valid_weights:
        return 0.10

    # 본인이 가진 가장 특이성이 높은 뼈대 PPV를 온전히 점수로 반영
    best_ppv = max(valid_weights)
    return best_ppv

def get_residue_score_only(
    mol, patterns=RESIDUE_PATTERNS, weights=RESIDUE_WEIGHTS
):
    scf = MurckoScaffold.GetScaffoldForMol(mol)
    sidechains = Chem.ReplaceCore(mol, scf)
    tgt = sidechains if sidechains is not None else mol

    matched_ppv_sum = 0.0
    for p, w in zip(patterns, weights):
        try:
            if tgt.HasSubstructMatch(p):
                matched_ppv_sum += w
        except Exception:
            continue

    try:
        n_scf = scf.GetNumHeavyAtoms() if scf is not None else 0
        n_side = max(0, mol.GetNumHeavyAtoms() - n_scf)
        ratio = n_side / (mol.GetNumHeavyAtoms() + 1e-9)
    except Exception:
        ratio = 0.5

    x_res = matched_ppv_sum * ratio
    return 1.0 / (1.0 + np.exp(-5.0 * (x_res - 0.25)))


# 4. 종합 구조 스코어링 함수를 정의한다 (골격 Scaffold와 잔기 Residue를 합산한다).
def get_scaffold_score(mol):
    scaf = get_scaffold_score_only(mol, patterns=SCAFFOLD_PATTERNS, weights=SCAFFOLD_WEIGHTS)
    res = get_residue_score_only(mol, patterns=RESIDUE_PATTERNS, weights=RESIDUE_WEIGHTS)
    return best_w_scf * scaf + (1.0 - best_w_scf) * res

# 5. 최종 리워드 스코어 함수를 정의한다 (지문 유사도는 배제한다).
def calculate_reward_score(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -1e9
    p_score = get_property_score_hist(mol)
    struct_score = get_scaffold_score(mol)
    final_score = w_p * p_score + w_s * struct_score
    return final_score
