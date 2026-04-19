#!/usr/bin/env python3
"""
运行TMDWF拟合的脚本，基于notebook配置但更新为任务要求。
"""

import sys
from pathlib import Path

# 添加src目录到路径
REPO_ROOT = Path.cwd()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lqcd_analysis.notebook_workflows import (
    run_tmdwf_fit_from_notebook,
    validate_tmdwf_notebook_config,
    pretty_print_config,
)

# 更新后的配置，匹配任务要求
workflow_config = {
    # Data settings
    "title_pattern": "l64c64a076_m140_fit_pz*",
    "ns": 64,
    "nt": 64,
    "lattice_spacing_fm": 0.076,
    "pzlist": [5, 6, 7, 8],
    "gmlist": ["T5"],  # use ["Z5"] for gamma_z gamma_5
    "etalist": ["eta0"],
    "Tdirlist": ["b_X", "b_Y"],
    "bTlist": [bT for bT in range(0, 21)],
    "bzlist": [bz for bz in range(0, 21)],
    "qtmdwf_h5": "/Users/xiang/Desktop/docs/0-2025/projects/3-CSkernel/data/l64c64a076_m140/comb_qTMDWF/qTMDWF_CG_1HYP_M140_GSRC_W45_k6_src5_O{gm}.h5",
    "dataset_path_template": "SP/{gm}/PX0PY0PZ{pz}/{Tdir}/{eta}/bT{bT}/bz{bz}",
    "two_point_fit_root": "/Users/xiang/Desktop/mycodes_CODEX/local_runs/CSkernel/analysis_l64c64a076_m140_src5/1-c2pt-fit/results_nstate_fit_2state",
    "two_point_fit_window_by_pz": {5: [4, 13], 6: [4, 13], 7: [4, 13], 8: [4, 13]},
    "c2pt": "/Users/xiang/Desktop/docs/0-2025/projects/3-CSkernel/data/l64c64a076_m140/comb_c2pt_csv/c2pt_5_5_k6_pz*_real.csv",
    "fold_t": "periodic",
    "tsrange": [0, 20],

    # Fit settings
    "fit_target": "ratio",
    "fit_component": "both",
    "nstates": [2],
    "binsize": 10,
    "bootstrap_samples": 200,
    "bootstrap_size": 200,
    "seed": 2026,
    "fit_window": {5: [6, 12], 6: [6, 12], 7: [6, 12], 8: [6, 12]},
    "plot": True,

    # Output settings
    "results_dir": "/Users/xiang/Desktop/mycodes_CODEX/local_runs/CSkernel/analysis_l64c64a076_m140_src5/2-bm/results_fit_nst2",
}

print("配置验证...")
parsed = validate_tmdwf_notebook_config(workflow_config)
print(f"验证通过: {parsed.title_pattern}")

print("\n配置详情:")
print(pretty_print_config(workflow_config))

print("\n开始运行TMDWF拟合...")
outputs = run_tmdwf_fit_from_notebook(workflow_config)

print(f"\n完成! 生成 {len(outputs)} 个输出")
for i, output in enumerate(outputs):
    print(f"{i+1}. {output}")