"""Deep-learning detector: dataset, models, training, inference.

Scored through the SAME `src/eval` module and the SAME peak extraction as CFAR — see
docs/dl_design.md for the design record and the fairness requirements.
"""

from dl.dataset import RDDataset, normalize_rd, project_root
from dl.models import TinyCNN, count_params, measure_receptive_field
from dl.train import make_loss, overfit_one_batch, train
from dl.infer import (
    cfar_valid_mask,
    collect_split,
    dl_roc,
    evaluate_at_threshold,
    predict_heatmap,
    predict_points,
)

__all__ = [
    "RDDataset", "normalize_rd", "project_root",
    "TinyCNN", "count_params", "measure_receptive_field",
    "make_loss", "overfit_one_batch", "train",
    "cfar_valid_mask", "collect_split", "dl_roc", "evaluate_at_threshold",
    "predict_heatmap", "predict_points",
]
