"""Classical CFAR detection (CA / GO / OS) on Range-Doppler maps."""

from cfar.cfar import CfarConfig, cfar_ratio_map
from cfar.detect import extract_detections, all_peaks

__all__ = ["CfarConfig", "cfar_ratio_map", "extract_detections", "all_peaks"]
