"""Log GenMatter artifacts to a Rerun .rrd file."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

from genmatter.custom.config_schema import VizConfig
from genmatter.viz.artifacts import VizArtifacts
from genmatter.viz.colors import (
    blob_mean_rgb_colors,
    distinct_palette,
    hyperblob_covariances_from_blobs,
    point_colors_from_assignments,
    point_colors_from_hyperblob_assignments,
)
from genmatter.viz.geometry import covariance_batch_to_ellipsoids


def export_to_rrd(
    artifacts: VizArtifacts,
    cfg: VizConfig,
    output_path: Path,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Write dense per-frame visualization to ``output_path`` (.rrd)."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vid = artifacts.video_id
    T = artifacts.num_frames
    H, W = artifacts.height, artifacts.width
    max_blobs = int(np.max(artifacts.tracking["n_blobs"]))
    max_hyper = int(np.max(artifacts.tracking["n_hyperblobs"]))
    blob_palette = distinct_palette(max(max_blobs, 1), seed=cfg.feature_pca_seed)
    hyper_palette = distinct_palette(max(max_hyper, 1), seed=cfg.feature_pca_seed + 1)

    rr.init(f"genmatter_{vid}", spawn=False)
    root = vid

    for t in range(T):
        if on_progress is not None:
            on_progress(t + 1, T)
        rr.set_time("frame", sequence=t)

        tr = artifacts.tracking
        n_blobs = int(tr["n_blobs"][t])
        n_hyper = int(tr["n_hyperblobs"][t])

        positions = tr["datapoint_positions"][t]
        assign = tr["blob_assignments"][t]
        rgb_flat = artifacts.colors[t].reshape(-1, 3)

        # 2D RGB camera
        bgr = cv2.imread(str(artifacts.rgb_frame_paths[t]))
        if bgr is not None:
            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"{root}/camera/rgb", rr.Image(rgb_img))

        if cfg.include_depth:
            depth = artifacts.points_3d[t, :, :, 2].astype(np.float32)
            rr.log(
                f"{root}/conditioning/depth",
                rr.DepthImage(depth, meter=1.0),
            )

        if cfg.include_flow:
            flow = artifacts.motion_vectors_3d[t].reshape(-1, 3) * cfg.flow_arrow_scale
            rr.log(
                f"{root}/conditioning/flow",
                rr.Arrows3D(
                    origins=positions,
                    vectors=flow,
                    colors=[100, 180, 255],
                    radii=0.008,
                ),
            )

        rr.log(
            f"{root}/points/rgb",
            rr.Points3D(
                positions,
                colors=rgb_flat,
                radii=cfg.point_radius,
            ),
        )

        rr.log(
            f"{root}/points/features_pca",
            rr.Points3D(
                positions,
                colors=artifacts.feature_rgb[t],
                radii=cfg.point_radius,
            ),
        )

        if cfg.include_assignment_colors:
            assign_colors = point_colors_from_assignments(assign, n_blobs, blob_palette)
            rr.log(
                f"{root}/points/by_assignment",
                rr.Points3D(
                    positions,
                    colors=assign_colors,
                    radii=cfg.point_radius,
                ),
            )

        if cfg.include_hyperblob_assignment_colors and n_blobs > 0:
            hb_assign_per_blob = tr["hyperblob_assignments"][t, :n_blobs]
            hb_point_colors = point_colors_from_hyperblob_assignments(
                assign,
                hb_assign_per_blob,
                n_blobs,
                n_hyper,
                hyper_palette,
            )
            rr.log(
                f"{root}/points/by_hyperblob_assignment",
                rr.Points3D(
                    positions,
                    colors=hb_point_colors,
                    radii=cfg.point_radius,
                ),
            )

        if n_blobs > 0:
            blob_means = tr["blob_means"][t, :n_blobs]
            blob_covs = tr["blob_covs"][t, :n_blobs]
            half_sizes, quats = covariance_batch_to_ellipsoids(
                blob_covs, sigma_scale=cfg.ellipsoid_sigma_scale
            )
            id_colors = blob_palette[:n_blobs]

            rr.log(
                f"{root}/blobs/by_identity",
                rr.Ellipsoids3D(
                    centers=blob_means,
                    half_sizes=half_sizes,
                    quaternions=quats,
                    colors=id_colors,
                ),
            )

            mean_rgb = blob_mean_rgb_colors(assign, rgb_flat, n_blobs)
            rr.log(
                f"{root}/blobs/by_mean_rgb",
                rr.Ellipsoids3D(
                    centers=blob_means,
                    half_sizes=half_sizes,
                    quaternions=quats,
                    colors=mean_rgb,
                ),
            )

        if n_hyper > 0:
            hb_means = tr["hyperblob_means"][t, :n_hyper]
            hb_assign = tr["hyperblob_assignments"][t, :n_blobs]
            hb_covs = hyperblob_covariances_from_blobs(
                hb_assign, tr["blob_covs"][t, :n_blobs], n_hyper
            )
            hb_half, hb_quats = covariance_batch_to_ellipsoids(
                hb_covs, sigma_scale=cfg.ellipsoid_sigma_scale
            )
            hb_colors = hyper_palette[:n_hyper]
            rr.log(
                f"{root}/hyperblobs/ellipsoids",
                rr.Ellipsoids3D(
                    centers=hb_means,
                    half_sizes=hb_half,
                    quaternions=hb_quats,
                    colors=hb_colors,
                ),
            )

    rr.save(str(output_path))
    return output_path.stat().st_size
