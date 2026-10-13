"""One causal inference pass, with no access to subsequent annotations."""
import platform
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from .data import (Capture, fingerprint, implementation_fingerprint, fit_frame,
                   iter_frames, labels_from_mask, select_foreground, rate, ROOT)
from .geometry import (align_depth, correspondence_confidence, depth_to_xyz,
                       measured_motion, rigid_alignment, warp_previous, sharpen_object_depth)


def sample_indices(labels,number=2925,foreground_fraction=.65,seed=42):
    """Deterministic stratified samples; give small objects all available pixels."""
    rng = np.random.default_rng(seed)
    flat = labels.reshape(-1)
    fg,bg = np.flatnonzero(flat>0),np.flatnonzero(flat==0)
    nf = min(len(fg),round(number*foreground_fraction))
    nb = min(len(bg),number-nf)
    nf = min(len(fg),number-nb)
    result = np.r_[rng.choice(fg,nf,replace=False),rng.choice(bg,nb,replace=False)]
    if len(result) != number:
        raise ValueError('Tracking grid has fewer pixels than required samples')
    # A global foreground quota alone can omit a very small instance. Replace
    # a background sample only when an identity would otherwise be absent.
    for label in np.setdiff1d(np.unique(flat),np.unique(flat[result])):
        if label<=0:
            continue
        replace=np.flatnonzero(flat[result]==0)
        if not len(replace):
            raise ValueError('Sample budget cannot represent all initial identities')
        result[replace[-1]]=rng.choice(np.flatnonzero(flat==label))
    return np.sort(result)


def seed_coordinates(labels,stride):
    """Keep tiny seed instances that fall between regular sampling-grid points.

    Only omitted instances alter the regular grid. A nearby background slot is
    assigned a real pixel of the missing object; its geometry, RGB and feature
    interpolation are sampled at that pixel rather than relabelling background.
    """
    h,w=labels.shape
    yy,xx=np.meshgrid(np.arange(stride//2,h,stride),np.arange(stride//2,w,stride),indexing='ij')
    for label in np.setdiff1d(np.unique(labels),np.unique(labels[yy,xx])):
        if label<=0:
            continue
        points=np.column_stack(np.where(labels==label))
        point=points[np.argmin(np.linalg.norm(points-points.mean(0),axis=1))]
        available=np.column_stack(np.where(labels[yy,xx]==0))
        if not len(available):
            raise ValueError('Initial seed needs a background sampling slot for tiny instances')
        locations=np.column_stack([yy[available[:,0],available[:,1]],xx[available[:,0],available[:,1]]])
        slot=available[np.argmin(np.linalg.norm(locations-point,axis=1))]
        yy[tuple(slot)],xx[tuple(slot)]=point
    return yy,xx


def predict_particles(means,blob_ids,previous_xyz,xyz,previous_labels,warped_labels,
                      forward,reliable,rotation,translation,velocity,covs=None,
                      current_rotation=None,current_translation=None):
    """Use reliable image motion without copying depth from an occluding surface.

    A visible pixel can share an object's ID with an occluded particle (a cloth
    fold, limb or back surface). Its flow is not that particle's flow. Checking
    3D support prevents these particles from repeatedly jumping toward the camera.
    An uncertain surface retains image-plane flow and its own object-predicted
    depth; an invalid image correspondence falls back to robust object motion.
    """
    camera = (means-translation)@rotation
    uv = camera[:,:2]*500/np.maximum(camera[:,2:3],.01)+[320,180]
    mx,my = uv[:,0].astype(np.float32)[:,None],uv[:,1].astype(np.float32)[:,None]
    sample = lambda a,x,y,nearest=False: cv2.remap(a.astype(np.float32),x,y,
        cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(means),-1)
    motion = sample(forward.transpose(1,2,0),mx,my)
    nx,ny = mx+motion[:,0:1],my+motion[:,1:2]
    old = sample(previous_xyz,mx,my)
    new = sample(xyz,nx,ny)
    old_ids = sample(previous_labels,mx,my,True)[:,0].astype(int)
    ok = (sample(reliable,nx,ny,True)[:,0]>.5)&(old_ids==blob_ids)
    ok &= sample(warped_labels,nx,ny,True)[:,0].astype(int)==blob_ids
    # Require a full bilinear footprint: zero padding outside the image must
    # never act as an observed 3D point at the camera origin.
    ok &= (uv[:,0]>=0)&(uv[:,0]<=638)&(uv[:,1]>=0)&(uv[:,1]<=358)
    ok &= (nx[:,0]>=0)&(nx[:,0]<=638)&(ny[:,0]>=0)&(ny[:,0]<=358)
    image_ok=ok.copy()
    if covs is not None:
        residual=old-means
        inverse=np.linalg.inv(np.asarray(covs)+np.eye(3)[None]*.0016)
        distance=np.einsum('ni,nij,nj->n',residual,inverse,residual)
        ok &= distance <= 3.5**2
    shift = new-old
    current_rotation=rotation if current_rotation is None else current_rotation
    current_translation=translation if current_translation is None else current_translation
    for label in np.unique(blob_ids):
        take = reliable&(warped_labels==label)
        fallback = np.median(velocity[take],axis=0) if take.any() else np.zeros(3)
        shift[(blob_ids==label)&~ok] = fallback
        # Uncertain depth does not invalidate reliable image motion. Carry the
        # particle along its observed image ray at its own predicted depth,
        # rather than replacing its depth with a different visible surface.
        # Discarding both components freezes small, rapidly moving objects.
        planar=(blob_ids==label)&image_ok&~ok
        if planar.any():
            predicted_camera=(means[planar]+fallback-current_translation)@current_rotation
            depth=np.maximum(predicted_camera[:,2],.05)
            rays=np.column_stack([(nx[planar,0]-320)/500,(ny[planar,0]-180)/500,np.ones(planar.sum())])
            transported=rays*depth[:,None]@current_rotation.T+current_translation
            shift[planar]=transported-means[planar]
    length = np.linalg.norm(shift,axis=1,keepdims=True)
    shift *= np.minimum(1.,.5/(length+1e-8))
    return (means+shift).astype(np.float32),shift.astype(np.float32)


def guided_probabilities(probabilities, bgr):
    """Edge-aware interpolation of posterior probabilities, never label colors."""
    h, w = bgr.shape[:2]
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    box = lambda a: cv2.boxFilter(a, -1, (13,13), normalize=True)
    mean = box(guide)
    variance = box(guide*guide)-mean*mean
    results = []
    for k in range(probabilities.shape[-1]):
        p = cv2.resize(probabilities[...,k], (w,h), interpolation=cv2.INTER_LINEAR)
        mp = box(p)
        a = (box(guide*p)-mean*mp)/(variance+.002)
        b = mp-a*mean
        results.append(np.clip(box(a)*guide+box(b),0,1))
    return np.stack(results,-1)


def track(clip, config, output, max_frames=-1, models=None, save_states=True):
    tracking_start = time.perf_counter()
    # Heavy imports belong here, never in render/help/metrics code paths.
    import jax
    import jax.numpy as jnp
    import genmatter_rt as rt
    from .models import Models
    from .seed import balanced_seed
    from .inference import AppearanceMemory, posterior, refine_geometry,appearance_estimate
    reused_models = models is not None
    models = models or Models(config)
    models.reset()
    model_load_seconds = models.load_seconds
    model_construction_seconds = 0. if reused_models else model_load_seconds
    attributed_model_load_seconds = model_load_seconds if models.clips_tracked == 0 else 0.
    models.clips_tracked += 1
    if clip.source.is_file():
        cap = cv2.VideoCapture(str(clip.source))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if source_fps > 0:
            clip.fps = source_fps
    initial = labels_from_mask(clip.mask)
    if clip.provenance.startswith('DAVIS'):
        initial[initial == 255] = 0  # DAVIS void pixels are not an object identity
    initial, original_ids = select_foreground(initial, clip.foreground_ids)
    seed_mask, _ = fit_frame(initial, nearest=True)
    manifest = dict(clip=clip.name, source=str(clip.source.resolve()),
                    source_sha256=fingerprint(clip.source), initial_mask=str(clip.mask.resolve()),
                    initial_mask_sha256=fingerprint(clip.mask), mask_provenance=clip.provenance,
                    foreground_ids=original_ids, config=config, source_fps=clip.fps,
                    playback_rate_source='video container average' if clip.source.is_file() else 'configured image-sequence rate',
                    working_resolution=[640,360], depth_units='relative, not metric',
                    input_intrinsics=[[500.,0.,320.],[0.,500.,180.],[0.,0.,1.]],
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                    implementation_sha256=implementation_fingerprint(),
                    inference_implementation_sha256=implementation_fingerprint('inference'),
                    runtime=dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,
                                 torch=models.torch.__version__,cuda=models.torch.version.cuda,
                                 gpu=models.torch.cuda.get_device_name()),
                    model_versions=models.versions,
                    reused_models=reused_models,
                    inference='GenMatter Gibbs + object-conditioned MAP and robust posterior geometry')
    capture = Capture(output, manifest) if save_states else None
    cfg = rt.load_yaml_hypers(ROOT/'configs/streaming_render_v2.yaml')
    cfg['tracking'].update(use_calibrated_priors=False, init_gibbs_sweeps=0)
    cfg['tracking']['hyperparams']['sigma_F'] = config['feature_variance']
    stride = config['sampling_stride']
    offset = stride//2
    grid_slice = (slice(offset,None,stride),slice(offset,None,stride))
    grid_shape = (360//stride,640//stride)
    key = jax.random.key(config['seed'])
    previous_depth = previous_xyz = previous_labels = None
    world_r, world_t = np.eye(3), np.zeros(3)
    bounds = None
    times, feature_times, tracker_times, io_times, correspondence_rates = [], [], [], [], []
    warmup = []
    n = 0
    process_start = time.perf_counter()
    for n, (frame,source_time) in enumerate(iter_frames(clip.source,max_frames,with_time=True,fps=clip.fps), 1):
        index = n-1
        start = time.perf_counter()
        bgr, valid_image = fit_frame(frame)
        raw, forward, backward, features, feature_seconds = models.extract(bgr)
        tracker_start = time.perf_counter()
        reliable = correspondence_confidence(forward,backward) & valid_image
        if previous_labels is None:
            warped_labels = seed_mask
        else:
            warped_labels, _ = warp_previous(previous_labels,backward,nearest=True)
            warped_labels = warped_labels.astype(np.int32)
        depth, bounds = align_depth(raw,previous_depth,backward,reliable,warped_labels == 0,bounds)
        depth = sharpen_object_depth(depth,warped_labels,config.get('boundary_depth_radius',3))
        camera_xyz = depth_to_xyz(depth,bounds)
        previous_r,previous_t = world_r.copy(),world_t.copy()
        if previous_xyz is not None:
            # Previous xyz is in world coordinates. Fit camera -> world using
            # reliable background correspondences, excluding moving objects.
            old_world, _ = warp_previous(previous_xyz,backward)
            predicted_world = camera_xyz@world_r.T+world_t
            dr, dt = rigid_alignment(predicted_world,old_world,reliable & (warped_labels == 0))
            world_r, world_t = dr@world_r, dr@world_t+dt
        xyz = (camera_xyz@world_r.T+world_t).astype(np.float32)
        velocity = measured_motion(xyz,previous_xyz,backward,reliable)
        grid_xyz = xyz[grid_slice].reshape(-1,3)
        grid_velocity = velocity[grid_slice].reshape(-1,3)
        grid_features = features.reshape(-1,32)
        grid_rgb = bgr[grid_slice].reshape(-1,3)
        grid_labels=warped_labels[grid_slice]
        if index==0:
            sy,sx=seed_coordinates(seed_mask,stride)
            grid_xyz=xyz[sy,sx].reshape(-1,3)
            grid_velocity=velocity[sy,sx].reshape(-1,3)
            grid_rgb=bgr[sy,sx].reshape(-1,3)
            grid_labels=seed_mask[sy,sx]
            grid_features=cv2.remap(features,((sx-offset)/stride).astype(np.float32),
                ((sy-offset)/stride).astype(np.float32),cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE).reshape(-1,32)
        indices = sample_indices(grid_labels,seed=config['seed'])
        if index == 0:
            grid_seed = grid_labels.reshape(-1)
            missing = set(range(len(original_ids)+1))-set(np.unique(grid_seed[indices]))
            if missing:
                raise ValueError(f'Initial objects {missing} are smaller than the tracking grid; use a closer crop')
            choice, blob_ids = balanced_seed(grid_xyz[indices],grid_velocity[indices],grid_seed[indices],
                config['particles'],config['foreground_fraction'],config['seed'])
            state,key = rt.init_state(grid_xyz[indices],grid_velocity[indices],grid_features[indices],key,
                                      yaml_cfg=cfg,seed_choice_map=choice,num_warmup_sweeps=0)
            memory = AppearanceMemory(grid_features,grid_seed)
            blob_anchor = state.blobs_state.blob_features
            covariance_anchor = state.blobs_state.blob_covs
            hb_anchor = state.hyperblobs_state.hyperblob_features
            color_memory = np.zeros((len(blob_ids),3),np.float32)
        else:
            predicted_means,observed_velocity = predict_particles(
                np.asarray(state.blobs_state.blob_means),blob_ids,previous_xyz,xyz,previous_labels,
                warped_labels,forward,reliable,previous_r,previous_t,velocity,
                np.asarray(state.blobs_state.blob_covs),world_r,world_t)
            previous_covs = state.blobs_state.blob_covs
            state = state.replace({'blobs_state':{'blob_vel_means':jnp.asarray(observed_velocity)}})
        evidence = memory.logits(grid_features)
        arguments = (jnp.asarray(grid_xyz),jnp.asarray(grid_features),jnp.asarray(evidence),
                     jnp.asarray(grid_labels.reshape(-1)),
                     jnp.asarray(reliable[grid_slice].reshape(-1)),jnp.asarray(blob_ids),
                     config['memory_weight'],config['persistence'],config['feature_weight'])
        if index:
            predicted = state.replace({'blobs_state':{'blob_means':jnp.asarray(predicted_means)}})
            proposal,_,_ = posterior(predicted,*arguments)
            state = state.replace({'datapoints_state':{'blob_assignments':proposal[indices]}})
            state,key = rt.step_multi_sweep(state,grid_xyz[indices],grid_velocity[indices],grid_features[indices],key,
                num_sweeps=config['sweeps'],feature_aware_final=True,final_outlier=False,
                freeze_hyperblob_assignment=True,blob_means_updates=1,
                blob_feat_anchor=blob_anchor,hb_feat_anchor=hb_anchor,
                feature_update_damping=config['appearance_update'],
                use_robust_mean=True,mean_update_robust_delta=config['huber_delta'])
        assignments, probabilities,responsibilities = posterior(state,*arguments)
        if index:
            state = refine_geometry(state,assignments[indices],predicted_means,previous_covs,
                                    jnp.asarray(blob_ids),config['huber_delta'],
                                    config.get('mean_prior_count',1.5),config.get('covariance_prior_count',4.),
                                    config.get('particle_weight_prior',3.),covariance_anchor,
                                    config.get('covariance_growth_limit',9.),responsibilities[indices])
            assignments, probabilities,responsibilities = posterior(state,*arguments)
            state = state.replace({'datapoints_state':{'blob_assignments':assignments[indices]}})
        assignments = np.asarray(assignments)
        probabilities = np.asarray(probabilities).reshape(*grid_shape,-1)
        labels = guided_probabilities(probabilities,bgr).argmax(-1).astype(np.uint16)
        if not index:
            labels = seed_mask.astype(np.uint16)  # the explicitly provided initial observation
        labels[~valid_image] = 0
        means,covs = rt.extract_blob_means_and_covs(state)
        weights = np.asarray(state.blobs_state.blob_weights)
        if not (np.isfinite(means).all() and np.isfinite(covs).all()):
            raise RuntimeError(f'Non-finite posterior at frame {index}')
        colors,reconstructed = appearance_estimate(responsibilities,jnp.asarray(grid_rgb,dtype=jnp.float32),
                                                  jnp.asarray(color_memory),1. if not index else .2)
        color_memory = np.asarray(colors)
        support = np.asarray(responsibilities[indices].sum(0))
        reconstruction = cv2.resize(np.asarray(reconstructed).reshape(*grid_shape,3), (640,360),
                                    interpolation=cv2.INTER_LINEAR).clip(0,255).astype(np.uint8)
        jax.block_until_ready(state)
        tracker_seconds = time.perf_counter()-tracker_start
        seconds = time.perf_counter()-start
        if index < config['warmup_frames']:
            warmup.append(seconds)
        else:
            times.append(seconds); feature_times.append(feature_seconds); tracker_times.append(tracker_seconds)
        correspondence_rates.append(float(reliable.mean()))
        io_start = time.perf_counter()
        if capture:
            capture.write(index,bgr=bgr,labels=labels,reconstruction=reconstruction,
                          means=means,covs=covs,weights=weights,colors=color_memory.astype(np.uint8),
                          support=support.astype(np.float32),
                          object_ids=blob_ids,world_rotation=world_r,world_translation=world_t,
                          backward_grid=(backward[:,4::8,4::8]/8).astype(np.float16),
                          reliable_grid=reliable[4::8,4::8],
                          correspondence=float(reliable.mean()),source_time=source_time,playback_time=index/clip.fps,
                          inference_seconds=seconds,feature_seconds=feature_seconds,tracker_seconds=tracker_seconds)
        io_times.append(time.perf_counter()-io_start)
        previous_depth,previous_xyz,previous_labels = depth,xyz,labels
        if n % 10 == 0 or n == 1:
            print(f'{clip.name}: frame {n}, inference {rate(times) or 0:.2f} frames/s (warmup excluded)',flush=True)
    if n == 0:
        raise ValueError('Input contains no frames')
    summary = dict(frames=n, initialization_seconds=sum(warmup), model_load_seconds=model_load_seconds,
        attributed_model_load_seconds=attributed_model_load_seconds,
        timed_frames=len(times),inference_fps=rate(times),feature_fps=rate(feature_times),tracker_fps=rate(tracker_times),
        inference_ms_p50=float(np.percentile(times,50)*1000) if times else None,
        inference_ms_p95=float(np.percentile(times,95)*1000) if times else None,
        feature_seconds=sum(feature_times),tracker_seconds=sum(tracker_times),
        inference_seconds=sum(times),capture_io_seconds=sum(io_times),
        setup_seconds=process_start-tracking_start-model_construction_seconds,
        tracking_wall_seconds=time.perf_counter()-tracking_start-model_construction_seconds,
        correspondence_mean=float(np.mean(correspondence_rates)),
        timing_definition='completed frames / sum synchronized durations; initial warmup frames excluded; source decoding, rendering and encoding separate')
    if capture:
        capture.finish(**summary)
    return {**manifest,**summary}
