"""Temporal display state, separate from the immutable inference posterior.

Observation count is not existence: a particle can stop receiving image samples
when another surface covers it. Visibility below is a display policy, not an
inferred probability of physical existence or an occlusion benchmark.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .geometry import orbit_camera, Camera, look_at


def fixed_scene_camera(means, covs, alpha, width=1104, height=1130, camera_positions=None,
                       yaw=28., pitch=20., focal=650.):
    """Fit the complete visible clip once; no following, zoom, or camera animation.

    Covariance support functions fit the actual ellipsoids against the frustum
    planes, including scene context, without wasting space on empty box corners.
    This is offline presentation fitting, not an inference input.
    """
    selected = alpha > .005
    points, covariance = means[selected], covs[selected]
    if not len(points):
        points, covariance = means.reshape(-1, 3), covs.reshape(-1, 3, 3)
    if camera_positions is not None:
        points=np.concatenate([points,camera_positions])
        covariance=np.concatenate([covariance,np.tile(np.eye(3)[None]*.01,(len(camera_positions),1,1))])
    rotation = orbit_camera(np.zeros(3), yaw, pitch, 1., 650., width, height).rotation
    local = points@rotation.T
    local_cov = np.einsum('ij,njk,lk->nil', rotation, covariance, rotation)
    limits = np.array([width, height])*.45/focal
    bounds = []
    # Fit the true support functions of the ellipsoids against four side
    # planes, rather than fitting empty corners of a large 3D bounding box.
    for axis, limit in enumerate(limits):
        minus, plus = np.eye(3)[axis].copy(), np.eye(3)[axis].copy()
        minus[2], plus[2] = -limit, limit
        support = lambda v: 1.8*np.sqrt(np.maximum(np.einsum('i,nij,j->n',v,local_cov,v),1e-8))
        lower = np.max(local@minus+support(minus))
        upper = np.min(local@plus-support(plus))
        bounds.append((lower, upper))
    z = min((upper-lower)/(2*limit) for (lower,upper),limit in zip(bounds,limits))
    z = min(z, float(np.min(local[:,2]-1.8*np.sqrt(local_cov[:,2,2])))-.1)
    xy = [(lower+upper)/2 for lower,upper in bounds]
    return Camera(np.array([*xy,z])@rotation, rotation, focal, width, height)


def following_scene_camera(means,covs,alpha,ids,targets,fps,camera_points):
    """A nearly frontal, object-centered overview with a smooth distance track.

    Fading particles remain rendered but cannot force the overview to retreat.
    Only presentation framing uses the confidence cutoff. Synthesis poses and
    particle geometry are unchanged.
    """
    width,height,focal=2440,1000,10000.
    rotation=orbit_camera(np.zeros(3),8.,16.,1.,focal,width,height).rotation
    foreground=ids>0
    if not foreground.any(): foreground=np.ones(len(ids),bool)
    selected=[];centers=[]
    for i,points in enumerate(means):
        use=foreground & (alpha[i]>.35)
        selected.append(use)
        if use.sum()<3:
            centers.append(centers[-1] if centers else targets[i])
        else:
            lo,hi=np.percentile(points[use],[2,98],axis=0)
            centers.append((lo+hi)/2)
    centers=gaussian_filter1d(np.asarray(centers),max(fps*.30,.5),axis=0,mode='nearest')
    camera_covs=np.broadcast_to(np.eye(3)*.003,(len(camera_points),3,3))
    distances=[]
    for i,use in enumerate(selected):
        if use.sum()<3:
            use=foreground
        points=np.concatenate([means[i,use],camera_points])
        covariance=np.concatenate([covs[i,use],camera_covs])
        local=(points-centers[i])@rotation.T
        local_cov=np.einsum('ij,njk,lk->nil',rotation,covariance,rotation)
        required=[]
        for axis,limit in enumerate(np.array([width,height])*.42/focal):
            for sign in (-1,1):
                plane=np.eye(3)[axis]*sign;plane[2]=-limit
                support=1.8*np.sqrt(np.maximum(np.einsum('i,nij,j->n',plane,local_cov,plane),1e-8))
                required.append(float(np.max(local@plane+support))/limit)
        required.append(float(np.max(-local[:,2]+1.8*np.sqrt(local_cov[:,2,2])))+.15)
        distances.append(max(required))
    distances=np.asarray(distances)
    smooth=gaussian_filter1d(distances,max(fps*.35,.5),mode='nearest')
    # One constant safety margin covers smoothing lag without abrupt per-frame
    # distance corrections. Camera names retain space inside the view borders.
    smooth+=max(0.,float(np.max(distances-smooth)))
    positions=centers-smooth[:,None]*rotation[2]
    camera=Camera(positions[len(positions)//2],rotation,focal,width,height)
    return camera,positions,centers


def playback_fps(timestamps,fallback):
    """Rate of decoded presentation times, robust to stale MP4 edit-list metadata.

    Output is constant-rate at the mean decoded cadence. For genuinely variable
    cadence this preserves total duration, not every individual source interval.
    """
    times=np.asarray(timestamps,float)
    intervals=np.diff(times)
    if len(intervals) and np.isfinite(times).all() and np.all(intervals>0):
        return float(len(times)/(times[-1]-times[0]+np.median(intervals)))
    return float(fallback)


def camera_coordinates(frame):
    """Rebase geometry AND its covariances into the observing camera's frame.

    This removes arbitrary accumulated pose noise from the viewer's coordinate
    system. It preserves every source-camera projection and pairwise distance.
    A camera-relative display does not claim world-coordinate stabilization.
    """
    r = np.asarray(frame.get('world_rotation', np.eye(3)))
    t = np.asarray(frame.get('world_translation', np.zeros(3)))
    out = dict(frame)
    out['means'] = (frame['means']-t)@r
    out['covs'] = np.einsum('ji,njk,kl->nil',r,frame['covs'],r)
    out['world_rotation'] = np.eye(3)
    out['world_translation'] = np.zeros(3)
    return out


def visibility_trace(support, fps):
    """Causal support hysteresis with finite memory, keyed by particle identity.

    Hold brief observation gaps (0.20 s), then fade over 0.35 s. Recovery is also
    gradual (0.10 s); no hard threshold deletes an existing particle. Persistent
    failure still disappears, and never-observed particles stay invisible.
    All constants are in seconds, so different source rates behave consistently.
    """
    support = np.asarray(support,float)
    n,b = support.shape
    alpha = np.zeros((n,b),np.float32)
    missing = np.zeros((n,b),np.float32)
    a = np.clip(support[0]/1.5,0,1)
    age = np.zeros(b)
    rise = 1-np.exp(-1/(fps*.10))
    fall = np.exp(-1/(fps*.35))
    for i,s in enumerate(support):
        observed = s >= .5
        age = np.where(observed,0,age+1/fps)
        target = np.clip(s/1.5,0,1)
        # Low support can mean occlusion or association uncertainty; retain the
        # last supported representation briefly, without declaring it observed.
        a = np.where(observed,a+rise*(target-a),np.where(age<=.20,a,a*fall))
        alpha[i] = a
        missing[i] = age
    return alpha,missing


class Presentation:
    """Reusable, deterministic presentation tracks for video and review sheets."""
    def __init__(self,capture,stabilized=True):
        self.manifest = dict(capture.manifest)
        self.fps = float(capture.manifest.get('source_fps',30.))
        # Keep only particle arrays during the fitting pass, not decoded images.
        fields=('means','covs','object_ids','support','weights','colors')
        from .motion import rollout_rates
        frames=[];timestamps=[];durations=[]
        for i in range(capture.manifest['frames']):
            f=camera_coordinates(capture.read(i))
            frames.append({k:f[k] for k in fields if k in f})
            timestamps.append(float(f.get('source_time',i/self.fps)))
            durations.append(float(f.get('inference_seconds',np.nan)))
        self.fps=playback_fps(timestamps,self.fps)
        self.manifest['source_fps']=self.fps
        self.warmup=int(self.manifest.get('config',{}).get('warmup_frames',3))
        self.rollout_fps=rollout_rates(durations,self.warmup)
        measured=self.rollout_fps[np.isfinite(self.rollout_fps)]
        self.initial_rollout_fps=float(measured[0]) if len(measured) else np.nan
        self.means = np.stack([f['means'] for f in frames])
        self.covs = np.stack([f['covs'] for f in frames])
        self.ids = frames[0]['object_ids']
        if any(not np.array_equal(f['object_ids'],self.ids) for f in frames):
            raise ValueError('Presentation requires persistent particle indices and object identities')
        support = np.stack([f.get('support',f['weights']*2925) for f in frames])
        self.alpha,self.missing = visibility_trace(support,self.fps)
        from .stability import stabilize, diagnostics
        self.stabilized = stabilized
        self.display_means,self.display_covs,self.flow_valid = stabilize(
            self.means,self.covs,support,self.ids,capture.read,self.fps)
        geometry_diagnostics=diagnostics(self.means,self.display_means,self.covs,self.display_covs,self.ids)
        self.colors = np.stack([f['colors'] for f in frames])
        for i in range(1,len(frames)):
            unseen = support[i] < .5
            self.colors[i,unseen] = self.colors[i-1,unseen]
        centers,radii = [],[]
        for i,points in enumerate(self.means):
            use = (self.ids>0)&(support[i]>.5)
            if use.sum()<3 and centers:
                centers.append(centers[-1]);radii.append(radii[-1]);continue
            if use.sum()<3:
                use = self.ids>0
            if not use.any():
                use = np.ones(len(points),bool)
            lo,hi = np.percentile(points[use],[2,98],axis=0)
            centers.append((lo+hi)/2)
            radii.append(np.linalg.norm(hi-lo)/2+.15)
        self.targets = gaussian_filter1d(np.asarray(centers),max(self.fps*.22,.5),axis=0,mode='nearest')
        # Follow a smooth 3D target at fixed orientation/focal length/distance.
        # Freezing target depth at its clip median can put an early foreground
        # behind the viewer when the subject subsequently recedes. Include the
        # smoothing lag in the fixed clip-wide extent, without frame-wise zoom.
        for i,points in enumerate(self.means):
            use=(self.ids>0)&(support[i]>.5)
            if use.any():
                radii[i]=max(radii[i],float(np.percentile(np.linalg.norm(points[use]-self.targets[i],axis=1),98))+.15)
        self.radius = max(float(np.percentile(radii,98)),.25)
        self.diagram_depth=max(float(np.percentile(self.targets[:,2],98)),1.)
        self.surface_spacing=max(self.radius/64,.015)
        # Readout cameras are fitted once for the complete clip. The overview
        # follows the smooth target with constant orientation and focal length.
        foreground=self.ids>0
        self.novel_target=np.median(self.targets,axis=0)
        self.novel_yaw=-8.
        self.novel_yaws=[-8.,8.]
        self.view_selection={}
        self.novel_available=True
        from .render import views
        self.source_camera,novel=views(dict(world_rotation=np.eye(3),world_translation=np.zeros(3)),
                                      self.novel_target,self.novel_yaws)
        self.novel_cameras=novel
        self.novel_camera=novel[0]
        fit_means=np.concatenate([self.means,self.display_means])
        fit_covs=np.concatenate([self.covs,self.display_covs])
        fit_alpha=np.concatenate([self.alpha,self.alpha])
        use=foreground if foreground.any() else np.ones(len(self.ids),bool)
        self.detail_camera=fixed_scene_camera(fit_means[:,use],fit_covs[:,use],fit_alpha[:,use],
                                              560,400,yaw=20.,pitch=14.)
        # Readout selection calls frame() before the final overview is fitted.
        self.scene_camera=self.detail_camera
        self.scene_positions=np.repeat(self.scene_camera.position[None],len(frames),axis=0)
        raw = np.clip(support/1.5,0,1);raw[raw<.08]=0
        fg = self.ids>0
        delta = lambda x: np.abs(np.diff(x[:,fg],axis=0))
        before,after = delta(raw),delta(self.alpha)
        self.diagnostics = dict(coordinate_system='observing-camera relative',
            geometry='causal flow-compensated center and log-covariance display filter (0.14 s); raw captures unchanged',
            geometry_stability=geometry_diagnostics,
            geometry_filter_enabled=stabilized,
            opacity='causal support hysteresis: 0.20 s hold, 0.35 s decay, 0.10 s recovery',
            raw_alpha_jumps_over_half=int((before>.5).sum()),
            display_alpha_jumps_over_half=int((after>.5).sum()),
            raw_mean_alpha_delta=float(before.mean()) if before.size else 0.,
            display_mean_alpha_delta=float(after.mean()) if after.size else 0.,
            camera_target_acceleration_p95=float(np.percentile(np.linalg.norm(np.diff(self.targets,n=2,axis=0),axis=-1),95)) if len(frames)>2 else 0.)
        # Select a fixed, spread-out set of trajectories; never swap IDs per frame.
        candidates = np.flatnonzero(fg & (support[0]>1.5))
        chosen=[]
        if len(candidates):
            chosen=[int(candidates[np.argmax(support[0,candidates])])]
            while len(chosen)<min(8,len(candidates)):
                distance=np.linalg.norm(self.means[0,candidates,None]-self.means[0,chosen][None],axis=-1).min(1)
                chosen.append(int(candidates[np.argmax(distance)]))
        self.trail_ids = np.array(chosen,int)
        self.ground_available=False
        if 'labels' in capture.read(0) and 'reconstruction' in capture.read(0):
            from .novel import choose_offset_views
            self.novel_cameras,self.view_selection,surface_bounds=choose_offset_views(capture,self)
            surface_points=surface_bounds['figure']
            self.novel_available=len(self.novel_cameras)==2
            self.novel_yaws=self.view_selection['selected_degrees']
            self.novel_yaw=self.novel_yaws[0] if self.novel_yaws else 0.
            self.novel_camera=self.novel_cameras[0] if self.novel_available else self.source_camera
            self.ground_available=bool(len(surface_bounds['ground']))
            if len(surface_points):
                self.detail_camera=fixed_scene_camera(surface_points[None],
                    np.broadcast_to(np.eye(3)*1e-6,(1,len(surface_points),3,3)),
                    np.ones((1,len(surface_points))),1024,576,yaw=20.,pitch=14.,focal=1100.)
        # The camera positions, rotations and fields of view below are the
        # ACTUAL synthesis cameras, drawn in the same coordinates as the matter.
        from .render import frustum_points
        self.glyph_target=self.novel_target
        self.glyph_cameras=self.novel_cameras+[self.source_camera]
        # Keep the framing envelope unchanged when shortening the drawn cones.
        camera_points=(np.concatenate([frustum_points(c,self.glyph_target,k,length_scale=1.)
                       for k,c in enumerate(self.novel_cameras)]+[
                       frustum_points(self.source_camera,self.glyph_target,2,length_scale=1.)]))
        self.scene_camera,self.scene_positions,self.scene_targets=following_scene_camera(
            self.display_means,self.display_covs,self.alpha,self.ids,self.targets,self.fps,camera_points)
        # One fixed, oblique scene-surface camera, without per-frame fitting.
        self.ground_camera=orbit_camera(np.array([0.,0.,3.]),20.,14.,5.8,540.,560,400)
        if self.ground_available and len(surface_bounds['ground']):
            points=surface_bounds['ground']
            self.ground_camera=fixed_scene_camera(points[None],
                np.broadcast_to(np.eye(3)*1e-6,(1,len(points),3,3)),np.ones((1,len(points))),
                1024,576,yaw=20.,pitch=14.,focal=1100.)

    def frame(self,raw,index,stabilized=None):
        out = camera_coordinates(raw)
        enabled=self.stabilized if stabilized is None else stabilized
        out.update(raw_means=out['means'],raw_covs=out['covs'])
        if enabled:
            out.update(means=self.display_means[index],covs=self.display_covs[index])
        out.update(display_alpha=self.alpha[index],colors=self.colors[index],
                   observation_age=self.missing[index],diagram_depth=self.diagram_depth,
                   playback_time=index/self.fps,surface_spacing=self.surface_spacing)
        out.update(rollout_fps=self.rollout_fps[index],
                   initial_rollout_fps=self.initial_rollout_fps,
                   inference_warmup=index<self.warmup)
        viewer=Camera(self.scene_positions[index],self.scene_camera.rotation,
                      self.scene_camera.focal,self.scene_camera.width,self.scene_camera.height)
        out.update(scene_camera=viewer, novel_camera=self.novel_camera,
                   source_camera=self.source_camera,novel_cameras=self.novel_cameras,
                   detail_camera=self.detail_camera,novel_target=self.novel_target,novel_yaw=self.novel_yaw,
                   novel_available=self.novel_available,novel_yaws=self.novel_yaws,
                   ground_camera=self.ground_camera if hasattr(self,'ground_camera') else self.detail_camera,
                   ground_available=self.ground_available if hasattr(self,'ground_available') else False)
        if hasattr(self,'glyph_cameras'):
            out.update(glyph_cameras=self.glyph_cameras,glyph_target=self.glyph_target)
        start=max(0,index-round(self.fps*.22))
        out['trail_points']=(self.display_means if enabled else self.means)[start:index+1,self.trail_ids]
        out['trail_alpha']=self.alpha[start:index+1,self.trail_ids]
        out['trail_object_ids']=self.ids[self.trail_ids]
        return out

    def save(self,path):
        np.savez_compressed(path,targets=self.targets,radius=self.radius,
                            display_means=self.display_means,display_covs=self.display_covs,
                            geometry_filter_enabled=self.stabilized,flow_valid=self.flow_valid,
                            rollout_fps=self.rollout_fps,
                            alpha=self.alpha,observation_age=self.missing,
                            trail_ids=self.trail_ids,coordinate_system='observing-camera relative',
                            diagram_depth=self.diagram_depth,
                            scene_position=self.scene_camera.position,
                            scene_positions=self.scene_positions,
                            scene_targets=self.scene_targets,
                            scene_rotation=self.scene_camera.rotation,
                            scene_focal=self.scene_camera.focal,
                            scene_size=np.array([self.scene_camera.width,self.scene_camera.height]),
                            detail_position=self.detail_camera.position,detail_rotation=self.detail_camera.rotation,
                            detail_focal=self.detail_camera.focal,detail_size=np.array([self.detail_camera.width,self.detail_camera.height]),
                            novel_positions=np.array([c.position for c in self.novel_cameras]),
                            novel_rotations=np.array([c.rotation for c in self.novel_cameras]),
                            novel_focals=np.array([c.focal for c in self.novel_cameras]),
                            novel_position=self.novel_camera.position,
                            novel_rotation=self.novel_camera.rotation,
                            novel_focal=self.novel_camera.focal,novel_target=self.novel_target,
                            camera_policy='near-frontal overview; smooth object-centered position and distance; constant orientation/focal length; fixed readout poses; observing-camera coordinates',
                            surface_spacing=self.surface_spacing,surface_density_level=.24,
                            novel_yaws=np.array(self.novel_yaws),
                            novel_available=self.novel_available,
                            ground_available=self.ground_available,
                            default_position=self.source_camera.position,default_rotation=self.source_camera.rotation,
                            default_focal=self.source_camera.focal,default_size=np.array([640,360]),
                            ground_position=self.ground_camera.position,ground_rotation=self.ground_camera.rotation,
                            ground_focal=self.ground_camera.focal,ground_size=np.array([self.ground_camera.width,self.ground_camera.height]),
                            view_size=np.array([640,360]),view_near=.5,playback_fps=self.fps)
