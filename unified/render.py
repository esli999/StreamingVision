"""CPU-only figure renderer for immutable inference captures."""
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .data import write_json, rate, fingerprint, implementation_fingerprint
from .geometry import Camera, orbit_camera, look_at
from .presentation import Presentation

WHITE = (255,255,255)
INK = (30,40,54)
MUTED = (104,115,131)
PALETTE = np.array([[164,174,183],[36,166,219],[234,143,52],[177,101,215],
                    [46,170,130],[223,94,128],[143,162,54],[85,122,228],
                    [205,76,66],[144,99,62],[71,168,170],[175,100,157],[181,164,69]],np.uint8)  # RGB
FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')


def color_for(ids):
    ids = np.asarray(ids)
    index = np.where(ids == 0,0,1+(ids-1)%(len(PALETTE)-1))
    return PALETTE[index]


def text(draw, xy, words, size=24, color=INK, bold=False):
    font = ImageFont.truetype(str(FONT_ROOT/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)
    draw.text(xy,words,fill=color,font=font)


def splat_particles(frame, camera, *, semantic=True, background_alpha=.09, sigma=1.8,near=.05):
    """Continuous, order-independent surface splats with directional shading.

    Per-pixel front-surface depth replaces center-depth painter ordering. A soft
    nearest-surface blend is continuous when ellipsoids intersect. It is an
    illustrative transparency model, not a physically based volume renderer.
    """
    h,w = camera.height,camera.width
    means,covs = frame['means'],frame['covs']
    uv,z = camera.project(means)
    footprint = camera.footprints(means,covs,sigma)
    eigen,v = np.linalg.eigh(footprint)
    radii = np.sqrt(np.maximum(eigen,.5))
    colors = frame['colors'][:,::-1].astype(np.float32)
    identities = color_for(frame['object_ids']).astype(np.float32)
    foreground = frame['object_ids']>0
    confidence = frame.get('display_alpha',np.clip(frame.get('support',frame['weights']*2925)/1.5,0,1))
    # A feathered near plane prevents stale, unsupported geometry crossing the
    # viewing camera from turning into an enormous projected ellipse. The source
    # geometry uses relative depths 1..5; its views use near=0.5. This is frustum
    # clipping, not a per-frame particle-size cutoff.
    clip=np.clip((z-near)/max(near*.5,.025),0,1)
    confidence=confidence*clip*clip*(3-2*clip)
    depth_radius = sigma*np.sqrt(np.maximum(np.einsum('i,nij,j->n',camera.rotation[2],covs,camera.rotation[2]),1e-8))
    # A fixed scale from this frame's geometry; independent of drawing order.
    softness = max(float(np.median(depth_radius))*.16,.006)
    layers=[]
    for selected,opacity in ((~foreground,background_alpha),(foreground,.96)):
        near=np.full((h,w),np.inf,np.float32)
        total=np.zeros((h,w),np.float32)
        rgb=np.zeros((h,w,3),np.float32)
        coverage=np.zeros((h,w),np.float32)
        for b in np.flatnonzero(selected):
            if z[b]<=.05 or not np.isfinite(uv[b]).all() or confidence[b]<=.0001:
                continue
            r=radii[b]
            extent=np.sqrt(np.sum((v[b]*r[None])**2,axis=1))
            x0,y0=np.maximum(np.floor(uv[b]-extent-2),[0,0]).astype(int)
            x1,y1=np.minimum(np.ceil(uv[b]+extent+2),[w,h]).astype(int)
            if x1<=x0 or y1<=y0:
                continue
            yy,xx=np.mgrid[y0:y1,x0:x1].astype(np.float32)
            dx,dy=xx-uv[b,0],yy-uv[b,1]
            nx=(dx*v[b,0,0]+dy*v[b,1,0])/r[0]
            ny=(dx*v[b,0,1]+dy*v[b,1,1])/r[1]
            rho=nx*nx+ny*ny
            nz=np.sqrt(np.maximum(0,1-rho))
            light=np.clip(-.32*dx/max(extent[0],1)-.40*dy/max(extent[1],1)+.85*nz,0,1)
            fill=colors[b]
            if semantic and foreground[b]:
                fill=.58*fill+.42*identities[b]
            if not foreground[b]:
                fill=.25*fill+.75*np.array([167,182,192])
            shaded=np.clip(fill[None,None]*(.66+.34*light[...,None])+22*light[...,None]**18,0,255)
            if semantic and foreground[b]:
                rim=np.clip((rho-.87)/.13,0,1)*.42
                shaded=shaded*(1-rim[...,None])+identities[b]*rim[...,None]
            a=np.clip((1-rho)*min(r)*.9,0,1)*confidence[b]
            depth=z[b]-depth_radius[b]*nz
            old=near[y0:y1,x0:x1]
            new=np.minimum(old,np.where(a>0,depth,np.inf))
            # Ignore uncovered pixels to avoid inf-inf; every update is a
            # commutative soft-min reduction over per-pixel surface depth.
            valid=np.isfinite(new)
            safe=np.where(valid,new,0.)
            old_scale=np.exp(np.minimum(0,(safe-old)/softness))
            weight=a*np.exp(np.minimum(0,(safe-depth)/softness))
            rgb[y0:y1,x0:x1]*=old_scale[...,None]
            rgb[y0:y1,x0:x1]+=shaded*weight[...,None]
            total[y0:y1,x0:x1]*=old_scale
            total[y0:y1,x0:x1]+=weight
            old[:]=new
            patch=coverage[y0:y1,x0:x1]
            patch[:]=np.maximum(patch,a)
        color=rgb/np.maximum(total[...,None],1e-8)
        layers.append((color,coverage*opacity,near))
    bg,ba,bz=layers[0];fg,fa,fz=layers[1]
    # Composite the foreground/context layers according to surface depth too.
    # Context remains translucent by design, including when it occludes matter.
    bg_first=255*(1-ba[...,None])+bg*ba[...,None]
    bg_first=bg_first*(1-fa[...,None])+fg*fa[...,None]
    fg_first=255*(1-fa[...,None])+fg*fa[...,None]
    fg_first=fg_first*(1-ba[...,None])+bg*ba[...,None]
    difference=np.where(np.isfinite(bz),bz,1e5)-np.where(np.isfinite(fz),fz,1e5)
    front=1/(1+np.exp(-np.clip(difference/softness,-30,30)))
    canvas=bg_first*front[...,None]+fg_first*(1-front[...,None])
    return canvas.clip(0,255).astype(np.uint8)


def views(frame, target, angles=(-15,15)):
    position = frame['world_translation']
    rotation = frame['world_rotation'].T
    source = Camera(position,rotation,500.,640,360)
    # Orbits are expressed in the source camera's coordinate system and then
    # transformed to world coordinates.
    local_target = (target-position)@rotation.T
    cameras = []
    for angle in angles:
        yaw = np.deg2rad(angle)
        rotation_y = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        local_position = local_target+rotation_y@(-local_target)
        local = look_at(local_position,local_target,500.,640,360)
        cameras.append(Camera(local.position@frame['world_rotation'].T+position,
                              local.rotation@rotation,500.,640,360))
    return source,cameras


def paste_fit(image, array, box):
    x,y,w,h = box
    tile = Image.fromarray(array)
    scale=min(w/tile.width,h/tile.height)
    tile=tile.resize((max(1,round(tile.width*scale)),max(1,round(tile.height*scale))),Image.Resampling.LANCZOS)
    image.paste(tile,(x+(w-tile.width)//2,y+(h-tile.height)//2))


# Shared paper-figure coordinates: matter above its readouts, novel poses right.
CANVAS_SIZE = (5120, 2080)
CONTENT_BOXES = {
    'input': (40, 310, 1064, 886),
    'scene': (1520, 125, 3960, 1125),
    'surface': (1496, 1430, 2520, 2006),
    'ground': (2896, 1430, 3920, 2006),
    'segmentation': (40, 1430, 1064, 2006),
    'observed': (4280, 320, 5080, 770),
    'novel_left': (4280, 925, 5080, 1375),
    'novel': (4280, 1530, 5080, 1980),
}
VIEW_COLORS = [(196,112,39),(132,83,177),(42,123,158)]
DEFAULT_COLOR = (37,102,118)
FRUSTUM_BADGE_OFFSETS = [(45,48),(-44,-54),(-28,106)]


def arrow(draw,a,b,color=MUTED,width=3):
    """One solid polygon: the shaft stops at the head rather than its tip."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    length=np.linalg.norm(b-a)
    if length<=0: return
    direction=(b-a)/length
    side=np.array([-direction[1],direction[0]])
    neck=b-direction*min(length*.45,max(12,width*2))
    shaft=side*width/2;wing=side*max(6,width*1.2)
    draw.polygon([tuple(p) for p in (a-shaft,neck-shaft,neck-wing,b,
                                    neck+wing,neck+shaft,a+shaft)],fill=color)


def tracking_overlay(frame):
    labels=frame['labels']
    rgb=frame['bgr'][...,::-1].astype(np.float32)
    seg=.22*rgb+.78*248
    fg=labels>0
    seg[fg]=.40*rgb[fg]+.60*color_for(labels[fg])
    for label in np.unique(labels):
        if label==0: continue
        contours,_=cv2.findContours((labels==label).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg,contours,-1,tuple(float(c) for c in color_for(label)),1,cv2.LINE_AA)
    return seg.astype(np.uint8)


def scaled_camera(camera, scale):
    return Camera(camera.position, camera.rotation, camera.focal*scale,
                  max(1, round(camera.width*scale)), max(1, round(camera.height*scale)))


def frustum_points(camera, target, index=0, length_scale=.64):
    """A true perspective volume: camera center, near rectangle, far rectangle."""
    length=np.linalg.norm(target-camera.position)*(.36 if index==2 else .32)*length_scale
    points=[camera.position]
    for depth in (length*.12,length):
        half_x=camera.width*depth/(2*camera.focal)
        half_y=camera.height*depth/(2*camera.focal)
        corners=np.array([[-half_x,-half_y,depth],[half_x,-half_y,depth],
                          [half_x,half_y,depth],[-half_x,half_y,depth]])
        points.extend(camera.position+corners@camera.rotation)
    return np.asarray(points)


def scene_cameras(cloud, viewer, cameras, target, scale=1.,default_camera=None):
    """Actual synthesis extrinsics/FOV, in the particle scene's coordinate frame."""
    image=Image.fromarray(cloud).convert('RGBA')
    overlay=Image.new('RGBA',image.size);draw=ImageDraw.Draw(overlay)
    entries=[]
    query_cameras=list(enumerate(cameras))
    if default_camera is not None: query_cameras.append((2,default_camera))
    for k,camera in query_cameras:
        vertices=frustum_points(camera,target,k)
        uv,z=viewer.project(vertices)
        if np.any(z<=.05): continue
        entries.append((float(z.mean()),k,uv))
    # Face painting into one RGBA layer used to ERASE another camera's edges.
    # Blend faces first, then draw every wire and badge over the finished faces.
    for _,k,uv in sorted(entries,reverse=True):
        color=DEFAULT_COLOR if k==2 else VIEW_COLORS[k];points=[tuple(p) for p in uv]
        face_layer=Image.new('RGBA',image.size);faces=ImageDraw.Draw(face_layer)
        for face in ([1,2,6,5],[2,3,7,6],[3,4,8,7],[4,1,5,8],[5,6,7,8]):
            faces.polygon([points[j] for j in face],fill=(*color,10 if k==2 else 6))
        image=Image.alpha_composite(image,face_layer)
    for _,k,uv in sorted(entries,key=lambda e:(e[1]==2,-e[0])):
        color=DEFAULT_COLOR if k==2 else VIEW_COLORS[k];points=[tuple(p) for p in uv]
        width=max(1,round((12 if k==2 else 10)*scale))
        draw.line([points[j] for j in [5,6,7,8,5]],fill=(*color,255),width=width)
        for j in range(1,5):
            draw.line([points[0],points[j+4]],fill=(*color,255),width=width)
    # Put every camera name above all wires, including intersecting frustums.
    for _,k,uv in entries:
        color=DEFAULT_COLOR if k==2 else VIEW_COLORS[k]
        px,py=uv[0];dx,dy=FRUSTUM_BADGE_OFFSETS[k];lx,ly=px+dx*scale,py+dy*scale
        draw.line([(px,py),(lx,ly)],fill=(*color,255),width=max(1,round(5*scale)))
        label=['Pose 1','Pose 2','Initial'][k]
        font=ImageFont.truetype(str(FONT_ROOT/'DejaVuSans-Bold.ttf'),max(12,round(58*scale)))
        half=draw.textlength(label,font=font)/2+18*scale
        draw.rounded_rectangle((lx-half,ly-43*scale,lx+half,ly+43*scale),
                               radius=14*scale,fill=(*color,255))
        draw.text((lx,ly),label,font=font,fill=WHITE,anchor='mm')
    return np.asarray(Image.alpha_composite(image,overlay).convert('RGB'))


def figure_boxes(ground_available=True):
    boxes=dict(CONTENT_BOXES)
    if not ground_available:
        boxes['surface']=(2208,1430,3232,2006)
    return boxes


def compose(frame,manifest,target,radius,index,render_scale=1.,representation='mesh'):
    """Particles, three mid-level percepts and three camera-colored RGB views."""
    from .surface import particle_surfaces, raster_surfaces
    from .projection import project_readouts
    from .novel import layered_readouts, shaded_surface
    if representation not in ('mesh','particles'):
        raise ValueError('representation must be mesh or particles')
    image=Image.new('RGB',CANVAS_SIZE,WHITE);draw=ImageDraw.Draw(image)
    boxes=figure_boxes(frame.get('ground_available',False))
    def place(array,key):
        x,y,r,b=boxes[key];paste_fit(image,array,(x,y,r-x,b-y))
    text(draw,(40,206),'Input video',72,INK,True)
    text(draw,(1520,28),'3D particle-based matter representation',72,INK,True)
    place(frame['bgr'][...,::-1],'input')
    particles=particle_surfaces(frame,highlight=True,context_opacity=.90)
    hero=scaled_camera(frame['scene_camera'],render_scale)
    cloud=raster_surfaces(particles,hero,semantic=False)
    cloud=scene_cameras(cloud,hero,frame['novel_cameras'],frame['novel_target'],render_scale,
                        default_camera=frame['source_camera'])
    place(cloud,'scene')
    readouts=layered_readouts(frame)
    source=scaled_camera(frame['source_camera'],1.6*render_scale)
    _,seg,_,_=project_readouts(readouts.observed,source)
    place(seg,'segmentation')
    if representation=='mesh':
        surfaces=[shaded_surface(readouts.figure)]
    else:
        surfaces=[s for s in particle_surfaces(frame,highlight=False) if s.object_id>0]
    place(raster_surfaces(surfaces,scaled_camera(frame['detail_camera'],render_scale),semantic=False),'surface')
    if frame.get('ground_available',False):
        place(raster_surfaces([shaded_surface(readouts.ground_observed)],
            scaled_camera(frame['ground_camera'],render_scale),semantic=False),'ground')
    accent=(91,111,128)
    # Dedicated gutters keep every arrow outside imagery and camera labels.
    arrow(draw,(1100,598),(1480,598),accent,44)
    arrow(draw,(2740,1150),(2740,1324),accent,44)
    arrow(draw,(4000,625),(4240,625),accent,44)
    text(draw,(40,1240),'Inferred mid-level visual representations from initial camera view',68,DEFAULT_COLOR,True)
    draw.line([(40,1340),(3920,1340)],fill=DEFAULT_COLOR,width=5)
    text(draw,(40,1356),'2D segmentation',62,INK,True)
    if frame.get('ground_available',False):
        text(draw,(1496,1364),'Deformable object · 2.5D surface',54,INK,True)
        text(draw,(2896,1364),'Background · 2.5D surface',54,INK,True)
    else:
        text(draw,(2208,1364),'Deformable object · 2.5D surface',54,INK,True)
    text(draw,(4280,28),'2D renderings from',64,INK,True)
    text(draw,(4280,108),'the 3D particle model',64,INK,True)
    projections=[(frame['source_camera'],readouts.observed,'observed','Initial camera pose',DEFAULT_COLOR)]
    for k,(camera,key,angle) in enumerate(zip(frame['novel_cameras'],['novel_left','novel'],frame['novel_yaws'])):
        # Negative orbit yaw moves the camera to the initial camera's right.
        label=f'Reposed camera {k+1}\n{abs(angle):g}° '+('right' if angle<0 else 'left')
        projections.append((camera,readouts.scene,key,label,VIEW_COLORS[k]))
    for camera,surface,key,label,tint in projections:
        rgb,_,_,_=project_readouts(surface,scaled_camera(camera,1.25*render_scale))
        place(rgb,key)
        x,y,r,b=boxes[key]
        if '\n' in label:
            title,offset=label.split('\n')
            text(draw,(x,y-152),title,62,tint,True)
            text(draw,(x,y-78),offset,42,tint,True)
        else:
            text(draw,(x,y-92),label,62,tint,True)
        draw.rectangle((x-4,y-4,r+4,b+4),outline=tint,width=7)
    fps=frame.get('rollout_fps',np.nan)
    if not np.isfinite(fps): fps=frame.get('initial_rollout_fps',np.nan)
    speed=f'{fps:.1f} FPS' if np.isfinite(fps) else '— FPS'
    text(draw,(1124,320),'Real-time',62,INK,True)
    text(draw,(1124,400),'inference',62,INK,True)
    text(draw,(1160,480),speed,56,INK,True)
    return image


def render(capture,output,width=CANVAS_SIZE[0],representation='mesh',stabilized=True):
    if width < 640 or width % 32:
        raise ValueError('Render width must be >=640 and a multiple of 32')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    presentation = Presentation(capture,stabilized=stabilized)
    targets,radius = presentation.targets,presentation.radius
    presentation.save(output.with_suffix('.cameras.npz'))
    fitting_seconds = time.perf_counter()-started
    height = 2*round(width*CANVAS_SIZE[1]/CANVAS_SIZE[0]/2)
    fps = presentation.fps
    temporary = output.with_name(output.stem+'.partial.mp4')
    command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
               '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264',
               '-preset','medium','-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    times,write_times = [],[]
    process = subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(capture.manifest['frames']):
            t = time.perf_counter()
            frame = presentation.frame(capture.read(i),i)
            image = compose(frame,presentation.manifest,targets[i],radius,i,min(1.,width/CANVAS_SIZE[0]),representation)
            if width != CANVAS_SIZE[0]:
                image = image.resize((width,height),Image.Resampling.LANCZOS)
            times.append(time.perf_counter()-t)
            if i in {0,capture.manifest['frames']//2,capture.manifest['frames']-1}:
                image.save(output.with_name(f'{output.stem}_{i:05d}.jpg'),quality=94)
                image.save(output.with_name(f'{output.stem}_{i:05d}.png'))
            t = time.perf_counter()
            process.stdin.write(image.tobytes())
            write_times.append(time.perf_counter()-t)
            if (i+1)%20 == 0:
                print(f'render {output.stem}: {i+1}/{capture.manifest["frames"]}',flush=True)
        process.stdin.close()
        t = time.perf_counter()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed to encode the figure')
        flush_seconds = time.perf_counter()-t
    except BaseException:
        process.kill(); process.wait()
        raise
    temporary.replace(output)
    report = dict(video=str(output),frames=capture.manifest['frames'],width=width,height=height,
                  capture_path=str(capture.path.resolve()),capture_manifest_sha256=fingerprint(capture.path/'manifest.json'),
                  renderer_implementation_sha256=implementation_fingerprint(),
                  source_fps=fps,render_fps=rate(times),render_seconds=sum(times),
                  nominal_capture_fps=capture.manifest['source_fps'],
                  representation=representation,surface_spacing=presentation.surface_spacing,
                  layout='Wide canvas with solid block arrows in dedicated gutters; inference FPS above the input arrow; particle scene branches down to inferred mid-level visual percepts and at mid-height to 2D renderings from the 3D particle model; plain-language pose offsets',
                  geometry_filter_enabled=stabilized,
                  scene_representation='rasterized covariance ellipsoids; all observed scene particles',
                  camera_policy='overview follows the smoothed scene target with constant rotation and focal length; synthesis and surface cameras retain fixed poses in observing-camera coordinates',
                  novel_view_degrees=presentation.novel_yaws,
                  novel_view_selection=presentation.view_selection,
                  readouts='Observed posterior RGB/membership; offset synthesis from fixed layered 2.5D geometry with background-Gaussian conditional RGB/depth and observed figure-before-ground ordering',
                  frustums='Initial is the source camera; Pose 1 and Pose 2 are actual offset poses. All extrinsics and fields of view are in particle coordinates, with matching rendering borders.',
                  surface_readout='natural-color 2.5D sheets; no semantic highlight',
                  ground_surface_shown=presentation.ground_available,
                  ground_surface_policy='Show the background-Gaussian sheet within the default frustum whenever geometry exists; support is reported, not a panel-omission gate.',
                  default_readouts='Source-pose posterior RGB and membership; figure and ground 2.5D sheets queried within that frustum and displayed obliquely.',
                  playback_timing='Mean decoded source timestamps; constant-rate output preserves duration',
                  encoder_pipe_seconds=sum(write_times),encoder_flush_seconds=flush_seconds,
                  camera_fitting_seconds=fitting_seconds,render_wall_seconds=time.perf_counter()-started,
                  note='Encoding overlaps rendering; pipe/flush times are blocking wall time, not isolated encoder compute.',
                  inference_fps=capture.manifest.get('inference_fps'),
                  displayed_inference_timing='causal 15-frame rolling completed frames / synchronized engine seconds; warmup excluded from measurements; opening frames display the first measured rate',
                  temporal_display=presentation.diagnostics)
    report['track_and_render_wall_seconds'] = (capture.manifest['tracking_wall_seconds']
        + capture.manifest.get('attributed_model_load_seconds',0)+report['render_wall_seconds'])
    report['end_to_end_fps'] = report['frames']/report['track_and_render_wall_seconds']
    write_json(output.with_suffix('.render.json'),report)
    print(f'Saved {output}',flush=True)
    return report
