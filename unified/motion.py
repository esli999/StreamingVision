"""Display observed scene motion and synchronized inference timing."""
import cv2
import numpy as np


def motion_vectors(frame):
    """Previous-to-current displacement, anchored at the current source pixels."""
    if 'backward_grid' not in frame:
        return np.zeros((45,80,2),np.float32),np.zeros((45,80),bool)
    vectors=-np.asarray(frame['backward_grid'],np.float32).transpose(1,2,0)*8.
    valid=np.asarray(frame['reliable_grid'],bool)&np.isfinite(vectors).all(-1)
    return np.where(valid[...,None],vectors,0.),valid


def motion_colors(vectors, scale):
    angle=np.arctan2(vectors[...,1],vectors[...,0])
    magnitude=np.linalg.norm(vectors,axis=-1)
    hsv=np.zeros((*magnitude.shape,3),np.uint8)
    hsv[...,0]=((angle/(2*np.pi)+1)%1*179).astype(np.uint8)
    hsv[...,1]=210
    hsv[...,2]=(80+175*np.sqrt(np.clip(magnitude/max(scale,1.),0,1))).astype(np.uint8)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)


def scene_motion(frame, scale):
    """Measured direction color plus sparse displacement arrows over RGB edges.

    A clip-wide scale avoids per-frame contrast pumping. Vectors are neither
    camera-compensated nor augmented with invented trajectories. Their arrows
    point from the measured previous pixel to its current location.
    """
    vectors,valid=motion_vectors(frame)
    yy,xx=np.mgrid[:360,:640].astype(np.float32)
    field=cv2.remap(vectors,(xx-4)/8,(yy-4)/8,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    confidence=cv2.remap(valid.astype(np.float32),(xx-4)/8,(yy-4)/8,cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    colors=motion_colors(field,scale).astype(float)
    gray=cv2.cvtColor(frame['bgr'],cv2.COLOR_BGR2GRAY).astype(float)/255.
    speed=np.linalg.norm(field,axis=-1)
    amount=.85*np.sqrt(np.clip(speed/max(scale,1.),0,1))*confidence
    backdrop=np.array([15.,23.,34.])[None,None]+gray[...,None]*44
    canvas=(backdrop*(1-amount[...,None])+colors*amount[...,None]).clip(0,255).astype(np.uint8)
    edges=cv2.Canny(frame['bgr'],65,145).astype(float)/255.
    canvas=np.clip(canvas.astype(float)+edges[...,None]*25,0,255).astype(np.uint8)
    for gy in range(3,45,6):
        for gx in range(3+(gy//6%2)*3,80,6):
            if not valid[gy,gx]: continue
            motion=vectors[gy,gx];length=float(np.linalg.norm(motion))
            if length<.35: continue
            end=np.array([4+8*gx,4+8*gy],float)
            # Amplification is fixed for the clip, with length capped for clear
            # glyphs. Color retains the actual speed even when glyphs saturate.
            delta=motion/length*min(24.,length*3.)
            a=tuple(np.rint(end-delta).astype(int));b=tuple(end.astype(int))
            cv2.arrowedLine(canvas,a,b,(8,14,23),2,cv2.LINE_AA,tipLength=.28)
            cv2.arrowedLine(canvas,a,b,(229,242,249),1,cv2.LINE_AA,tipLength=.28)
    # A compact direction key uses the exact same hue convention as the field.
    y,x=np.mgrid[-21:22,-21:22];wheel=np.stack([x,y],-1).astype(float)
    ring=(x*x+y*y<21**2)&(x*x+y*y>12**2)
    patch=canvas[302:345,582:625]
    patch[ring]=motion_colors(wheel,21)[ring]
    return canvas


def rollout_rates(seconds, warmup=3, window=15):
    """Causal throughput: completed frames / synchronized engine duration.

    Missing durations remain unavailable; never substitute playback cadence or
    the clip's future mean. Warmup is excluded from each rolling window.
    """
    seconds=np.asarray(seconds,float)
    rates=np.full(len(seconds),np.nan)
    for i in range(warmup,len(seconds)):
        values=seconds[max(warmup,i-window+1):i+1]
        if np.isfinite(values).all() and np.all(values>0):
            rates[i]=len(values)/values.sum()
    return rates
