#!/usr/bin/env python3
"""Audit encoded paper figures, true offset poses, masks, framing and engine FPS."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from unified.data import Capture,write_json,implementation_fingerprint
from unified.geometry import Camera
from unified.presentation import camera_coordinates
from unified.novel import layered_readouts,coverage_statistics,shaded_surface
from unified.projection import project_readouts
from unified.surface import raster_surfaces
from unified.render import CANVAS_SIZE,figure_boxes,color_for,text,frustum_points,FRUSTUM_BADGE_OFFSETS,DEFAULT_COLOR,VIEW_COLORS


def audit(video):
    report=json.loads(video.with_suffix('.render.json').read_text())
    assert report['renderer_implementation_sha256']==implementation_fingerprint()
    assert (report['width'],report['height'])==CANVAS_SIZE
    capture=Capture(report['capture_path']);saved=np.load(video.with_suffix('.cameras.npz'))
    source=Camera(np.zeros(3),np.eye(3),500.,640,360)
    np.testing.assert_array_equal(saved['default_position'],source.position)
    np.testing.assert_array_equal(saved['default_rotation'],source.rotation)
    assert saved['default_focal']==source.focal
    cameras=[Camera(p,r,float(f),640,360) for p,r,f in
             zip(saved['novel_positions'],saved['novel_rotations'],saved['novel_focals'])]
    assert len(cameras)==2 and len(saved['novel_yaws'])==2
    assert saved['novel_yaws'][0]<=-16 and saved['novel_yaws'][1]>=16
    pose_checks=[]
    target=saved['novel_target']
    for camera in cameras:
        baseline=float(np.linalg.norm(camera.position))
        angle=float(np.degrees(np.arccos(np.clip(camera.rotation[2,2],-1,1))))
        assert baseline>.1 and angle>=5.99
        np.testing.assert_allclose(camera.rotation[2],(target-camera.position)/np.linalg.norm(target-camera.position))
        probes=np.array([target,target+np.array([0.,0.,2.])])
        uv,_=camera.project(probes); parallax=float(abs(uv[0,0]-uv[1,0]))
        assert parallax>10
        pose_checks.append(dict(translation_norm=baseline,rotation_degrees=angle,
                                depth_dependent_parallax_pixels=parallax))
    viewer=Camera(saved['scene_position'],saved['scene_rotation'],float(saved['scene_focal']),*saved['scene_size'])
    viewer_positions=(saved['scene_positions'] if 'scene_positions' in saved else
                      np.tile(viewer.position,(capture.manifest['frames'],1)))
    detail=Camera(saved['detail_position'],saved['detail_rotation'],float(saved['detail_focal']),*saved['detail_size'])
    ground_camera=Camera(saved['ground_position'],saved['ground_rotation'],float(saved['ground_focal']),*saved['ground_size'])
    assert saved['ground_available']
    for k,camera in enumerate(cameras+[source]):
        points=frustum_points(camera,target,k)
        uv,_=camera.project(points[1:])
        np.testing.assert_allclose(uv,np.tile([[0,0],[640,0],[640,360],[0,360]],(2,1)),atol=1e-8)
        uv,z=viewer.project(points)
        assert np.all(z>0) and np.all(uv>0) and np.all(uv<[viewer.width,viewer.height])
        badge=uv[0]+FRUSTUM_BADGE_OFFSETS[k]
        # Named camera pills are wider than the former single-letter circles.
        margin=np.array([100,36])
        assert np.all(badge>margin) and np.all(badge<np.array([viewer.width,viewer.height])-margin),('badge clipped',k,badge)
    boxes=figure_boxes(bool(saved['ground_available']))
    out=video.parent/'encoded_review';out.mkdir(exist_ok=True)
    decoder=cv2.VideoCapture(str(video));frames=[];pages=[];seconds=[];margin_min=np.inf
    prior_dir='default_debug' if capture.manifest['clip']=='test' else 'animal_default'
    old_path=video.parent.parent/prior_dir/video.with_suffix('.cameras.npz').name
    old=np.load(old_path) if old_path.exists() else None
    if old is not None:
        np.testing.assert_array_equal(saved['display_means'],old['display_means'])
        np.testing.assert_array_equal(saved['display_covs'],old['display_covs'])
    previous=None;transitions=[];held_colors=None
    for i in range(capture.manifest['frames']):
        viewer=Camera(viewer_positions[i],viewer.rotation,viewer.focal,viewer.width,viewer.height)
        ok,bgr=decoder.read();assert ok, f'Missing encoded frame {i}'
        assert bgr.shape[:2]==CANVAS_SIZE[::-1]
        raw=capture.read(i);seconds.append(float(raw['inference_seconds']))
        colors=raw['colors'].copy()
        if held_colors is not None: colors[raw['support']<.5]=held_colors[raw['support']<.5]
        held_colors=colors
        f=camera_coordinates(raw)
        f.update(means=saved['display_means'][i],covs=saved['display_covs'][i],
                 display_alpha=saved['alpha'][i],colors=colors)
        readouts=layered_readouts(f)
        source_rgb,_,source_mask,source_coverage=project_readouts(readouts.observed,source)
        target_mask=raw['labels']>0; union=(source_mask|target_mask).sum()
        overlap=float((source_mask&target_mask).sum()/max(union,1))
        mae=float(np.abs(source_rgb.astype(float)-raw['reconstruction'][...,::-1]).mean())
        assert overlap>.9999 and float((source_coverage<.5).mean())<.001
        checks=dict(frame=i+1,source_mask_iou=overlap,source_rgb_mae=mae,
                    ground_supported_fraction=readouts.ground_supported_fraction,
                    ground_occlusion_adjusted_fraction=readouts.ground_occlusion_adjusted_fraction)
        # The surface panel is the source-frustum subset of the ground query.
        uv,z=source.project(readouts.ground_observed.vertices)
        assert np.all(z>.5) and np.all(uv>=-1e-3) and np.all(uv<=[640.001,360.001])
        uv,z=ground_camera.project(readouts.ground_observed.vertices)
        assert np.all(z>0) and np.all(uv>0) and np.all(uv<[ground_camera.width,ground_camera.height])
        surface=shaded_surface(readouts.ground_observed)
        assert np.isfinite(surface.normals).all() and len(surface.faces)>0
        ground_rgb=raster_surfaces([surface],ground_camera,semantic=False)
        x,y,r,b=boxes['ground']
        ground_error=float(np.abs(bgr[y:b,x:r,::-1].astype(float)-ground_rgb).mean())
        assert ground_error<2.,(i,'ground encoded mismatch',ground_error)
        checks['ground_encoded_rgb_mae']=ground_error
        *_,coverage=project_readouts(readouts.ground_observed,ground_camera)
        missing=(coverage<.5).astype(np.uint8)
        count,components,stats,_=cv2.connectedComponentsWithStats(missing,8)
        border=np.unique(np.r_[components[0],components[-1],components[:,0],components[:,-1]])
        interior=[stats[k,cv2.CC_STAT_AREA] for k in range(1,count) if k not in border]
        checks['ground_interior_hole_pixels']=int(sum(interior))
        checks['ground_projected_area_pixels']=int((coverage>=.5).sum())
        for k,camera in enumerate(cameras+[source]):
            uv,_=viewer.project(camera.position[None]);badge=uv[0]+FRUSTUM_BADGE_OFFSETS[k]+np.array(boxes['scene'][:2])
            cx,cy=np.round(badge).astype(int);patch=bgr[cy-28:cy+29,cx-28:cx+29,::-1].astype(float)
            color=DEFAULT_COLOR if k==2 else VIEW_COLORS[k]
            assert np.count_nonzero(np.linalg.norm(patch-color,axis=-1)<14)>500,(i,'unreadable frustum badge',k)
        x,y,r,b=boxes['segmentation'];scale=(r-x)/source.width
        mask_camera=Camera(source.position,source.rotation,source.focal*scale,r-x,b-y)
        _,_,mask,_=project_readouts(readouts.observed,mask_camera)
        encoded=bgr[y:b,x:r,::-1].astype(float)
        fg=np.linalg.norm(encoded-color_for(1),axis=-1)
        bg=np.linalg.norm(encoded-np.array([211,219,225]),axis=-1)
        agreement=float(np.mean((fg<bg)[3:-3,3:-3]==mask[3:-3,3:-3]))
        assert agreement>.995,(i,'encoded membership',agreement)
        checks['segmentation_encoded_agreement']=agreement
        query_pairs=[(source,readouts.observed,'observed'),
                     (cameras[0],readouts.scene,'novel_left'),
                     (cameras[1],readouts.scene,'novel')]
        for camera,surface,rgb_key in query_pairs:
            x,y,r,b=boxes[rgb_key];scale=(r-x)/camera.width
            cam=Camera(camera.position,camera.rotation,camera.focal*scale,r-x,b-y)
            rgb,_,mask,coverage=project_readouts(surface,cam)
            if camera is not source:
                audit_cam=Camera(camera.position,camera.rotation,camera.focal*.875,560,315)
                _,_,visible_mask,small_coverage=project_readouts(surface,audit_cam)
                _,_,isolated_mask,_=project_readouts(readouts.figure,audit_cam)
                retained=float(visible_mask.sum()/max(isolated_mask.sum(),1))
                assert retained>=.92,(i,rgb_key,'figure hidden by ground',retained)
                checks[rgb_key+'_visible_figure_fraction']=retained
                missing,gap=coverage_statistics(small_coverage)
                assert missing<=.001 and gap<=96,(i,rgb_key,missing,gap)
                uv,z=camera.project(readouts.figure.vertices)
                assert np.all(z>.5) and np.all(uv>=[4,4]) and np.all(uv<=[636,356])
                checks[rgb_key+'_missing_fraction']=missing
                checks[rgb_key+'_largest_gap_pixels']=gap
            encoded_rgb=bgr[y:b,x:r,::-1].astype(float)
            encoded_error=float(np.abs(encoded_rgb[4:-4,4:-4]-rgb[4:-4,4:-4]).mean())
            assert encoded_error<2.,(i,rgb_key,'encoded RGB mismatch',encoded_error)
            checks[rgb_key+'_encoded_rgb_mae']=encoded_error
        # Fit visible foreground; fading particles remain drawn but may crop.
        use=(f['object_ids']>0)&(f['display_alpha']>.35)
        local=(f['means'][use]-viewer.position)@viewer.rotation.T
        covariance=np.einsum('ij,njk,lk->nil',viewer.rotation,f['covs'][use],viewer.rotation)
        for axis,size in [(0,viewer.width),(1,viewer.height)]:
            for sign in (-1,1):
                normal=np.zeros(3);normal[axis]=sign;normal[2]=-size/(2*viewer.focal)
                support=1.8*np.sqrt(np.einsum('i,nij,j->n',normal,covariance,normal))
                margin=-(local@normal+support)
                if len(margin): margin_min=min(margin_min,float(margin.min()))
                assert np.all(margin>=-1e-7),(i,'foreground clipped in particle viewer')
        uv,z=detail.project(readouts.figure.vertices)
        assert np.all(z>0) and np.all(uv>0) and np.all(uv<[detail.width,detail.height]),(i,'surface clipped')
        if previous is not None:
            x,y,r,b=boxes['scene']
            transitions.append(dict(frame=i+1,scene_mean_pixel_change=float(
                np.abs(bgr[y:b,x:r].astype(float)-previous[y:b,x:r]).mean())))
        previous=bgr
        if i%6==0:
            page=Image.new('RGB',(2800,2100),'white');draw=ImageDraw.Draw(page)
        encoded=Image.fromarray(bgr[...,::-1]);x=(i%2)*1400;y=((i%6)//2)*700
        crops=[('scene',0,36,700,346),('input',710,36,340,191),('surface',1060,36,340,244),
               ('segmentation',710,238,340,191),('observed',0,480,450,197),
               ('novel_left',470,480,450,197),('novel',940,480,450,197)]
        if saved['ground_available']: crops.append(('ground',1060,280,300,190))
        for key,dx,dy,w,h in crops:
            tile=encoded.crop(boxes[key]);tile.thumbnail((w,h),Image.Resampling.LANCZOS)
            page.paste(tile,(x+dx,y+dy))
        text(draw,(x+8,y+2),f"{capture.manifest['clip']} · {i+1}/{capture.manifest['frames']}",23)
        if (i+1)%6==0 or i+1==capture.manifest['frames']:
            path=out/f'{video.stem}_{i//6:02d}.jpg';page.save(path,quality=94);pages.append(path.name)
        frames.append(checks)
        if (i+1)%20==0: print(f'Audited {video.stem}: {i+1}/{capture.manifest["frames"]}',flush=True)
    assert not decoder.read()[0], 'Unexpected extra encoded frame'
    decoder.release()
    warmup=int(capture.manifest['config'].get('warmup_frames',3));seconds=np.asarray(seconds)
    rates=[len(seconds[max(warmup,i-14):i+1])/seconds[max(warmup,i-14):i+1].sum()
           for i in range(warmup,len(seconds))]
    np.testing.assert_allclose(saved['rollout_fps'][warmup:],rates,atol=1e-12,rtol=0)
    result=dict(clip=capture.manifest['clip'],frames=len(frames),checks=frames,sheets=pages,
        source_mask_iou_min=min(f['source_mask_iou'] for f in frames),
        source_mask_iou_mean=float(np.mean([f['source_mask_iou'] for f in frames])),
        source_rgb_mae_mean=float(np.mean([f['source_rgb_mae'] for f in frames])),
        foreground_frustum_margin_min=margin_min,camera_policy=report['camera_policy'],
        previous_display_geometry_identical=old is not None,rollout_timing_verified=True,
        view_selection=report['novel_view_selection'],novel_poses=pose_checks,
        ground_surface_shown=bool(saved['ground_available']),
        original_frustum_verified=True,ground_original_frustum_verified=True,ground_fixed_framing_verified=True,
        largest_transitions=sorted(transitions,key=lambda x:x['scene_mean_pixel_change'],reverse=True)[:5],
        renderer_implementation_sha256=implementation_fingerprint())
    write_json(out/f'{video.stem}.json',result)
    print(f"Audited {video.stem}: {len(frames)} frames, two real offset poses, source-mask IoU {result['source_mask_iou_min']:.4f}+",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos',type=Path,nargs='+')
    for video in parser.parse_args().videos: audit(video)


if __name__=='__main__': main()
