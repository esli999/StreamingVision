"""Novel views use offset extrinsics, layered geometry and model appearance."""
import numpy as np
import cv2
from unified.geometry import Camera
from unified.novel import layered_readouts, offset_cameras, choose_offset_views, shaded_surface,ground_fit_points
from unified.projection import project_readouts


def state():
    labels=np.zeros((360,640),np.uint16); labels[95:265,240:400]=1
    rgb=np.full((360,640,3),[30,80,130],np.uint8); rgb[labels>0]=[180,50,25]
    return dict(labels=labels,reconstruction=rgb,bgr=np.zeros_like(rgb),
        means=np.array([[0.,0.,4.],[0.,0.,2.]]),
        covs=np.array([np.diag([9.,9.,.02]),np.diag([.2,.2,.01])]),
        weights=np.array([.6,.4]),object_ids=np.array([0,1]),
        display_alpha=np.ones(2),colors=np.array([[30,80,130],[180,50,25]],np.uint8))


def test_offset_poses_translate_rotate_and_produce_depth_dependent_parallax():
    target=np.array([0.,0.,2.]); cameras=offset_cameras(target,[-12.,12.])
    points=np.array([[0.,0.,2.],[0.,0.,4.]])
    projections=[]
    for camera in cameras:
        assert np.linalg.norm(camera.position)>.4
        assert not np.allclose(camera.rotation,np.eye(3))
        np.testing.assert_allclose(camera.rotation[2],(target-camera.position)/np.linalg.norm(target-camera.position))
        uv,_=camera.project(points)
        assert abs(uv[0,0]-uv[1,0])>40
        projections.append(uv)
    assert (projections[0][1,0]-320)*(projections[1][1,0]-320)<0


def test_ground_layer_is_a_particle_query_and_can_be_revealed_behind_the_figure():
    f=state(); readouts=layered_readouts(f)
    camera=offset_cameras(np.array([0.,0.,2.]),[-28.],focal=410.)[0]
    rgb,_,mask,coverage=project_readouts(readouts.scene,camera)
    assert np.all(coverage==1) and mask.any()
    interior=cv2.erode((~mask).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    np.testing.assert_allclose(rgb[interior],np.broadcast_to([130,80,30],rgb[interior].shape),atol=1)
    # Neither the input photo nor captured background pixels supply hidden ground.
    f['bgr'][:]=255; f['reconstruction'][f['labels']==0]=255
    changed=project_readouts(layered_readouts(f).scene,camera)
    np.testing.assert_array_equal(rgb,changed[0])
    np.testing.assert_array_equal(mask,changed[2])
    # The SAME RGB and semantic geometry is rendered, not a copied source mask.
    assert np.count_nonzero(mask!=(f['labels']>0))>1000


def test_unrepresented_ground_stays_absent_and_surface_has_no_identity_highlight():
    f=state(); f['display_alpha'][0]=0
    readouts=layered_readouts(f)
    assert not len(readouts.ground.faces)
    camera=Camera(np.zeros(3),np.eye(3),500.,640,360)
    assert project_readouts(readouts.scene,camera)[3][10,10]==0
    surface=shaded_surface(readouts.figure)
    assert not surface.highlight and surface.object_id==0
    assert np.isfinite(surface.normals).all()
    np.testing.assert_array_equal(surface.colors,readouts.figure.colors)


def test_two_view_selection_checks_late_frames_and_does_not_switch_pose():
    class Saved:
        manifest={'frames':2}
        def read(self,i):
            f=state()
            if i: f['display_alpha'][0]=0
            return f
    class Display:
        stabilized=True
        manifest={'clip':'synthetic'}
        novel_target=np.array([0.,0.,2.])
        def frame(self,f,i): return f
    cameras,report,_=choose_offset_views(Saved(),Display())
    assert not cameras and not report['selected_degrees']
    assert max(report['evaluated_frames'])==2
    assert not any(report['passed'])


def test_ground_viewer_fits_complete_deforming_sheets_without_empty_bounding_corners():
    from unified.presentation import fixed_scene_camera
    rng=np.random.default_rng(42)
    points=rng.normal(size=(3,400,3))*[1.5,.6,.1]
    points[:,:,2]+=3.+.6*points[:,:,0]**2
    support=np.concatenate([ground_fit_points(p) for p in points])
    def fit(p):
        return fixed_scene_camera(p[None],np.broadcast_to(np.eye(3)*1e-6,(1,len(p),3,3)),
                                  np.ones((1,len(p))),1024,576,yaw=20,pitch=14,focal=1100)
    compact,full=fit(support),fit(points.reshape(-1,3))
    np.testing.assert_allclose(compact.position,full.position,atol=1e-12)
    uv,z=compact.project(points.reshape(-1,3))
    assert np.all(z>0) and np.all(uv>0) and np.all(uv<[1024,576])


def test_complete_coverage_does_not_qualify_a_pose_that_hides_the_figure(monkeypatch):
    import unified.novel as novel
    readouts=layered_readouts(state())
    # An opaque ground sheet in front of the figure, despite full raster coverage.
    count=len(readouts.ground.vertices)
    readouts.scene.vertices[:count]*=1.3/4.
    monkeypatch.setattr(novel,'layered_readouts',lambda frame:readouts)
    class Saved:
        manifest={'frames':1}
        def read(self,i):
            return state()
    class Display:
        stabilized=True
        manifest={'clip':'occluded'}
        novel_target=np.array([0.,0.,2.])
        def frame(self,f,i): return f
    cameras,report,_=choose_offset_views(Saved(),Display())
    assert not cameras
    assert max(report['maximum_missing_fraction'])==0
    assert max(report['minimum_visible_figure_fraction'])<.92


def test_hidden_ground_respects_observed_occlusion_without_changing_appearance():
    f=state();f['means'][0,2]=1.3
    readouts=layered_readouts(f)
    assert readouts.ground_occlusion_adjusted_fraction>0
    source=Camera(np.zeros(3),np.eye(3),500.,640,360)
    _,_,mask,_=project_readouts(readouts.scene,source)
    expected=f['labels']>0
    assert (mask&expected).sum()/expected.sum()>.999
    uv,_=source.project(readouts.ground.vertices)
    away=(uv[:,0]<230)|(uv[:,0]>410)|(uv[:,1]<85)|(uv[:,1]>275)
    rays=(uv[away]-[320.,180.])/500.
    expected_depth=1.3/(1+(.02+1e-8)/(9.+1e-8)*np.sum(rays*rays,axis=1))
    np.testing.assert_allclose(readouts.ground.vertices[away,2],expected_depth,atol=1e-6)
    np.testing.assert_allclose(readouts.ground.colors,
                               np.broadcast_to([130,80,30],readouts.ground.colors.shape),atol=1e-5)
