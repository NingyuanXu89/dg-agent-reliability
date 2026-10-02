import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from dg_euler.core import *
from dg_euler.reference import riemann

class EulerTests(unittest.TestCase):
    def setUp(self):
        self.c=EulerConfig(nx=8,ny=8,degree=2);self.s=EulerDG(self.c)
    def test_round_trip(self):
        q=np.array([[1.,.4,-.2,2.],[2.,-3.,4.,.1]])
        u=conservative(*q.T,self.c.gamma)
        np.testing.assert_allclose(primitive(u,self.c.gamma),q,atol=1e-14)
    def test_known_ideal_gas_state(self):
        # rho=2, velocity=(3,-2), kinetic energy=13, internal energy=7.
        u=np.array([2.,6.,-4.,20.])
        self.assertAlmostEqual(float(pressure(u,1.4)),2.8)
        np.testing.assert_allclose(conservative(2,3,-2,2.8,1.4),u)
    def test_consistent_flux(self):
        for vx,vy in [(0,0),(3,-2),(-3,2),(.3,-.2)]:
            u=conservative(1,vx,vy,1,self.c.gamma)
            for axis in (0,1):
                f,_=numerical_flux(u,u,self.c.gamma,axis)
                np.testing.assert_allclose(f,physical_flux(u,self.c.gamma,axis),atol=1e-14)
    def test_rotation(self):
        l=conservative(1,.3,-.7,1,self.c.gamma);r=conservative(2,-.5,.2,.4,self.c.gamma);swap=[0,2,1,3]
        f,_=numerical_flux(l,r,self.c.gamma,0);g,_=numerical_flux(l[swap],r[swap],self.c.gamma,1)
        np.testing.assert_allclose(f,g[swap])
    def test_contacts_and_supersonic(self):
        for speed in (0,.3,5,-5):
            l=conservative(1,speed,.2,1,self.c.gamma);r=conservative(2,speed,-.4,1,self.c.gamma)
            f,n=numerical_flux(l,r,self.c.gamma,0)
            np.testing.assert_allclose(f,physical_flux(l if speed>=0 else r,self.c.gamma,0),atol=1e-13)
            self.assertEqual(n,0)
    def test_fallback(self):
        l=conservative(1,-10,0,.1,self.c.gamma);r=conservative(1,10,0,.1,self.c.gamma)
        f,n=numerical_flux(l,r,self.c.gamma,0);expected,_=numerical_flux(l,r,self.c.gamma,0,'llf')
        self.assertEqual(n,1);np.testing.assert_allclose(f,expected)
    def test_projection(self):
        initial=lambda x,y:np.broadcast_to((1+2*x+3*y+x*y)[...,None],np.broadcast_shapes(x.shape,y.shape)+(4,))
        a=self.s.project(initial);u=self.s.reconstruct(a)
        x=(np.arange(8)[None,:,None,None]+.5*(1+self.s.z)[None,None,None,:])/8
        y=(np.arange(8)[:,None,None,None]+.5*(1+self.s.z)[None,None,:,None])/8
        np.testing.assert_allclose(u,initial(x,y),atol=2e-14)
    def test_free_stream(self):
        a=self.s.project(lambda x,y:conservative(np.ones(np.broadcast_shapes(x.shape,y.shape)),.3,-.2,1,self.c.gamma))
        self.assertLess(abs(self.s.residual(a)).max(),2e-12)
        np.testing.assert_allclose(self.s.step(a,.001),a,atol=1e-13)
    def test_conservation(self):
        a=self.s.project(khi_initial(self.c))
        np.testing.assert_allclose(self.s.residual(a)[:,:,0,0].sum((0,1)),0,atol=1e-12)
    def test_limiter_means(self):
        a=self.s.project(khi_initial(self.c));a[:,:,0,1,0]=2
        means=a[:,:,0,0].copy();a=self.s.tvb_limit(a);np.testing.assert_array_equal(a[:,:,0,0],means)
        self.assertGreater(self.s.limited,0)
    def test_positivity_means(self):
        a=self.s.project(lambda x,y:conservative(np.ones(np.broadcast_shapes(x.shape,y.shape)),0,0,1,self.c.gamma));a[:,:,0,1,0]=2;a[:,:,1,0,3]=4
        means=a[:,:,0,0].copy();a=self.s.positivity(a);u=self.s.check_values(a)
        np.testing.assert_array_equal(a[:,:,0,0],means)
        self.assertGreaterEqual(u[...,0].min(),self.c.floor*.99)
        self.assertGreaterEqual(pressure(u,self.c.gamma).min(),self.c.floor*.99)
        self.assertGreater(self.s.scaled,0)
    def test_retry(self):
        a=self.s.positivity(self.s.project(khi_initial(self.c)));original=self.s.step
        def step(a,dt):
            if dt>.001:raise StepRejected('deliberately rejected')
            return original(a,dt)
        self.s.step=step;b,dt=self.s.advance(a,.004)
        self.assertEqual(dt,.001);self.assertEqual(self.s.rejected,2);self.assertTrue(np.isfinite(b).all())
    def test_bad_means(self):
        a=np.zeros((8,8,3,3,4))
        with self.assertRaises(StepRejected):self.s.positivity(a)
    def test_reference(self):
        u=riemann(np.array([.1,.5,.9]),.1)
        np.testing.assert_allclose(primitive(u,1.4)[[0,2]],[[1,0,0,1],[.125,0,0,.1]],atol=1e-12)
        self.assertGreater(u[1,1],0)
    def test_terminal_failure_checkpoint(self):
        cfg=replace(self.c,final_time=.01)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(EulerDG,'step',side_effect=StepRejected('injected terminal failure')):
                with self.assertRaises(StepRejected):simulate(cfg,times=[0,.01],output=directory,accelerated=False)
            saved=np.load(Path(directory)/'failure_state.npz')
            self.assertEqual(float(saved['time']),0)
            self.assertTrue(np.isfinite(saved['coefficients']).all())
            self.assertTrue((Path(directory)/'failure.json').exists());saved.close()
    def test_zero_time_and_nonfinite_times(self):
        result=simulate(replace(self.c,final_time=0))
        self.assertEqual(len(result['history']),1)
        with self.assertRaises(ValueError):simulate(self.c,times=[0,float('nan')])
    def test_cell_vorticity(self):
        initial=lambda x,y:conservative(np.ones(np.broadcast_shapes(x.shape,y.shape)),0,2*x,1,self.c.gamma)
        a=self.s.project(initial);rho,vort=self.s.fields(a)
        np.testing.assert_allclose(rho,1,atol=1e-6);np.testing.assert_allclose(vort,2,atol=1e-6)
    def test_native_matches_numpy(self):
        from dg_euler.native import library_path,NativeKernel
        if not library_path().exists():self.skipTest('optional native kernel not built')
        for degree in (1,2):
            cfg=replace(self.c,degree=degree);solver=EulerDG(cfg);native=NativeKernel(solver)
            a=solver.positivity(solver.project(khi_initial(cfg)));dt=solver.timestep(a)
            self.assertAlmostEqual(dt,native.timestep(a),places=14)
            np.testing.assert_allclose(native.step(a,dt),solver.step(a,dt),atol=3e-13,rtol=1e-13)
        # Strong high modes require both density and pressure scaling.
        solver=EulerDG(replace(self.c,limiting=False));native=NativeKernel(solver)
        a=solver.project(lambda x,y:conservative(np.ones(np.broadcast_shapes(x.shape,y.shape)),0,0,1,self.c.gamma))
        a[:,:,0,1,0]=2;a[:,:,1,0,3]=4
        expected=solver.stabilize(a.copy());actual=native.stabilize(a)
        np.testing.assert_allclose(actual,expected,atol=2e-12)
    def test_resume_configuration_and_schedule(self):
        cfg=replace(self.c,final_time=.01)
        with tempfile.TemporaryDirectory() as directory:
            first=simulate(cfg,times=[0,.005,.01],output=directory)
            resumed=simulate(cfg,times=[0,.005,.01],output=directory,resume=True)
            np.testing.assert_array_equal(resumed['coefficients'],first['coefficients'])
            with self.assertRaises(ValueError):simulate(replace(cfg,cfl=.1),times=[0,.005,.01],output=directory,resume=True)
            with self.assertRaises(ValueError):simulate(cfg,times=[0,.002,.01],output=directory,resume=True)
    def test_degenerate_flux(self):
        u=conservative(1,0,0,1e-30,self.c.gamma)
        f,n=numerical_flux(u,u,self.c.gamma,0)
        self.assertEqual(n,1);np.testing.assert_allclose(f,physical_flux(u,self.c.gamma,0),atol=1e-40)
    def test_nonfinite_reconstruction_rejected(self):
        a=np.zeros((8,8,3,3,4));a[:,:,0,0]=[1.5e308,0,0,1];a[:,:,0,1,0]=1e308
        with self.assertRaises(StepRejected):self.s.positivity(a)
    def test_nonfinite_wave_speed_diagnosed(self):
        cfg=replace(self.c,gamma=1e200,final_time=.01)
        with tempfile.TemporaryDirectory() as directory:
            with np.errstate(over='ignore',invalid='ignore'):
                with self.assertRaises(StepRejected):simulate(cfg,initial=lambda x,y:np.array([1.,0,0,1.]),times=[0,.01],output=directory)
            self.assertTrue((Path(directory)/'failure_state.npz').exists())
    def test_invalid_configuration(self):
        for kwargs in [dict(degree=0),dict(gamma=1),dict(nx=0),dict(cfl=-1),dict(flux='bad')]:
            with self.assertRaises(ValueError):EulerConfig(**kwargs)

if __name__=='__main__':unittest.main()
