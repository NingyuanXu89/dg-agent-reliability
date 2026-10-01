"""Independent invariants and numerical acceptance tests (standard unittest)."""
import unittest
import numpy as np
from numpy.polynomial import legendre as leg
from dg_burgers.core import Config, DG, simulate, numerical_flux
from dg_burgers.reference import exact_solution, shock_position, shock_foot
from dg_burgers.diagnostics import errors, coefficient_distance


class DGTests(unittest.TestCase):
    def test_constant_preservation(self):
        for p in (1,2,3):
            for value in (-1.,0.,1.5):
                cfg=Config(cells=8,degree=p,final_time=.1)
                dg=DG(cfg)
                c=dg.project(lambda x: np.full_like(x,value))
                np.testing.assert_allclose(dg.rhs(c),0,atol=2e-12)
                result=simulate(cfg,initial=lambda x: np.full_like(x,value),times=[0.,.1])
                np.testing.assert_allclose(result.coefficients[-1],c,atol=2e-13)

    def test_flux_consistency_and_states(self):
        u=np.array([-2.,0.,1.5])
        np.testing.assert_allclose(numerical_flux(u,u),.5*u*u)
        # Hand calculations: positive, negative, rarefaction, compressive shock.
        for a,b,expected in ((1.,2.,.25),(-2.,-1.,.25),(-1.,1.,-.5),(1.,-1.,1.5)):
            self.assertAlmostEqual(float(numerical_flux(np.array(a),np.array(b))),expected)

    def test_mass_rhs_telescopes(self):
        rng=np.random.default_rng(3)
        for p in (1,2,3):
            dg=DG(Config(cells=11,degree=p))
            c=rng.normal(size=(11,p+1))
            self.assertLess(abs(dg.h*sum(dg.rhs(c)[:,0])),1e-12)

    def test_projection_of_local_polynomial(self):
        for p in (1,2,3):
            dg=DG(Config(cells=7,degree=p))
            expected=np.arange(1,p+2,dtype=float)/10
            def polynomial(x):
                ids=np.minimum(((x-dg.config.origin)/dg.h).astype(int),6)
                xi=2*(x-dg.centers[ids])/dg.h
                return leg.legval(xi,expected)
            np.testing.assert_allclose(dg.project(polynomial),np.tile(expected,(7,1)),atol=2e-14)

    def test_limiter_preserves_means(self):
        dg=DG(Config(cells=8,degree=3,tvb_m=0))
        c=np.random.default_rng(1).normal(size=(8,4))
        limited,count=dg.limit(c)
        self.assertGreater(count,0)
        np.testing.assert_array_equal(limited[:,0],c[:,0])
        self.assertAlmostEqual(dg.summary(c)["mass"],dg.summary(limited)["mass"])

    def test_exact_polynomial_extrema(self):
        for p in (1,2,3):
            dg=DG(Config(cells=10,degree=p))
            c=np.random.default_rng(p).normal(size=(10,p+1))
            lo,hi=dg.bounds(c)
            dense=c@leg.legvander(np.linspace(-1,1,10001),p).T
            self.assertTrue(np.all(lo<=dense.min(axis=1)+1e-12))
            self.assertTrue(np.all(hi>=dense.max(axis=1)-1e-12))
            np.testing.assert_allclose(lo,dense.min(axis=1),atol=1e-7)
            np.testing.assert_allclose(hi,dense.max(axis=1),atol=1e-7)

    def test_reference_characteristics(self):
        xi=np.array([.2,.8,1.2,5.,5.8])
        for t in (.25,.75,1.,1.5):
            if t>1:
                q=shock_foot(t)
                xi=np.array([.1,q*.5,2*np.pi-q*.5,6.1])
                self.assertAlmostEqual(q+t*np.sin(q),np.pi,places=12)
                # Rankine--Hugoniot: shock speed equals 1.
                ul,ur=1+np.sin(q),1-np.sin(q)
                self.assertAlmostEqual((.5*ul**2-.5*ur**2)/(ul-ur),1)
            x=xi+t*(1+np.sin(xi))
            np.testing.assert_allclose(exact_solution(x,t),1+np.sin(xi),atol=2e-12)
            np.testing.assert_allclose(exact_solution(x+2*np.pi,t),exact_solution(x,t),atol=2e-12)
        self.assertEqual(float(exact_solution(shock_position(1.5),1.5)),1.)

    def test_reference_satisfies_smooth_pde(self):
        x=np.array([.4,1.4,2.2,4.9,5.7]);t=.3;eps=1e-5
        ut=(exact_solution(x,t+eps)-exact_solution(x,t-eps))/(2*eps)
        flux_x=(.5*exact_solution(x+eps,t)**2-.5*exact_solution(x-eps,t)**2)/(2*eps)
        np.testing.assert_allclose(ut+flux_x,0,atol=3e-8)

    def test_constant_error_integral(self):
        dg=DG(Config(cells=9,degree=2,final_time=0))
        c=dg.project(lambda x:1+np.sin(x));e=errors(dg,c,0)
        self.assertLess(e["l2"],.01)
        self.assertGreater(e["l2"],0)
        self.assertAlmostEqual(coefficient_distance(dg,c,c+.0,0),0.)

    def test_smooth_convergence(self):
        for p in (1,2,3):
            es=[]
            for n in (16,32,64):
                cfg=Config(cells=n,degree=p,final_time=.5,cfl=.025,
                           limiter=False,time_accuracy=True)
                r=simulate(cfg,times=[0.,.5])
                es.append(errors(DG(cfg),r.coefficients[-1],.5)["l2"])
                self.assertLess(abs(r.diagnostics[-1]["mass_drift"]),1e-10)
            rate=np.log2(es[-2]/es[-1])
            self.assertLess(abs(rate-(p+1)),.4)

    def test_shock_finite_conservative_and_refines(self):
        es=[]
        for n in (32,64):
            cfg=Config(cells=n,degree=2,final_time=1.5)
            r=simulate(cfg,times=[0.,1.5]);d=r.diagnostics[-1]
            self.assertTrue(np.all(np.isfinite(r.coefficients)))
            self.assertLess(abs(d["mass_drift"]),1e-10)
            self.assertGreater(d["limiter_activations"],0)
            self.assertGreater(d["minimum"],-.05)
            self.assertLess(d["maximum"],2.05)
            es.append(errors(DG(cfg),r.coefficients[-1],1.5)["l1"])
        self.assertLess(es[1],es[0])

    def test_zero_state_with_reduced_step(self):
        cfg=Config(cells=8,degree=3,final_time=.2,dt_scale=.5,time_accuracy=True)
        r=simulate(cfg,initial=lambda x:np.zeros_like(x),times=[0.,.1,.2])
        self.assertEqual(r.steps,2)
        np.testing.assert_array_equal(r.coefficients,0.)

    def test_all_degrees_cross_shock(self):
        for p in (1,2,3):
            cfg=Config(cells=48,degree=p,final_time=1.5)
            r=simulate(cfg,times=[0.,1.,1.5])
            self.assertTrue(np.all(np.isfinite(r.coefficients)))
            self.assertLess(abs(r.diagnostics[-1]["mass_drift"]),1e-10)
            self.assertGreater(r.diagnostics[-1]["minimum"],-.05)
            self.assertLess(r.diagnostics[-1]["maximum"],2.05)

    def test_nonfinite_stage_is_rejected(self):
        dg=DG(Config(cells=4,degree=1))
        c=np.full((4,2),np.nan)
        with self.assertRaises(FloatingPointError):dg.step(c,.01)

    def test_invalid_configuration_and_times(self):
        for kw in (dict(cells=2),dict(degree=4),dict(degree=2.0),dict(cfl=0),dict(final_time=-1),dict(tvb_m=-1),dict(origin=np.nan)):
            with self.assertRaises(ValueError): Config(**kw)
        with self.assertRaises(ValueError):simulate(Config(final_time=.1),times=[0.,.2])
        with self.assertRaises(ValueError):simulate(Config(final_time=.1),initial=lambda x:np.nan*x)


if __name__=="__main__":
    unittest.main()
