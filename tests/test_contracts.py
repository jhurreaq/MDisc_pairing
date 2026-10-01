"""Independent mechanics, numerical and execution contracts for the corrected project."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from mdisc.mechanics import cardiac_library, isotropic_library
from mdisc.types import Observations
from mdisc.preprocessing import WeightPolicy, DesignScaling
from mdisc.regression import fit_nonnegative
from mdisc.criteria import information_criteria
from mdisc.paths import distinct_supports, discovery_path, PathSettings
from mdisc.discovery import grouped_mode_folds, discover, DiscoverySettings
from mdisc.runner import run_case

ROOT = Path(__file__).resolve().parents[1]


def fixture(points=8):
    F = np.array([np.diag([l, l**-0.5, l**-0.5]) for l in np.linspace(0.8, 1.8, points)])
    library = isotropic_library(mr_order=1)
    y = library.stress(F, np.array([0.5, 2.0]), measure='nominal')[:,0,0]
    return Observations(F, y, ((0,0),)*points, np.array(['UT']*points), np.arange(points), 'nominal', 'MPa')


class Mechanics(unittest.TestCase):
    def test_registry_preserves_original_32_ids_and_disables_12_terms(self):
        library = cardiac_library()
        self.assertEqual([t.term_id for t in library.terms], list(range(1,33)))
        self.assertEqual([t.index for t in library.terms if not t.enabled], [8,9,12,13,16,17,20,21,24,25,28,29])
        self.assertEqual(len(library.active_terms), 20)
        beta = library.expand(np.ones(20))
        self.assertTrue(np.all(beta[[8,9,12,13,16,17,20,21,24,25,28,29]] == 0))
        beta[8] = 1
        with self.assertRaises(ValueError): library.compact(beta)
        with self.assertRaises(ValueError): library.energy(np.eye(3), np.ones(20), {9:1.0})

    def test_each_enabled_term_has_zero_reference_energy_and_stress(self):
        for library in (cardiac_library(), isotropic_library(mr_order=4,ogden_exponents=(-4,-1,1,3,4))):
            for i in range(len(library.active_terms)):
                beta = np.eye(len(library.active_terms))[i]
                self.assertAlmostEqual(float(library.energy(np.eye(3),beta)), 0)
                np.testing.assert_allclose(library.stress(np.eye(3),beta), 0, atol=1e-12)

    def test_analytical_energy_derivatives_match_independent_finite_differences(self):
        F = np.array([[1.2,.17,.08],[.03,.9,.04],[.02,.09,1.1]])
        F /= np.linalg.det(F)**(1/3)
        h = 1e-6
        for library in (cardiac_library(), isotropic_library(mr_order=3,ogden_exponents=(-4,-1,1,3,4))):
            shapes = {t.index:1.3 for t in library.active_terms if t.nonlinear}
            derivatives = library._term_values(F,shapes)[1]
            for i in range(len(library.active_terms)):
                beta = np.eye(len(library.active_terms))[i]
                for a in range(3):
                    for b in range(3):
                        H = np.zeros((3,3)); H[a,b] = h
                        numeric = (library.energy(F+H,beta,shapes)-library.energy(F-H,beta,shapes))/(2*h)
                        np.testing.assert_allclose(derivatives[i,a,b],numeric, rtol=2e-5,atol=2e-7)

    def test_cardiac_basis_is_exact_full_tensor_prediction(self):
        library = cardiac_library()
        F = np.array([np.diag([1.2,1/1.2,1]), [[1,.2,0],[0,1,0],[0,0,1]]])
        obs = Observations(F,np.zeros(2),((0,0),(0,1)),np.array(['A','B']),np.arange(2),'cauchy','kPa')
        X = library.design(obs)
        for i in range(20):
            beta = np.eye(20)[i]
            stresses = library.stress(F,beta)
            np.testing.assert_allclose(X@beta,[stresses[0,0,0],stresses[1,0,1]],atol=1e-13)
        position = next(i for i,t in enumerate(library.active_terms) if t.index == 10)
        self.assertAlmostEqual(X[0,position], 2.5344,places=12)

    def test_tension_only_fibres_do_not_respond_under_compression(self):
        library = cardiac_library()
        i = next(i for i,t in enumerate(library.active_terms) if t.index == 10)
        beta = np.eye(20)[i]
        F = np.diag([.8,1/.8,1])
        self.assertEqual(float(library.energy(F,beta)),0)
        np.testing.assert_array_equal(library.stress(F,beta),np.zeros((3,3)))

    def test_nonlinear_shape_stress_jacobian(self):
        library = cardiac_library(); obs = fixture()
        obs = Observations(obs.F,obs.y,obs.components,obs.modes,obs.groups,'cauchy','kPa')
        for i,term in enumerate(library.active_terms):
            if not term.nonlinear: continue
            h=1e-6
            analytical=library.design(obs,{term.index:1.3},shape_derivative=True)[:,i]
            numeric=(library.design(obs,{term.index:1.3+h})[:,i]-library.design(obs,{term.index:1.3-h})[:,i])/(2*h)
            np.testing.assert_allclose(analytical,numeric,rtol=2e-6,atol=1e-7)

    def test_frame_indifference_and_symmetric_cauchy_stress(self):
        Q = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        F = np.array([[1,.2,.1],[0,1,.3],[0,0,1.]])
        for library in (cardiac_library(), isotropic_library(mr_order=2,ogden_exponents=(-3,3))):
            beta=np.ones(len(library.active_terms))
            self.assertAlmostEqual(float(library.energy(Q@F,beta)),float(library.energy(F,beta)),places=10)
            sigma=library.stress(F,beta)
            rotated=library.stress(Q@F,beta,normal=Q@library.pressure_normal)
            np.testing.assert_allclose(rotated,Q@sigma@Q.T,atol=1e-10)
            np.testing.assert_allclose(sigma,sigma.T,atol=1e-10)

    def test_orthotropic_material_axis_reflections(self):
        library=cardiac_library(); beta=np.ones(20)
        F=np.array([[1,.2,.1],[0,1,.3],[0,0,1.]])
        Q=np.diag([-1.,-1.,1.])
        self.assertAlmostEqual(float(library.energy(F@Q,beta)),float(library.energy(F,beta)),places=12)

    def test_nominal_stress_is_work_conjugate_along_isochoric_rank_one_path(self):
        library=cardiac_library(); beta=np.ones(20)
        F=np.diag([1.2,1/1.2,1.]); H=np.zeros((3,3)); H[0,1]=1
        h=1e-6
        derivative=(library.energy(F+h*H,beta)-library.energy(F-h*H,beta))/(2*h)
        self.assertAlmostEqual(float(derivative),float(np.sum(library.stress(F,beta,measure='nominal')*H)),places=7)

    def test_neo_hookean_uniaxial_formula(self):
        library=isotropic_library(mr_order=1)
        lam=1.6; c=2.0; F=np.diag([lam,lam**-.5,lam**-.5])
        self.assertAlmostEqual(library.stress(F,np.array([0,c]),measure='nominal')[0,0],2*c*(lam-lam**-2),places=12)

    def test_ogden_initial_shear_modulus(self):
        alpha=-3.; c=2.; library=isotropic_library(mr_order=0,ogden_exponents=(alpha,))
        F=np.eye(3); F[0,1]=1e-5
        self.assertAlmostEqual(library.stress(F,np.array([c]))[0,1]/1e-5,.5*c*alpha**2,places=7)

    def test_invalid_material_inputs_raise(self):
        library=cardiac_library()
        with self.assertRaises(ValueError): library.stress(2*np.eye(3),np.ones(20))
        with self.assertRaises(ValueError): library.stress(np.eye(3),-np.ones(20))
        with self.assertRaises(ValueError): library.energy(np.eye(3),np.ones(20),{1:-1})
        with self.assertRaises(FloatingPointError): library.energy(np.diag([2.,.5,1]),np.ones(20),{1:1000})

    def test_pressure_is_a_boundary_reaction_not_a_material_parameter(self):
        library=cardiac_library(); beta=np.ones(20)
        F=np.diag([1.2,1/1.2,1.])
        raw=library.stress(F,beta,pressure=0)
        shifted=library.stress(F,beta,pressure=2)
        np.testing.assert_allclose(raw-shifted,2*np.eye(3),atol=1e-13)
        self.assertAlmostEqual(library.stress(F,beta)[1,1],0,places=13)


class Statistics(unittest.TestCase):
    def test_nonlinear_cardiac_refit_recovers_a_known_energy_term(self):
        from Anisotropic.refinement import refine
        from mdisc.discovery import RefinementSettings
        library=cardiac_library()
        i=next(i for i,t in enumerate(library.active_terms) if t.index==23)
        F=np.repeat(np.eye(3)[None],20,axis=0)
        F[:,1,0]=np.linspace(-.5,.5,20)
        obs=Observations(F,np.zeros(20),((0,1),)*20,np.array(['fs']*20),np.arange(20),'cauchy','kPa')
        beta=np.zeros(20);beta[i]=2
        obs=obs.with_targets(library.predict(obs,beta,{23:3.0}))
        X=library.design(obs);weights=np.ones(20);scaling=DesignScaling.fit(X,obs.y,weights)
        fit=refine(library,obs,X,(i,),weights,scaling,DiscoverySettings(ridge=0))
        np.testing.assert_allclose(fit.prediction,obs.y,rtol=1e-7,atol=1e-9)
        self.assertAlmostEqual(fit.nonlinear[23],3,places=5)
        self.assertAlmostEqual(fit.coefficients[i],2,places=5)

    def test_weighted_affine_nnls_recovers_exact_line(self):
        X=np.arange(3.)[:,None]; y=1+2*X[:,0]
        fit=fit_nonnegative(X,y,weights=[1,4,9],fit_intercept=True)
        np.testing.assert_allclose(fit.coefficients,[2],atol=1e-14)
        self.assertAlmostEqual(fit.intercept,1,places=13)
        np.testing.assert_allclose(fit.prediction,y,atol=1e-14)

    def test_default_refit_has_no_stress_intercept(self):
        fit=fit_nonnegative(np.array([[0.],[1.],[2.]]),np.array([1.,3.,5.]))
        self.assertEqual(fit.intercept,0)
        self.assertEqual(fit.prediction[0],0)

    def test_fallback_converges_and_records_solver(self):
        with patch('mdisc.regression.nnls',side_effect=RuntimeError('iteration limit')):
            fit=fit_nonnegative(np.eye(2),np.array([2.,4.]))
        np.testing.assert_allclose(fit.coefficients,[2,4],atol=1e-12)
        self.assertEqual(fit.solver,'lsq_linear')

    def test_invalid_weights_never_silently_become_uniform(self):
        for weights in ([1], [1,-2],[1,np.nan],[0,1]):
            with self.assertRaises(ValueError): fit_nonnegative(np.eye(2),np.ones(2),weights=weights)

    def test_mode_relative_loss_is_balanced_across_counts_and_magnitudes(self):
        y=np.array([1.,1.,2.,2.,2.,2.]); modes=np.array(['A','A','B','B','B','B'])
        w=WeightPolicy().fit(y,modes).transform(modes)
        self.assertAlmostEqual(w[0]/w[2],8)
        self.assertAlmostEqual(float(np.sum(w[:2]*y[:2]**2)),float(np.sum(w[2:]*y[2:]**2)))

    def test_scaling_preserves_origin_and_physical_coefficients(self):
        X=np.array([[0.,0.],[1.,3.],[2.,4.]])
        y=np.array([0.,3.,4.]);w=np.array([1.,4.,9.])
        sc=DesignScaling.fit(X,y,w); A,b=sc.transform(X,y,w)
        np.testing.assert_array_equal(A[0],[0,0])
        beta=np.array([.2,.4])
        np.testing.assert_allclose(A@beta,(X@sc.coefficients_to_physical(beta))/sc.target*np.sqrt(w),atol=1e-14)

    def test_gaussian_ic_counts_variance_and_weight_determinant(self):
        y=np.array([1.,2.,3.]); pred=np.array([.9,2.1,2.8]);w=np.array([1.,2.,3.])
        aic,bic=information_criteria(y,pred,2,weights=w)
        likelihood=3*(np.log(2*np.pi*np.sum(w*(y-pred)**2)/3)+1)-np.log(w).sum()
        self.assertAlmostEqual(aic,likelihood+2*3)
        self.assertAlmostEqual(bic,likelihood+np.log(3)*3)
        np.testing.assert_allclose(information_criteria(y,pred,2,weights=7*w),(aic,bic),atol=1e-12)

    def test_exact_fit_ic_remains_finite(self):
        self.assertTrue(np.isfinite(information_criteria(np.arange(3.),np.arange(3.),0)).all())

    def test_same_size_distinct_supports_are_retained(self):
        supports=distinct_supports(np.array([[1.,1.,0.],[0.,2.,3.],[0.,0.,4.]]))
        self.assertEqual(supports,[(0,),(0,1),(1,2)])

    def test_lar_uses_documented_algorithm_and_refit_positivity(self):
        import mdisc.paths as paths
        with patch('mdisc.paths.lars_path',wraps=paths.lars_path) as wrapped:
            discovery_path(np.eye(3),np.ones(3),'LARS',PathSettings(path_count=3))
        self.assertEqual(wrapped.call_args.kwargs['method'],'lar')
        self.assertFalse(wrapped.call_args.kwargs['positive'])

    def test_paired_measurements_never_cross_validation_boundary(self):
        original=fixture(8)
        obs=Observations(np.repeat(original.F,2,axis=0),np.repeat(original.y,2),((0,0),(1,1))*8,
                         np.tile(['ff','nn'],8),np.repeat(np.arange(8),2),'nominal','MPa')
        for train,validation in grouped_mode_folds(obs,4,42):
            self.assertFalse(set(obs.groups[train]) & set(obs.groups[validation]))
            self.assertEqual(set(obs.modes[train]),{'ff','nn'})

    def test_discovery_and_weights_are_rebuilt_on_each_training_fold(self):
        import mdisc.discovery as engine
        obs=fixture();library=isotropic_library(mr_order=1)
        settings=DiscoverySettings(folds=2,ridge=0,paths=PathSettings(alpha_count=4,path_count=3))
        with patch('mdisc.discovery.discovery_path',wraps=engine.discovery_path) as path:
            result=discover(library,obs,'OMP',settings)
        self.assertEqual([call.args[0].shape[0] for call in path.call_args_list],[8,4,4])
        for fold,diag in zip(result.folds,result.diagnostics['folds']):
            tr=np.array(fold['train']); w=WeightPolicy().fit(obs.y[tr],obs.modes[tr]).transform(obs.modes[tr])
            scale=DesignScaling.fit(library.design(obs)[tr],obs.y[tr],w)
            self.assertEqual(diag['target_scale'],scale.target)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError): PathSettings(alpha_count=1)
        with self.assertRaises(ValueError): DiscoverySettings(folds=1)
        with self.assertRaises(ValueError): PathSettings(solver_tolerance=float('nan'))
        with self.assertRaises(ValueError): PathSettings(alpha_count=2.5)


class DataAndExecution(unittest.TestCase):
    def test_synthetic_ground_truth_is_independent_of_display_labels(self):
        from dataclasses import replace
        from mdisc.mechanics import EnergyLibrary
        from Isotropic.Benchmarks.configuration import GROUND_TRUTHS
        from Isotropic.Benchmarks.data import generate_data
        library=isotropic_library(mr_order=1)
        renamed=EnergyLibrary([replace(term,label='arbitrary display label') for term in library.terms])
        args=dict(noise=0,points=8,stretch_range=(.8,1.8),seed=42)
        original=generate_data(library,GROUND_TRUTHS[1].coefficients_pa,**args)
        altered=generate_data(renamed,GROUND_TRUTHS[1].coefficients_pa,**args)
        np.testing.assert_array_equal(original.P11.to_numpy(),altered.P11.to_numpy())

    def test_observation_components_and_units_are_explicit(self):
        obs=fixture()
        for components,unit,y in ((((3,0),)*len(obs.y),'MPa',obs.y),
                                 (obs.components,'unknown',obs.y),
                                 (obs.components,'MPa',obs.y[:,None])):
            with self.assertRaises(ValueError):
                Observations(obs.F,y,components,obs.modes,obs.groups,obs.measure,unit)

    def test_public_package_exports_runner(self):
        from mdisc import run_case as public_runner
        self.assertIs(public_runner,run_case)

    def test_kawabata_reference_row_is_not_discarded(self):
        from Isotropic.Treloar.data import load_data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for mode in ('UT','PS','EBT'):
                (root/f'TreloarData{mode}.csv').write_text('Stretch[-];Stress[MPa]\n1;0\n1.2;0.3\n')
            (root/'Kawabata.csv').write_text('l_1[-];l_2[-];P_11[MPa];P_22[MPa]\n1;1;0;0\n1.2;1.1;0.4;0.2\n')
            frame=load_data(root)
            self.assertEqual(len(frame[frame.dataset=='Kawabata']),2)
            self.assertEqual(frame.iloc[-2]['P11'],0)

    def test_copied_ebt_component_is_not_an_independent_measurement(self):
        from mdisc.data import isotropic_observations
        import pandas as pd
        frame=pd.DataFrame([dict(lambda1=1.2,lambda2=1.2,lambda3=1/1.2**2,P11=.3,P22=.3,mode='EBT',dataset='Treloar')])
        self.assertEqual(len(isotropic_observations(frame).y),1)
        frame['dataset']='Kawabata'
        observations=isotropic_observations(frame)
        self.assertEqual(len(observations.y),2)
        self.assertEqual(observations.groups[0],observations.groups[1])

    def test_imports_do_not_execute_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            process=subprocess.run([sys.executable,'-c','import Anisotropic.workflow, Isotropic.Treloar.workflow, Isotropic.Benchmarks.workflow'],
                                   cwd=directory,env=dict(os.environ,PYTHONPATH=str(ROOT),MPLBACKEND='Agg'),capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertEqual(process.stdout,'')
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_failure_preserves_cwd_rng_and_warning_filters(self):
        cwd=Path.cwd(); rng=np.random.get_state(); filters=list(warnings.filters)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_case('cardiac',input_dir=directory,output_dir=Path(directory)/'output',plots=False)
        self.assertEqual(Path.cwd(),cwd)
        self.assertEqual(warnings.filters,filters)
        after=np.random.get_state()
        np.testing.assert_array_equal(rng[1],after[1]);self.assertEqual(rng[2:],after[2:])

    def test_central_scripts_work_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in ('synthetic','treloar','cardiac'):
                process=subprocess.run([sys.executable,str(ROOT/'scripts'/f'run_{case}.py'),'--help'],cwd=directory,capture_output=True,text=True)
                self.assertEqual(process.returncode,0,process.stderr)
                self.assertIn('--output-dir',process.stdout)


if __name__=='__main__':
    unittest.main()
