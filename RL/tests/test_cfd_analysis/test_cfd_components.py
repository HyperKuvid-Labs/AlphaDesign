import pytest
import numpy as np
import sys
import os
import tempfile
import trimesh
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfd_analysis import STLWingAnalyzer, WindTunnelRig, F1CFDPipeline

class TestCFDAnalyzer:
    
    @pytest.fixture
    def mock_stl_file(self):
        # Create a simple cube mesh for testing
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2]   # Right
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            mesh.export(f.name)
            return f.name
    
    def test_cfd_analyzer_initialization(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test basic initialization
            assert analyzer is not None
            assert analyzer.stl_file_path == mock_stl_file
            assert analyzer.mesh is not None
            assert analyzer.air_density == 1.225
            assert analyzer.kinematic_viscosity == 1.5e-5
            
            # Test F1 conditions are loaded
            assert 'track_temperature_range' in analyzer.f1_conditions
            assert len(analyzer.test_speeds) > 0
            assert len(analyzer.test_angles) > 0
            
            # Test wing geometry extraction
            assert hasattr(analyzer, 'wingspan')
            assert hasattr(analyzer, 'reference_area')
            assert analyzer.wingspan > 0
            assert analyzer.reference_area > 0
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_performance_metrics_calculation(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test quick performance analysis
            result = analyzer.quick_performance_analysis(
                test_speed_kmh=200,
                ground_clearance=75,
                wing_angle=0
            )
            
            assert result is not None
            assert 'total_downforce' in result
            assert 'total_drag' in result
            assert 'efficiency_ratio' in result
            assert 'flow_characteristics' in result
            assert 'stall_margin' in result
            
            # Test that values are reasonable
            assert result['total_downforce'] > 0
            assert result['total_drag'] > 0
            assert result['efficiency_ratio'] > 0
            assert result['stall_margin'] >= 0
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_mesh_generation(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test mesh quality check
            quality_ok = analyzer.check_mesh_quality()
            
            # Should return True (proceeding with analysis regardless)
            assert quality_ok == True
            
            # Test mesh properties
            assert len(analyzer.mesh.vertices) > 0
            assert len(analyzer.mesh.faces) > 0
            assert analyzer.mesh_bounds is not None
            assert analyzer.mesh_center is not None
            
            # Test face aspect ratio calculation
            aspect_ratio_score = analyzer.calculate_face_aspect_ratios()
            assert 0 <= aspect_ratio_score <= 1
            
            # Test edge length consistency
            edge_consistency = analyzer.calculate_edge_length_consistency()
            assert 0 <= edge_consistency <= 1
            
            # Test normal consistency check
            normal_consistency = analyzer.check_normal_consistency()
            assert 0 <= normal_consistency <= 1
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_multi_element_analysis(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test multi-element analysis
            result = analyzer.multi_element_analysis(
                speed_ms=55.56,  # 200 km/h
                ground_clearance_mm=75,
                wing_angle_deg=5,
                setup_angles=[0, 2, 4, 6],  # Different angles for each element
                environmental_conditions={
                    'temperature': 25,
                    'pressure': 1013,
                    'humidity': 60,
                    'crosswind': 5,
                    'yaw_angle': 2,
                    'banking_angle': 0
                }
            )
            
            assert result is not None
            assert 'elements' in result
            assert 'total_downforce' in result
            assert 'total_drag' in result
            assert 'total_sideforce' in result
            assert 'efficiency_ratio' in result
            assert 'f1_specific_metrics' in result
            assert 'flow_characteristics' in result
            
            # Test element-specific results
            assert len(result['elements']) > 0
            for element in result['elements']:
                assert 'element_number' in element
                assert 'downforce_N' in element
                assert 'drag_N' in element
                assert 'lift_coefficient' in element
                assert 'drag_coefficient' in element
                assert 'ground_effect_factor' in element
                
            # Test F1-specific metrics
            f1_metrics = result['f1_specific_metrics']
            assert 'stall_margin' in f1_metrics
            assert 'downforce_to_weight_ratio' in f1_metrics
            assert 'yaw_sensitivity' in f1_metrics
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_comprehensive_analysis(self, mock_stl_file):
        """Test comprehensive F1 CFD analysis."""
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Run comprehensive analysis (this might take some time)
            results = analyzer.run_comprehensive_f1_analysis()
            
            assert results is not None
            assert 'speed_sweep' in results
            assert 'ground_clearance_sweep' in results
            assert 'angle_sweep' in results
            assert 'environmental_sweep' in results
            assert 'optimal_settings' in results
            assert 'f1_performance_metrics' in results
            
            # Test speed sweep results
            speed_sweep = results['speed_sweep']
            assert len(speed_sweep) > 0
            for speed_data in speed_sweep:
                assert 'speed_kmh' in speed_data
                assert 'downforce_N' in speed_data
                assert 'efficiency_LD' in speed_data
                assert 'stall_margin_deg' in speed_data
                
            # Test optimal settings identification
            optimal = results['optimal_settings']
            assert 'max_efficiency_speed_kmh' in optimal
            assert 'optimal_ground_clearance_mm' in optimal
            assert 'optimal_wing_angle_deg' in optimal
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_convergence_criteria(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test stall margin calculation
            elements = [
                {'effective_angle_deg': 8, 'camber': 0.08, 'thickness_ratio': 0.12},
                {'effective_angle_deg': 12, 'camber': 0.10, 'thickness_ratio': 0.10},
                {'effective_angle_deg': 15, 'camber': 0.12, 'thickness_ratio': 0.08}
            ]
            
            stall_margin = analyzer.calculate_stall_margin(elements)
            assert stall_margin >= 0
            assert isinstance(stall_margin, (int, float))
            
            # Test performance consistency calculation
            consistency = analyzer.calculate_performance_consistency(elements)
            assert 0 <= consistency <= 1
            
            # Test flow attachment assessment
            attachment = analyzer.assess_enhanced_flow_attachment(elements)
            assert attachment in [
                "Excellent attachment", 
                "Good attachment", 
                "Marginal attachment", 
                "Poor attachment/Stall risk"
            ]
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)

class TestCFDValidation:
    
    def test_result_validation(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test enhanced lift coefficient calculation
            cl = analyzer.enhanced_airfoil_lift_coefficient(
                angle_of_attack=5, 
                element_idx=0, 
                mach_number=0.15
            )
            
            # Lift coefficient should be reasonable for F1 conditions
            assert -2.0 <= cl <= 3.0  # Reasonable range for F1 wings
            
            # Test enhanced drag coefficient calculation
            cd = analyzer.enhanced_airfoil_drag_coefficient(
                angle_of_attack=5,
                reynolds_number=1e6,
                element_idx=0,
                mach_number=0.15
            )
            
            # Drag coefficient should be positive and reasonable
            assert 0 < cd < 1.0  # Reasonable range for airfoils
            
            # Test ground effect calculation
            ground_effect = analyzer.calculate_ground_effect(
                ground_clearance_mm=50,
                element_idx=0
            )
            
            # Ground effect should enhance performance
            assert ground_effect >= 1.0
            assert ground_effect <= 2.5  # Capped maximum
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_f1_regulation_compliance(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test geometry summary for regulation compliance
            geometry = analyzer.get_enhanced_geometry_summary()
            
            assert 'wingspan_mm' in geometry
            assert 'reference_area_m2' in geometry
            assert 'num_elements' in geometry
            assert 'chord_lengths_mm' in geometry
            
            # Basic F1 regulation checks
            wingspan_mm = geometry['wingspan_mm']
            assert 1400 <= wingspan_mm <= 1800  # F1 regulation limits
            
            # Element count should be reasonable for F1
            assert 1 <= geometry['num_elements'] <= 6
            
            # Chord lengths should be reasonable
            chord_lengths = geometry['chord_lengths_mm']
            assert all(100 <= chord <= 500 for chord in chord_lengths)
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_environmental_impact_assessment(self, mock_stl_file):
        try:
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test different environmental conditions
            conditions_list = [
                {'temperature': 15, 'pressure': 1000, 'crosswind': 0},
                {'temperature': 35, 'pressure': 1020, 'crosswind': 10},
                {'temperature': 25, 'pressure': 990, 'crosswind': 5}
            ]
            
            results_list = [
                {'total_downforce': 1500, 'total_sideforce': 50},
                {'total_downforce': 1400, 'total_sideforce': 200},
                {'total_downforce': 1450, 'total_sideforce': 100}
            ]
            
            for conditions, results in zip(conditions_list, results_list):
                impact = analyzer.assess_environmental_impact(conditions, results)
                
                assert impact in [
                    "Minimal environmental impact",
                    "Moderate environmental impact", 
                    "Significant environmental impact"
                ]
                
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)

class TestWindTunnelIntegration:
    
    def test_wind_tunnel_rig_initialization(self):
        rig = WindTunnelRig(model_scale=0.6, max_speed_ms=80)
        
        assert rig.model_scale == 0.6
        assert rig.max_speed_ms == 80
        assert rig.boundary_layer_suction == True
        assert rig.test_section_width_m > 0
        assert rig.test_section_height_m > 0
        
        # Test correction factors are reasonable
        assert 0.8 <= rig.downforce_correlation_factor <= 1.2
        assert 0.8 <= rig.drag_correlation_factor <= 1.2
    
    def test_tunnel_corrections(self, mock_stl_file):
        """Test wind tunnel correction applications."""
        try:
            rig = WindTunnelRig(model_scale=0.6)
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Test blockage ratio check
            blockage_ok = rig.check_blockage_ratio(analyzer)
            assert isinstance(blockage_ok, bool)
            
            # Test tunnel corrections
            raw_forces = {
                'downforce_N': 100,
                'drag_N': 10,
                'efficiency_LD': 10
            }
            
            model_geometry = {
                'reference_area_m2': 0.5
            }
            
            corrected_forces = rig.apply_tunnel_corrections(
                raw_forces, 50, model_geometry  # 50 m/s tunnel speed
            )
            
            assert 'downforce_N' in corrected_forces
            assert 'drag_N' in corrected_forces
            assert 'efficiency_LD' in corrected_forces
            
            # Corrections should modify the forces
            assert corrected_forces['downforce_N'] != raw_forces['downforce_N']
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_reynolds_correction(self):
        rig = WindTunnelRig()
        
        forces = {'downforce_N': 1000, 'drag_N': 100, 'efficiency_LD': 10}
        tunnel_re = 1e6
        full_scale_re = 5e6
        
        corrected = rig.apply_reynolds_correction(forces, tunnel_re, full_scale_re)
        
        assert 'downforce_N' in corrected
        assert 'drag_N' in corrected
        assert 'efficiency_LD' in corrected
        
        # Reynolds corrections should modify forces
        assert corrected['downforce_N'] != forces['downforce_N']
        assert corrected['drag_N'] != forces['drag_N']
    
    def test_virtual_tunnel_run(self, mock_stl_file):
        """Test virtual wind tunnel run simulation."""
        try:
            rig = WindTunnelRig(model_scale=0.6)
            analyzer = STLWingAnalyzer(mock_stl_file)
            
            # Run virtual tunnel test
            result = rig.virtual_run(
                analyzer, 
                tunnel_speed_ms=45,  # 162 km/h tunnel speed
                angle_deg=5,
                ride_height_mm=75
            )
            
            assert result is not None
            assert 'downforce_N' in result
            assert 'drag_N' in result
            assert 'L/D' in result
            assert 'tunnel_reynolds' in result
            assert 'full_scale_reynolds' in result
            assert 'blockage_acceptable' in result
            assert 'corrections_applied' in result
            
            # Values should be positive and reasonable
            assert result['downforce_N'] > 0
            assert result['drag_N'] > 0
            assert result['L/D'] > 0
            assert result['tunnel_reynolds'] > 0
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)

class TestF1CFDPipeline:
    
    def test_pipeline_initialization(self, mock_stl_file):
        try:
            pipeline = F1CFDPipeline(mock_stl_file, tunnel_scale=0.6)
            
            assert pipeline.analyzer is not None
            assert pipeline.rig is not None
            assert pipeline.cfd_solver_config is not None
            assert pipeline.correlation_database is not None
            
            # Test configuration
            assert pipeline.cfd_solver_config['turbulence_model'] == 'k-omega-SST'
            assert pipeline.cfd_solver_config['max_iterations'] > 0
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_mesh_generation_specs(self, mock_stl_file):
        try:
            pipeline = F1CFDPipeline(mock_stl_file)
            
            mesh_specs = pipeline.generate_volume_mesh(
                target_y_plus=1.0,
                refinement_level='medium'
            )
            
            assert mesh_specs is not None
            assert 'surface_elements' in mesh_specs
            assert 'volume_elements' in mesh_specs
            assert 'first_cell_height_mm' in mesh_specs
            assert 'boundary_layers' in mesh_specs
            
            # Values should be reasonable
            assert mesh_specs['surface_elements'] > 0
            assert mesh_specs['volume_elements'] > 0
            assert mesh_specs['first_cell_height_mm'] > 0
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_boundary_conditions_setup(self, mock_stl_file):
        try:
            pipeline = F1CFDPipeline(mock_stl_file)
            
            bc = pipeline.setup_cfd_boundary_conditions(
                speed_ms=55.56,  # 200 km/h
                ground_clearance_mm=75,
                turbulence_intensity=0.05
            )
            
            assert bc is not None
            assert 'inlet' in bc
            assert 'outlet' in bc
            assert 'ground' in bc
            assert 'wing_surfaces' in bc
            assert 'far_field' in bc
            
            # Inlet conditions
            inlet = bc['inlet']
            assert inlet['velocity_ms'] == 55.56
            assert inlet['turbulent_kinetic_energy'] > 0
            assert inlet['specific_dissipation_rate'] > 0
            
            # Ground conditions
            ground = bc['ground']
            assert ground['velocity_ms'] == 55.56  # Moving ground
            assert ground['distance_to_wing_mm'] == 75
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)
    
    def test_cfd_tunnel_correlation(self, mock_stl_file):
        try:
            pipeline = F1CFDPipeline(mock_stl_file, tunnel_scale=0.6)
            
            correlation = pipeline.correlate_with_tunnel(
                speed_kmh=200,
                angle_deg=0,
                ride_height_mm=75
            )
            
            assert correlation is not None
            assert 'test_conditions' in correlation
            assert 'CFD_full_scale' in correlation
            assert 'tunnel_scaled' in correlation
            assert 'track_prediction' in correlation
            assert 'correlation_metrics' in correlation
            
            # Test conditions match
            conditions = correlation['test_conditions']
            assert conditions['speed_kmh'] == 200
            assert conditions['angle_deg'] == 0
            assert conditions['ride_height_mm'] == 75
            
            # All predictions should have force values
            cfd_data = correlation['CFD_full_scale']
            tunnel_data = correlation['tunnel_scaled']
            track_data = correlation['track_prediction']
            
            assert cfd_data['downforce_N'] > 0
            assert tunnel_data['corrected_downforce_N'] > 0
            assert track_data['downforce_N'] > 0
            
            # Correlation metrics should be calculated
            metrics = correlation['correlation_metrics']
            assert 'cfd_tunnel_downforce_diff_percent' in metrics
            assert 'cfd_tunnel_drag_diff_percent' in metrics
            assert 'confidence_level' in metrics
            
        finally:
            if os.path.exists(mock_stl_file):
                os.unlink(mock_stl_file)

if __name__ == "__main__":
    pytest.main([__file__])
