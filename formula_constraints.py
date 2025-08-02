import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class F1FrontWingParams:
    # Main Wing Structure
    total_span: float
    root_chord: float
    tip_chord: float
    chord_taper_ratio: float
    sweep_angle: float
    dihedral_angle: float
    twist_distribution_range: List[float]
    
    # Airfoil Profile
    base_profile: str
    max_thickness_ratio: float
    camber_ratio: float
    camber_position: float
    leading_edge_radius: float
    trailing_edge_thickness: float
    upper_surface_radius: float
    lower_surface_radius: float
    
    # Flap System
    flap_count: int
    flap_spans: List[float]
    flap_root_chords: List[float]
    flap_tip_chords: List[float]
    flap_cambers: List[float]
    flap_slot_gaps: List[float]
    flap_vertical_offsets: List[float]
    flap_horizontal_offsets: List[float]
    
    # Endplate System
    endplate_height: float
    endplate_max_width: float
    endplate_min_width: float
    endplate_thickness_base: float
    endplate_forward_lean: float
    endplate_rearward_sweep: float
    endplate_outboard_wrap: float
    
    # Footplate Features
    footplate_extension: float
    footplate_height: float
    arch_radius: float
    footplate_thickness: float
    primary_strake_count: int
    strake_heights: List[float]
    
    # Y250 Vortex Region
    y250_width: float
    y250_step_height: float
    y250_transition_length: float
    central_slot_width: float
    
    # Mounting System
    pylon_count: int
    pylon_spacing: float
    pylon_major_axis: float
    pylon_minor_axis: float
    pylon_length: float
    
    # Cascade Elements
    cascade_enabled: bool
    primary_cascade_span: float
    primary_cascade_chord: float
    secondary_cascade_span: float
    secondary_cascade_chord: float
    
    # Manufacturing Parameters
    wall_thickness_structural: float
    wall_thickness_aerodynamic: float
    wall_thickness_details: float
    minimum_radius: float
    mesh_resolution_aero: float
    mesh_resolution_structural: float
    
    # Construction Parameters
    resolution_span: int
    resolution_chord: int
    mesh_density: float
    surface_smoothing: bool
    
    # Material Properties
    material: str
    density: float
    weight_estimate: float
    
    # Performance Targets
    target_downforce: float
    target_drag: float
    efficiency_factor: float

class F1FrontWingAnalyzer:
    
    def __init__(self, params: F1FrontWingParams):
        self.params = params
        self.validation_results = {}
        self.computed_values = {}
        self.compliance_status = {}
        
        # Physical constants
        self.air_density = 1.225  # kg/m³ at sea level
        self.kinematic_viscosity = 1.5e-5  # m²/s
        self.velocity_ref = 50.0  # m/s reference velocity
        self.material_density = params.density  # kg/m³
        self.elastic_modulus = 230e9  # Pa for carbon fiber
        self.material_ultimate_strength = 1500e6  # Pa
        self.safety_factor_required = 2.5
        
    def compute_derived_geometry(self) -> Dict[str, float]:
        p = self.params
        
        # Basic geometric derivations
        wing_planform_area = 0.5 * (p.root_chord + p.tip_chord) * p.total_span / 1000000  # m²
        effective_aspect_ratio = (p.total_span/1000)**2 / wing_planform_area
        average_thickness = (p.root_chord + p.tip_chord) * 0.5 * p.max_thickness_ratio / 1000  # m
        wing_volume = wing_planform_area * average_thickness
        
        # Sweep and taper effects
        quarter_chord_sweep = math.atan(math.tan(math.radians(p.sweep_angle)) - 
                                      (p.root_chord - p.tip_chord)/(2 * p.total_span))
        effective_span = p.total_span * math.cos(math.radians(p.dihedral_angle)) / 1000  # m
        mean_aerodynamic_chord = (2/3) * p.root_chord * (1 + p.chord_taper_ratio + 
                                p.chord_taper_ratio**2)/(1 + p.chord_taper_ratio) / 1000  # m
        
        derived = {
            'wing_planform_area': wing_planform_area,
            'effective_aspect_ratio': effective_aspect_ratio,
            'average_thickness': average_thickness,
            'wing_volume': wing_volume,
            'quarter_chord_sweep': math.degrees(quarter_chord_sweep),
            'effective_span': effective_span,
            'mean_aerodynamic_chord': mean_aerodynamic_chord
        }
        
        self.computed_values.update(derived)
        return derived
    
    def compute_flap_system_parameters(self) -> Dict[str, Any]:
        p = self.params
        
        # Total flap area
        total_flap_area = sum([0.5 * (p.flap_root_chords[i] + p.flap_tip_chords[i]) * 
                              p.flap_spans[i] for i in range(p.flap_count)]) / 1000000  # m²
        
        wing_planform_area = self.computed_values['wing_planform_area']
        flap_area_ratio = total_flap_area / wing_planform_area
        
        # Flap parameters
        flap_overlap = []
        flap_chord_ratio = []
        slot_convergence = []
        optimal_gap = []
        flap_slot_effect = []
        attachment_factor = []
        
        for i in range(p.flap_count):
            # Overlap calculation
            if i > 0:
                overlap = (p.flap_spans[i-1] - p.flap_spans[i]) / p.flap_spans[i]
            else:
                overlap = 0
            flap_overlap.append(overlap)
            
            # Chord ratio
            chord_ratio = p.flap_root_chords[i] / p.root_chord
            flap_chord_ratio.append(chord_ratio)
            
            # Slot convergence angle
            if i > 0:
                convergence = math.atan((p.flap_vertical_offsets[i] - p.flap_vertical_offsets[i-1]) / 
                                      (p.flap_horizontal_offsets[i] - p.flap_horizontal_offsets[i-1]))
            else:
                convergence = 0
            slot_convergence.append(math.degrees(convergence))
            
            # Boundary layer thickness for optimal gap
            bl_thickness = 0.37 * math.sqrt(p.root_chord/1000 * self.kinematic_viscosity / self.velocity_ref) * 1000  # mm
            opt_gap = 2.5 * bl_thickness * (1 + 0.5 * i)
            optimal_gap.append(opt_gap)
            
            # Slot effect and attachment
            slot_effect = 1.3 * (p.flap_slot_gaps[i] / p.flap_root_chords[i])**0.2
            flap_slot_effect.append(slot_effect)
            
            attach_factor = math.exp(-((p.flap_slot_gaps[i] - opt_gap) / opt_gap)**2)
            attachment_factor.append(attach_factor)
        
        flap_params = {
            'total_flap_area': total_flap_area,
            'flap_area_ratio': flap_area_ratio,
            'flap_overlap': flap_overlap,
            'flap_chord_ratio': flap_chord_ratio,
            'slot_convergence': slot_convergence,
            'optimal_gap': optimal_gap,
            'flap_slot_effect': flap_slot_effect,
            'attachment_factor': attachment_factor
        }
        
        self.computed_values.update(flap_params)
        return flap_params
    
    def compute_endplate_parameters(self) -> Dict[str, float]:
        p = self.params
        
        endplate_area = p.endplate_height * (p.endplate_max_width + p.endplate_min_width) * 0.5 / 1000000  # m²
        endplate_aspect_ratio = p.endplate_height / p.endplate_max_width
        endplate_taper_ratio = p.endplate_min_width / p.endplate_max_width
        
        # Vortex parameters
        endplate_vortex_core_radius = 0.05 * p.endplate_height / 1000  # m
        vortex_strength_coefficient = (p.endplate_outboard_wrap * 
                                     (p.endplate_height/p.total_span) * self.velocity_ref)
        
        # Footplate interaction
        footplate_area = p.footplate_extension * p.footplate_height / 1000000  # m²
        footplate_blockage_ratio = footplate_area / (p.endplate_height * p.endplate_min_width / 1000000)
        arch_curvature = 1 / p.arch_radius if p.arch_radius > 0 else 0
        
        endplate_params = {
            'endplate_area': endplate_area,
            'endplate_aspect_ratio': endplate_aspect_ratio,
            'endplate_taper_ratio': endplate_taper_ratio,
            'vortex_core_radius': endplate_vortex_core_radius,
            'vortex_strength_coefficient': vortex_strength_coefficient,
            'footplate_area': footplate_area,
            'footplate_blockage_ratio': footplate_blockage_ratio,
            'arch_curvature': arch_curvature
        }
        
        self.computed_values.update(endplate_params)
        return endplate_params
    
    def compute_y250_parameters(self) -> Dict[str, float]:
        p = self.params
        
        # Y250 compliance check
        y250_compliance_factor = p.y250_width / 500.0  # Should equal 1.0
        
        # Vortex formation parameters
        y250_velocity_ratio = math.sqrt(1 + (p.y250_step_height / p.y250_width)**2)
        y250_pressure_jump = 0.5 * self.air_density * self.velocity_ref**2 * y250_velocity_ratio**2
        y250_vorticity = (p.y250_step_height/1000 * self.velocity_ref) / (p.y250_transition_length/1000 * p.y250_width/1000)
        
        # Central slot effects
        slot_mass_flow = (self.air_density * self.velocity_ref * 
                         p.central_slot_width/1000 * p.y250_step_height/1000)
        slot_momentum_deficit = slot_mass_flow * self.velocity_ref * 0.3
        
        y250_params = {
            'y250_compliance_factor': y250_compliance_factor,
            'y250_velocity_ratio': y250_velocity_ratio,
            'y250_pressure_jump': y250_pressure_jump,
            'y250_vorticity': y250_vorticity,
            'slot_mass_flow': slot_mass_flow,
            'slot_momentum_deficit': slot_momentum_deficit
        }
        
        self.computed_values.update(y250_params)
        return y250_params
    
    def compute_aerodynamic_performance(self) -> Dict[str, float]:
        p = self.params
        
        # Reynolds numbers
        reynolds_main = (self.velocity_ref * p.root_chord/1000) / self.kinematic_viscosity
        reynolds_flaps = [(self.velocity_ref * chord/1000) / self.kinematic_viscosity 
                         for chord in p.flap_root_chords]
        
        # Section lift coefficients
        cl_main = 2 * math.pi * (p.camber_ratio + p.twist_distribution_range[0] * math.pi/180)
        cl_flaps = [2 * math.pi * p.flap_cambers[i] * 
                   (1 + self.computed_values['flap_slot_effect'][i]) 
                   for i in range(p.flap_count)]
        
        # Force calculations
        wing_area = self.computed_values['wing_planform_area']
        downforce_main = 0.5 * self.air_density * self.velocity_ref**2 * wing_area * cl_main
        
        downforce_flaps = []
        for i in range(p.flap_count):
            flap_area = 0.5 * (p.flap_root_chords[i] + p.flap_tip_chords[i]) * p.flap_spans[i] / 1000000
            downforce_flap = 0.5 * self.air_density * self.velocity_ref**2 * flap_area * cl_flaps[i]
            downforce_flaps.append(downforce_flap)
        
        total_downforce = downforce_main + sum(downforce_flaps)
        
        # Drag calculations
        effective_span = self.computed_values['effective_span']
        induced_drag_main = (downforce_main**2) / (math.pi * effective_span**2 * 0.5 * self.air_density * self.velocity_ref**2)
        
        # Profile drag
        cd_profile_main = 0.008 + 0.05 * p.max_thickness_ratio**2 + 0.02 * (p.camber_ratio**1.5)
        profile_drag_main = 0.5 * self.air_density * self.velocity_ref**2 * wing_area * cd_profile_main
        
        # Flap drag
        flap_drags = []
        for i in range(p.flap_count):
            flap_area = 0.5 * (p.flap_root_chords[i] + p.flap_tip_chords[i]) * p.flap_spans[i] / 1000000
            cd_flap = 0.012 + 0.08 * (p.flap_cambers[i]**1.8) + 0.003 * (p.flap_slot_gaps[i] / p.flap_root_chords[i])**0.5
            flap_drag = 0.5 * self.air_density * self.velocity_ref**2 * flap_area * cd_flap
            flap_drags.append(flap_drag)
        
        # Additional drag components
        endplate_area = self.computed_values['endplate_area']
        endplate_drag = 0.5 * self.air_density * self.velocity_ref**2 * endplate_area * (
            0.15 + 0.02 * (p.endplate_forward_lean/10)**2 + 0.03 * (p.endplate_rearward_sweep/15)**2
        )
        
        y250_drag = 0.5 * self.air_density * self.velocity_ref**2 * (p.y250_width/1000 * p.y250_step_height/1000) * 0.8
        
        cascade_drag = 0
        if p.cascade_enabled:
            cascade_drag = 0.5 * self.air_density * self.velocity_ref**2 * (
                p.primary_cascade_span/1000 * p.primary_cascade_chord/1000 * 0.2 +
                p.secondary_cascade_span/1000 * p.secondary_cascade_chord/1000 * 0.25
            )
        
        total_drag = induced_drag_main + profile_drag_main + sum(flap_drags) + endplate_drag + y250_drag + cascade_drag
        
        # Efficiency
        efficiency_computed = total_downforce / total_drag if total_drag > 0 else 0
        
        aero_performance = {
            'reynolds_main': reynolds_main,
            'reynolds_flaps': reynolds_flaps,
            'cl_main': cl_main,
            'cl_flaps': cl_flaps,
            'downforce_main': downforce_main,
            'downforce_flaps': downforce_flaps,
            'total_downforce': total_downforce,
            'induced_drag_main': induced_drag_main,
            'profile_drag_main': profile_drag_main,
            'flap_drags': flap_drags,
            'endplate_drag': endplate_drag,
            'y250_drag': y250_drag,
            'cascade_drag': cascade_drag,
            'total_drag': total_drag,
            'efficiency_computed': efficiency_computed
        }
        
        self.computed_values.update(aero_performance)
        return aero_performance
    
    def compute_structural_analysis(self) -> Dict[str, float]:
        p = self.params
        
        # Mass calculations
        wing_area = self.computed_values['wing_planform_area']
        main_element_mass = self.material_density * wing_area * p.wall_thickness_structural/1000
        
        flap_masses = []
        for i in range(p.flap_count):
            flap_area = 0.5 * (p.flap_root_chords[i] + p.flap_tip_chords[i]) * p.flap_spans[i] / 1000000
            flap_mass = self.material_density * flap_area * p.wall_thickness_aerodynamic/1000
            flap_masses.append(flap_mass)
        
        endplate_area = self.computed_values['endplate_area']
        endplate_mass = self.material_density * endplate_area * p.endplate_thickness_base/1000
        
        total_computed_mass = main_element_mass + sum(flap_masses) + 2 * endplate_mass
        
        # Structural loading (simplified)
        total_downforce = self.computed_values['total_downforce']
        max_bending_moment = total_downforce * p.total_span/1000 * 0.25
        max_shear_force = total_downforce * 0.6
        
        # Approximate section properties
        section_height = p.root_chord/1000 * p.max_thickness_ratio
        section_width = p.wall_thickness_structural/1000
        moment_of_inertia = section_width * section_height**3 / 12
        section_modulus = moment_of_inertia / (section_height/2)
        
        # Stress and safety factor
        stress_max = max_bending_moment / section_modulus if section_modulus > 0 else 0
        safety_factor = self.material_ultimate_strength / stress_max if stress_max > 0 else float('inf')
        
        structural_analysis = {
            'main_element_mass': main_element_mass,
            'flap_masses': flap_masses,
            'endplate_mass': endplate_mass,
            'total_computed_mass': total_computed_mass,
            'max_bending_moment': max_bending_moment,
            'max_shear_force': max_shear_force,
            'stress_max': stress_max,
            'safety_factor': safety_factor
        }
        
        self.computed_values.update(structural_analysis)
        return structural_analysis
    
    def validate_constraints(self) -> Dict[str, bool]:
        p = self.params
        computed = self.computed_values
        
        validations = {}
        
        # Geometric constraints
        validations['chord_taper_valid'] = 0.5 <= p.chord_taper_ratio <= 1.0
        validations['sweep_angle_valid'] = 0 <= p.sweep_angle <= 10
        validations['dihedral_valid'] = 0 <= p.dihedral_angle <= 5
        validations['aspect_ratio_valid'] = 2.0 <= computed['effective_aspect_ratio'] <= 6.0
        
        # Y250 compliance
        validations['y250_compliance'] = abs(computed['y250_compliance_factor'] - 1.0) < 0.001
        
        # Flap system constraints
        validations['flap_gap_optimal'] = all([
            abs(p.flap_slot_gaps[i] - computed['optimal_gap'][i]) / computed['optimal_gap'][i] < 0.3
            for i in range(p.flap_count)
        ])
        
        validations['flap_attachment'] = all([
            factor > 0.7 for factor in computed['attachment_factor']
        ])
        
        # Manufacturing constraints
        validations['wall_thickness_feasible'] = (
            p.wall_thickness_structural >= 2.0 and
            p.wall_thickness_aerodynamic >= 1.5 and
            p.wall_thickness_details >= 1.0
        )
        
        validations['minimum_radius_valid'] = p.minimum_radius >= 0.2
        
        # Structural safety
        validations['safety_factor_adequate'] = computed['safety_factor'] >= self.safety_factor_required
        
        # Performance targets
        validations['downforce_target_met'] = (
            abs(computed['total_downforce'] - p.target_downforce) / p.target_downforce < 0.1
        )
        
        validations['drag_target_met'] = (
            abs(computed['total_drag'] - p.target_drag) / p.target_drag < 0.15
        )
        
        validations['efficiency_target_met'] = (
            abs(computed['efficiency_computed'] - p.efficiency_factor) / p.efficiency_factor < 0.1
        )
        
        # Material and weight constraints
        validations['weight_estimate_accurate'] = (
            abs(computed['total_computed_mass'] - p.weight_estimate) / p.weight_estimate < 0.2
        )
        
        self.compliance_status = validations
        return validations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        print("F1 Front Wing Analysis Starting...")
        print("=" * 50)
        
        # Run all computations
        geometry = self.compute_derived_geometry()
        flap_system = self.compute_flap_system_parameters()
        endplate = self.compute_endplate_parameters()
        y250 = self.compute_y250_parameters()
        aero = self.compute_aerodynamic_performance()
        structural = self.compute_structural_analysis()
        validations = self.validate_constraints()
        
        # Summary results
        total_validations = len(validations)
        passed_validations = sum(validations.values())
        overall_compliance = passed_validations / total_validations
        
        results = {
            'computed_values': self.computed_values,
            'validation_results': validations,
            'overall_compliance': overall_compliance,
            'compliance_percentage': overall_compliance * 100,
            'passed_validations': passed_validations,
            'total_validations': total_validations
        }
        
        self.print_results_summary(results)
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print formatted results summary"""
        print(f"\nANALYSIS RESULTS SUMMARY")
        print("=" * 50)
        print(f"Overall Compliance: {results['compliance_percentage']:.1f}%")
        print(f"Passed Validations: {results['passed_validations']}/{results['total_validations']}")
        
        print(f"\nKEY COMPUTED VALUES:")
        print("-" * 30)
        computed = results['computed_values']
        print(f"Total Downforce: {computed['total_downforce']:.1f} N")
        print(f"Total Drag: {computed['total_drag']:.1f} N")
        print(f"Efficiency (L/D): {computed['efficiency_computed']:.2f}")
        print(f"Computed Mass: {computed['total_computed_mass']:.2f} kg")
        print(f"Safety Factor: {computed['safety_factor']:.2f}")
        
        print(f"\nVALIDATION STATUS:")
        print("-" * 30)
        validations = results['validation_results']
        for constraint, status in validations.items():
            status_symbol = "✓" if status else "✗"
            print(f"{status_symbol} {constraint}: {status}")
        
        if results['overall_compliance'] >= 0.8:
            print(f"\n✓ DESIGN COMPLIANT - Ready for further optimization")
        elif results['overall_compliance'] >= 0.6:
            print(f"\n⚠ DESIGN PARTIALLY COMPLIANT - Address failed constraints")
        else:
            print(f"\n✗ DESIGN NON-COMPLIANT - Major revisions required")

def main():
    """Main function to run the F1 front wing analysis"""
    
    # Sample parameters (replace with your actual input parameters)
    sample_params = F1FrontWingParams(
        # Main Wing Structure
        total_span=1600,
        root_chord=280,
        tip_chord=250,
        chord_taper_ratio=0.89,
        sweep_angle=3.5,
        dihedral_angle=2.5,
        twist_distribution_range=[-1.5, 0.5],
        
        # Airfoil Profile
        base_profile="NACA_64A010_modified",
        max_thickness_ratio=0.15,
        camber_ratio=0.08,
        camber_position=0.40,
        leading_edge_radius=2.8,
        trailing_edge_thickness=2.5,
        upper_surface_radius=800,
        lower_surface_radius=1100,
        
        # Flap System
        flap_count=3,
        flap_spans=[1600, 1500, 1400],
        flap_root_chords=[220, 180, 140],
        flap_tip_chords=[200, 160, 120],
        flap_cambers=[0.12, 0.10, 0.08],
        flap_slot_gaps=[14, 12, 10],
        flap_vertical_offsets=[25, 45, 70],
        flap_horizontal_offsets=[30, 60, 85],
        
        # Endplate System
        endplate_height=280,
        endplate_max_width=120,
        endplate_min_width=40,
        endplate_thickness_base=10,
        endplate_forward_lean=6,
        endplate_rearward_sweep=10,
        endplate_outboard_wrap=18,
        
        # Footplate
        footplate_extension=70,
        footplate_height=30,
        arch_radius=130,
        footplate_thickness=5,
        primary_strake_count=2,
        strake_heights=[45, 35],
        
        # Y250 Region
        y250_width=500,
        y250_step_height=18,
        y250_transition_length=80,
        central_slot_width=30,
        
        # Mounting System
        pylon_count=2,
        pylon_spacing=320,
        pylon_major_axis=38,
        pylon_minor_axis=25,
        pylon_length=120,
        
        # Cascade Elements
        cascade_enabled=True,
        primary_cascade_span=250,
        primary_cascade_chord=55,
        secondary_cascade_span=160,
        secondary_cascade_chord=40,
        
        # Manufacturing
        wall_thickness_structural=4,
        wall_thickness_aerodynamic=2.5,
        wall_thickness_details=2.0,
        minimum_radius=0.4,
        mesh_resolution_aero=0.4,
        mesh_resolution_structural=0.6,
        
        # Construction
        resolution_span=40,
        resolution_chord=25,
        mesh_density=1.5,
        surface_smoothing=True,
        
        # Material
        material="Standard Carbon Fiber",
        density=1600,
        weight_estimate=4.0,
        
        # Performance Targets
        target_downforce=1000,
        target_drag=180,
        efficiency_factor=0.75
    )
    
    # Create analyzer and run analysis
    analyzer = F1FrontWingAnalyzer(sample_params)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
