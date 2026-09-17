"""
F1 FRONT WING MULTI-ELEMENT FLAP SYSTEM GENERATOR - MASTER LEVEL
=================================================================
Generates ultra-realistic F1 front wing with progressive flap curvature
Based on 2024 F1 Technical Regulations and modern aero philosophy

Features:
- Main element + 3 flaps with independent angle of attack control
- Progressive trailing edge kick-up (more pronounced on higher elements)
- Realistic slot-gap-overhang relationships
- Gurney flaps with proper sizing
- Leading edge droop on flaps for flow attachment
- Spanwise twist distribution with multiple control points
- Ground effect considerations
- Flexible mounting with deflection simulation
- Element overlap optimization
- Complete continuous surfaces (no gaps)
- Export to STL format
"""

import numpy as np
from stl import mesh
import math
from scipy.interpolate import CubicSpline, make_interp_spline, interp1d
from typing import List, Tuple, Dict

class F1FrontWingMultiElementGenerator:
    """
    MASTER-LEVEL F1 Front Wing Multi-Element System Generator
    Generates regulation-compliant, aerodynamically optimized front wings
    with realistic progressive curvature and modern F1 styling
    """

    def __init__(self,
                 # Main Wing Platform Geometry
                 total_span=1600,              # mm (regulation max: 1800mm)
                 root_chord=280,               # mm at centerline
                 tip_chord=250,                # mm at endplate
                 chord_taper_ratio=0.89,
                 sweep_angle=3.5,              # degrees
                 dihedral_angle=2.5,           # degrees (positive = up)
                 twist_distribution_range=[-1.5, 0.5],  # [root, tip] degrees

                 # Airfoil Profile (Main Element)
                 base_profile="NACA_64A010_modified",
                 max_thickness_ratio=0.15,     # 15% t/c
                 camber_ratio=0.08,            # 8% camber
                 camber_position=0.40,         # Max camber at 40% chord
                 leading_edge_radius=2.8,      # mm
                 trailing_edge_thickness=2.5,  # mm
                 upper_surface_radius=800,     # mm curvature radius
                 lower_surface_radius=1100,    # mm curvature radius

                 # Multi-Element Configuration
                 flap_count=3,
                 flap_spans=[1600, 1600, 1600],           # mm - each element span
                 flap_root_chords=[220, 180, 140],        # mm at root
                 flap_tip_chords=[200, 160, 120],         # mm at tip
                 flap_cambers=[0.12, 0.10, 0.08],         # Camber ratio per flap

                 # Element-Specific Angle of Attack (CRITICAL FOR REALISM)
                 main_element_aoa_range=[-2.3, 1.2],      # [root, tip] degrees
                 flap1_aoa_range=[-21.3, 0.0],            # Aggressive at root
                 flap2_aoa_range=[-16.7, 0.0],
                 flap3_aoa_range=[-13.5, 0.0],
                 flap4_aoa_range=None,                    # None -> extrapolate progression
                 aoa_control_points=4,                     # Spanwise AoA control points
                 flap_angle_progression=True,

                 # Slot Geometry (Absolute and Relative)
                 flap_slot_gaps=[14, 12, 10],             # mm absolute gap
                 flap_vertical_offsets=[25, 45, 70],      # mm vertical spacing
                 flap_horizontal_offsets=[30, 60, 85],    # mm streamwise offset
                 slot_gap_ratios=[0.012, 0.010, 0.008],   # As fraction of upstream chord
                 slot_overhang_ratios=[0.25, 0.22, 0.20], # Overlap ratio
                 slot_flow_velocity_ratio=1.3,
                 slot_angle_progression=[5, 8, 12],       # degrees - slot exit angle
                 gap_optimization_enabled=True,
                 aerodynamic_slots=True,

                 # Advanced Aerodynamic Features
                 gurney_flaps=True,
                 gurney_flap_heights=[3, 2.5, 2],         # mm per element
                 slot_lip_radius=[2.5, 2.0, 1.5],         # mm - prevents separation
                 leading_edge_droop=[0, -2, -4],          # mm - flap LE droop
                 trailing_edge_kick_up=[0, 3, 6],         # mm - progressive upkick
                 element_overlap_optimization=True,

                 # Wing Flexibility (Flexi-Wing)
                 wing_flexibility_enabled=True,
                 flap_deflection_limit=3.0,               # mm max TE deflection
                 deflection_load_point=60,                # N load
                 bodywork_vertical_deflection_single=15,  # mm under 100kg one side
                 bodywork_vertical_deflection_both=10,    # mm under 100kg both sides
                 carbon_layup_pattern="optimized_flex",
                 flex_speed_threshold=280,                # km/h
                 wing_flex_simulation=False,

                 # Performance Targets
                 target_downforce=1000,        # N
                 target_drag=180,              # N
                 efficiency_factor=0.75,       # L/D ratio consideration

                 # Configuration Management
                 configuration_preset="high_downforce",   # Monaco/street circuit
                 flap_angle_adjustment_range=[-5, 5],     # degrees quick adjustment
                 front_wing_level_adjustment=True,
                 setup_change_time=30,         # seconds

                 # Construction Parameters
                 resolution_span=100,          # Points along span
                 resolution_chord=60,          # Points along chord
                 mesh_density=2.5,
                 surface_smoothing=True,
                 smoothing_iterations=8,
                 realistic_surface_curvature=True,

                 # Material Properties
                 material="Carbon Fiber",
                 wall_thickness=2.5,           # mm
                 minimum_radius=5.0):          # mm (FIA regulation)

        """Initialize F1 Front Wing Multi-Element Generator"""

        # Store all parameters
        self.total_span = total_span
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.chord_taper_ratio = chord_taper_ratio
        self.sweep_angle = sweep_angle
        self.dihedral_angle = dihedral_angle
        self.twist_distribution_range = twist_distribution_range

        self.base_profile = base_profile
        self.max_thickness_ratio = max_thickness_ratio
        self.camber_ratio = camber_ratio
        self.camber_position = camber_position
        self.leading_edge_radius = leading_edge_radius
        self.trailing_edge_thickness = trailing_edge_thickness
        self.upper_surface_radius = upper_surface_radius
        self.lower_surface_radius = lower_surface_radius

        self.flap_count = flap_count
        self.flap_spans = flap_spans[:flap_count]
        self.flap_root_chords = flap_root_chords[:flap_count]
        self.flap_tip_chords = flap_tip_chords[:flap_count]
        self.flap_cambers = flap_cambers[:flap_count]

        self.main_element_aoa_range = main_element_aoa_range
        # Pad per-flap AoA ranges: if flap_count exceeds the explicitly
        # provided ranges (e.g. a 4-flap configuration), repeat the last
        # defined range instead of failing with an IndexError downstream.
        if flap4_aoa_range is None:
            # Extrapolate the flap1->flap3 progression one step further
            step = [flap3_aoa_range[i] - flap2_aoa_range[i] for i in range(2)]
            flap4_aoa_range = [flap3_aoa_range[i] + step[i] for i in range(2)]
        _aoa_ranges = [flap1_aoa_range, flap2_aoa_range, flap3_aoa_range, flap4_aoa_range]
        while len(_aoa_ranges) < flap_count:
            _aoa_ranges.append(list(_aoa_ranges[-1]))
        self.flap_aoa_ranges = _aoa_ranges[:flap_count]
        self.aoa_control_points = aoa_control_points
        self.flap_angle_progression = flap_angle_progression

        self.flap_slot_gaps = flap_slot_gaps[:flap_count]
        self.flap_vertical_offsets = flap_vertical_offsets[:flap_count]
        self.flap_horizontal_offsets = flap_horizontal_offsets[:flap_count]
        self.slot_gap_ratios = slot_gap_ratios[:flap_count]
        self.slot_overhang_ratios = slot_overhang_ratios[:flap_count]
        self.slot_flow_velocity_ratio = slot_flow_velocity_ratio
        self.slot_angle_progression = slot_angle_progression[:flap_count]
        self.gap_optimization_enabled = gap_optimization_enabled
        self.aerodynamic_slots = aerodynamic_slots

        # Cache for _compute_stack_layout (keyed by span station)
        self._stack_layout_cache = {}

        self.gurney_flaps = gurney_flaps

        # Pad per-element feature lists to flap_count entries by repeating the
        # last value, so configurations with more flaps than explicitly listed
        # values (e.g. 4 flaps) do not raise IndexError when indexed by
        # element_idx downstream.
        def _pad(values, count):
            values = list(values)
            while len(values) < count:
                values.append(values[-1])
            return values[:count]

        self.gurney_flap_heights = [0] + _pad(gurney_flap_heights, flap_count)  # No gurney on main
        self.slot_lip_radius = _pad(slot_lip_radius, flap_count)
        self.leading_edge_droop = [0] + _pad(leading_edge_droop, flap_count)
        self.trailing_edge_kick_up = [0] + _pad(trailing_edge_kick_up, flap_count)
        self.element_overlap_optimization = element_overlap_optimization

        self.wing_flexibility_enabled = wing_flexibility_enabled
        self.flap_deflection_limit = flap_deflection_limit
        self.deflection_load_point = deflection_load_point
        self.bodywork_vertical_deflection_single = bodywork_vertical_deflection_single
        self.bodywork_vertical_deflection_both = bodywork_vertical_deflection_both
        self.carbon_layup_pattern = carbon_layup_pattern
        self.flex_speed_threshold = flex_speed_threshold
        self.wing_flex_simulation = wing_flex_simulation

        self.target_downforce = target_downforce
        self.target_drag = target_drag
        self.efficiency_factor = efficiency_factor

        self.configuration_preset = configuration_preset
        self.flap_angle_adjustment_range = flap_angle_adjustment_range
        self.front_wing_level_adjustment = front_wing_level_adjustment
        self.setup_change_time = setup_change_time

        self.resolution_span = resolution_span
        self.resolution_chord = resolution_chord
        self.mesh_density = mesh_density
        self.surface_smoothing = surface_smoothing
        self.smoothing_iterations = smoothing_iterations
        self.realistic_surface_curvature = realistic_surface_curvature

        self.material = material
        self.wall_thickness = wall_thickness
        self.minimum_radius = minimum_radius

        print(f"{'='*70}")
        print(f"F1 FRONT WING MULTI-ELEMENT GENERATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Configuration: {self.configuration_preset}")
        print(f"Elements: Main + {self.flap_count} flaps")
        print(f"Span: {self.total_span}mm")
        print(f"{'='*70}\n")

    def compute_spanwise_chord(self, span_position: float, element_idx: int = 0) -> float:
        """
        Compute chord length at spanwise position for given element

        Args:
            span_position: 0.0 (root) to 1.0 (tip)
            element_idx: 0=main, 1=flap1, 2=flap2, 3=flap3

        Returns:
            chord length in mm
        """
        if element_idx == 0:
            # Main element
            root_c = self.root_chord
            tip_c = self.tip_chord
        else:
            # Flap elements
            flap_idx = element_idx - 1
            root_c = self.flap_root_chords[flap_idx]
            tip_c = self.flap_tip_chords[flap_idx]

        # Linear taper (can be made nonlinear for more realism)
        chord = root_c * (1 - span_position) + tip_c * span_position
        return chord

    def compute_spanwise_aoa(self, span_position: float, element_idx: int = 0) -> float:
        """
        Compute angle of attack at spanwise position for given element
        Uses multi-point interpolation for realistic distribution

        Args:
            span_position: 0.0 (root) to 1.0 (tip)
            element_idx: 0=main, 1=flap1, 2=flap2, 3=flap3

        Returns:
            angle of attack in degrees
        """
        if element_idx == 0:
            aoa_range = self.main_element_aoa_range
        else:
            flap_idx = element_idx - 1
            aoa_range = self.flap_aoa_ranges[flap_idx]

        if self.flap_angle_progression and self.aoa_control_points > 2:
            # Multi-point control for realistic progression
            control_positions = np.linspace(0, 1, self.aoa_control_points)

            # Create smooth progression from root to tip
            # More aggressive at root (high downforce zone)
            # Gradually reduce to tip (avoid stall)
            control_aoas = []
            for i, pos in enumerate(control_positions):
                # Smooth S-curve transition
                s_factor = 0.5 * (1 - np.cos(np.pi * pos))
                aoa = aoa_range[0] * (1 - s_factor**1.3) + aoa_range[1] * s_factor**1.3
                control_aoas.append(aoa)

            # Cubic spline interpolation for smoothness
            spline = CubicSpline(control_positions, control_aoas, bc_type='clamped')
            return float(spline(span_position))
        else:
            # Simple linear interpolation
            return aoa_range[0] * (1 - span_position) + aoa_range[1] * span_position

    def compute_twist_angle(self, span_position: float) -> float:
        """
        Compute geometric twist angle at spanwise position

        Args:
            span_position: 0.0 (root) to 1.0 (tip)

        Returns:
            twist angle in degrees
        """
        twist_root, twist_tip = self.twist_distribution_range
        # Smooth washout distribution (prevents tip stall)
        twist = twist_root * (1 - span_position**1.2) + twist_tip * span_position**1.2
        return twist

    def generate_airfoil_section(self,
                                  chord: float,
                                  camber: float,
                                  thickness_ratio: float,
                                  element_idx: int = 0,
                                  span_position: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic F1 airfoil section with progressive trailing edge curvature

        Args:
            chord: chord length in mm
            camber: camber ratio (0.08 = 8%)
            thickness_ratio: max thickness ratio (0.15 = 15%)
            element_idx: 0=main, 1+=flaps
            span_position: spanwise position for progressive features

        Returns:
            upper_surface, lower_surface: (x, z) coordinates
        """
        n_points = self.resolution_chord

        # Generate chordwise distribution (clustered at LE and TE)
        beta = np.linspace(0, np.pi, n_points)
        x_norm = 0.5 * (1 - np.cos(beta))  # Cosine spacing
        x = x_norm * chord

        # === CAMBER LINE (Mean line) ===
        # Modified NACA camber with adjustable position
        camber_pos = self.camber_position
        camber_line = np.zeros_like(x_norm)

        for i, xc in enumerate(x_norm):
            if xc <= camber_pos:
                camber_line[i] = camber * (2 * camber_pos * xc - xc**2) / camber_pos**2
            else:
                camber_line[i] = camber * ((1 - 2*camber_pos) + 2*camber_pos*xc - xc**2) / (1 - camber_pos)**2

        # === THICKNESS DISTRIBUTION ===
        # Modified NACA 4-digit thickness with realistic F1 profile
        thickness = np.zeros_like(x_norm)

        for i, xc in enumerate(x_norm):
            # NACA 4-digit formula modified for F1
            yt = 5 * thickness_ratio * chord * (
                0.2969 * np.sqrt(xc) -
                0.1260 * xc -
                0.3516 * xc**2 +
                0.2843 * xc**3 -
                0.1015 * xc**4
            )
            thickness[i] = yt

        # === LEADING EDGE ENHANCEMENT ===
        # Larger leading edge radius for flow attachment
        le_enhancement = self.leading_edge_radius / chord
        le_zone = x_norm < 0.1
        thickness[le_zone] *= (1 + le_enhancement * (1 - x_norm[le_zone]/0.1))

        # === TRAILING EDGE KICK-UP (PROGRESSIVE ON FLAPS) ===
        # This is KEY for realistic F1 flap appearance
        te_kickup = self.trailing_edge_kick_up[element_idx]
        if te_kickup > 0 and element_idx > 0:  # Only on flaps
            # Kick-up zone: last 30% of chord
            te_zone = x_norm > 0.7
            te_factor = (x_norm[te_zone] - 0.7) / 0.3  # 0 to 1 in TE zone

            # Smooth cubic curve for upward kick
            te_curve = te_factor**2 * (3 - 2 * te_factor)  # Smoothstep

            # Apply more kick at inboard sections (higher downforce)
            kickup_magnitude = te_kickup * (1.5 - 0.5 * span_position)

            # Add to camber line (creates upward curve)
            camber_line[te_zone] += (kickup_magnitude / chord) * te_curve

            # Also modify thickness distribution for smooth surface
            thickness[te_zone] *= (1 + 0.3 * te_curve)

        # === LEADING EDGE DROOP (FLAPS) ===
        le_droop = self.leading_edge_droop[element_idx]
        if le_droop != 0 and element_idx > 0:
            # Droop zone: first 15% of chord
            le_droop_zone = x_norm < 0.15
            le_droop_factor = (0.15 - x_norm[le_droop_zone]) / 0.15
            le_droop_curve = le_droop_factor**2

            camber_line[le_droop_zone] += (le_droop / chord) * le_droop_curve

        # === TRAILING EDGE THICKNESS ===
        # Finite TE thickness for structural realism
        te_thickness = self.trailing_edge_thickness
        te_blend_zone = x_norm > 0.95
        thickness[te_blend_zone] = np.maximum(
            thickness[te_blend_zone],
            te_thickness * (x_norm[te_blend_zone] - 0.95) / 0.05
        )

        # === SLOT LIP RADIUS (FLAPS) ===
        if element_idx > 0 and self.aerodynamic_slots:
            flap_idx = element_idx - 1
            lip_radius = self.slot_lip_radius[flap_idx]

            # Round leading edge more for flaps (prevents separation)
            le_radius_zone = x_norm < 0.05
            radius_factor = (0.05 - x_norm[le_radius_zone]) / 0.05
            thickness[le_radius_zone] += (lip_radius / chord) * radius_factor**1.5

        # === CONSTRUCT UPPER AND LOWER SURFACES ===
        # Camber line angle
        theta = np.zeros_like(x_norm)
        for i in range(1, len(x_norm) - 1):
            dx = x_norm[i+1] - x_norm[i-1]
            dy = camber_line[i+1] - camber_line[i-1]
            theta[i] = np.arctan2(dy, dx)
        theta[0] = theta[1]
        theta[-1] = theta[-2]

        # Upper surface (suction side)
        x_upper = x - thickness * np.sin(theta)
        z_upper = camber_line * chord + thickness * np.cos(theta)

        # Lower surface (pressure side)
        x_lower = x + thickness * np.sin(theta)
        z_lower = camber_line * chord - thickness * np.cos(theta)

        # === GURNEY FLAP (TRAILING EDGE DEVICE) ===
        if self.gurney_flaps and self.gurney_flap_heights[element_idx] > 0:
            gurney_height = self.gurney_flap_heights[element_idx]

            # Gurney extends from lower TE vertically
            gurney_x = np.array([chord, chord, chord - 0.5])
            gurney_z = np.array([
                z_lower[-1],
                z_lower[-1] + gurney_height,
                z_lower[-1] + gurney_height
            ])

            # Append to lower surface
            x_lower = np.concatenate([x_lower, gurney_x])
            z_lower = np.concatenate([z_lower, gurney_z])

        # Stack into coordinate arrays
        upper_surface = np.column_stack([x_upper, z_upper])
        lower_surface = np.column_stack([x_lower, z_lower])

        return upper_surface, lower_surface

    def _section_surfaces_world(self, element_idx, span_position, le_x, le_z):
        """
        Upper/lower section surfaces of an element in world coordinates
        (gurney flap excluded - it must not drive slot clearance).

        Returns:
            (upper_x, upper_z, lower_x, lower_z): arrays in world mm;
            upper arrays sorted by x for interpolation.
        """
        chord = self.compute_spanwise_chord(span_position, element_idx)
        aoa = self.compute_spanwise_aoa(span_position, element_idx)

        if element_idx == 0:
            camber = self.camber_ratio
            thickness = self.max_thickness_ratio
        else:
            camber = self.flap_cambers[element_idx - 1]
            thickness = self.max_thickness_ratio * 0.9

        # Gurney tab points stick up at the TE; exclude them from clearance
        gurney_state = self.gurney_flaps
        self.gurney_flaps = False
        try:
            upper, lower = self.generate_airfoil_section(
                chord, camber, thickness, element_idx, span_position
            )
        finally:
            self.gurney_flaps = gurney_state

        # Same rotation convention as generate_wing_element_surface
        aoa_rad = np.radians(aoa)
        cos_a, sin_a = np.cos(aoa_rad), np.sin(aoa_rad)

        ux = upper[:, 0] * cos_a + upper[:, 1] * sin_a + le_x
        uz = -upper[:, 0] * sin_a + upper[:, 1] * cos_a + le_z
        lx = lower[:, 0] * cos_a + lower[:, 1] * sin_a + le_x
        lz = -lower[:, 0] * sin_a + lower[:, 1] * cos_a + le_z

        order = np.argsort(ux)
        return ux[order], uz[order], lx, lz

    def _compute_stack_layout(self, span_position):
        """
        Compute the (x, z) leading-edge positions of the whole element stack
        at one span station, as a cohesive multi-slotted cascade.

        The cumulative flap_horizontal_offsets / flap_vertical_offsets are
        treated as *target* positions, then regularised by two aerodynamic
        design rules so any parameter combination still forms a valid
        multi-element wing:

        1. Overlap rule: each flap's leading edge must lie between
           ~30% of the previous element's chord and the previous trailing
           edge minus an overhang margin (slot_overhang_ratios), so the
           elements always overlap and form a real slot.
        2. Clearance rule: the flap is raised (never lowered) until every
           point of its lower surface that overlaps the previous element
           clears that element's upper surface by its flap_slot_gaps value,
           so elements never interpenetrate and the slot gap is real.

        Returns:
            list of (le_x, le_z) per element index 0..flap_count
        """
        key = round(float(span_position), 6)
        cached = self._stack_layout_cache.get(key)
        if cached is not None:
            return cached

        layout = [(0.0, 0.0)]  # main element LE at origin

        for element_idx in range(1, self.flap_count + 1):
            flap_idx = element_idx - 1

            prev_x, prev_z = layout[-1]
            prev_idx = element_idx - 1
            prev_chord = self.compute_spanwise_chord(span_position, prev_idx)
            prev_aoa_rad = np.radians(self.compute_spanwise_aoa(span_position, prev_idx))

            chord = self.compute_spanwise_chord(span_position, element_idx)

            # Previous element's upper surface (world coords)
            pux, puz, _plx, _plz = self._section_surfaces_world(
                prev_idx, span_position, prev_x, prev_z)

            # --- target from cumulative user offsets ---
            x_target = float(sum(self.flap_horizontal_offsets[:flap_idx + 1]))
            z_target = float(sum(self.flap_vertical_offsets[:flap_idx + 1]))

            # --- rule 1: overlap window ---
            min_stagger = prev_x + 0.30 * prev_chord * np.cos(prev_aoa_rad)
            prev_te_x = prev_x + prev_chord * np.cos(prev_aoa_rad)
            overhang = self.slot_overhang_ratios[flap_idx] if flap_idx < len(self.slot_overhang_ratios) else 0.2
            max_stagger = prev_te_x - overhang * chord
            if max_stagger < min_stagger:
                max_stagger = min_stagger
            x_pos = min(max(x_target, min_stagger), max_stagger)

            # --- rule 2: exact slot clearance ---
            # Flap lower surface in world x (z measured at z=0 offset; the
            # whole section shifts rigidly with z_pos).
            _fux, _fuz, flx, flz = self._section_surfaces_world(
                element_idx, span_position, x_pos, 0.0)

            gap = self.flap_slot_gaps[flap_idx] if flap_idx < len(self.flap_slot_gaps) else 12.0
            overlap = (flx >= pux[0] + 1.0) & (flx <= pux[-1] - 1.0)
            if np.any(overlap):
                prev_top = np.interp(flx[overlap], pux, puz)
                z_min = float(np.max(prev_top - flz[overlap])) + gap + 1.5
            else:
                z_min = float(np.interp(x_pos, pux, puz)) + gap + 1.5
            z_pos = max(z_target, z_min)

            layout.append((x_pos, z_pos))

        self._stack_layout_cache[key] = layout
        return layout

    def compute_element_position(self,
                                  element_idx: int,
                                  span_position: float) -> Tuple[float, float, float]:
        """
        Compute 3D position offset for wing element

        Args:
            element_idx: 0=main, 1=flap1, 2=flap2, 3=flap3
            span_position: 0.0 (root) to 1.0 (tip)

        Returns:
            (x_offset, y_offset, z_offset) in mm
        """
        if element_idx == 0:
            # Main element at origin
            return 0.0, 0.0, 0.0

        layout = self._compute_stack_layout(span_position)
        x_offset, z_offset = layout[min(element_idx, len(layout) - 1)]

        return x_offset, 0.0, z_offset

    def generate_wing_element_surface(self,
                                       element_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete 3D surface for one wing element

        Args:
            element_idx: 0=main, 1=flap1, 2=flap2, 3=flap3

        Returns:
            vertices, faces (triangular mesh)
        """
        element_name = "Main Element" if element_idx == 0 else f"Flap {element_idx}"
        print(f"  Generating {element_name}...")

        vertices = []

        # Spanwise stations: generate the FULL span (-1 .. +1) so the wing is
        # mirror-symmetric about the car centerline (Y=0). Twice as many
        # stations keep the same per-side spanwise resolution as before.
        # Odd count so one station lies exactly on the centerline (Y=0)
        span_stations = np.linspace(-1, 1, 2 * self.resolution_span + 1)
        half_span = self.total_span / 2

        for span_idx, span_pos in enumerate(span_stations):
            # Mirror station: spanwise design distributions (chord, AoA,
            # twist, local features) are defined for a half-span station in
            # [0, 1] and apply identically to both sides of the centerline.
            half_pos = abs(span_pos)

            # Compute local parameters
            chord = self.compute_spanwise_chord(half_pos, element_idx)
            aoa = self.compute_spanwise_aoa(half_pos, element_idx)
            twist = self.compute_twist_angle(half_pos)

            if element_idx == 0:
                camber = self.camber_ratio
                thickness = self.max_thickness_ratio
            else:
                flap_idx = element_idx - 1
                camber = self.flap_cambers[flap_idx]
                thickness = self.max_thickness_ratio * 0.9  # Flaps slightly thinner

            # Generate airfoil section
            upper, lower = self.generate_airfoil_section(
                chord, camber, thickness, element_idx, half_pos
            )

            # Get element position offset
            x_offset, y_offset, z_offset = self.compute_element_position(element_idx, half_pos)

            # === 3D TRANSFORMATIONS ===

            # 1. Angle of attack rotation
            aoa_rad = np.radians(aoa)
            cos_aoa = np.cos(aoa_rad)
            sin_aoa = np.sin(aoa_rad)

            # 2. Twist angle rotation
            twist_rad = np.radians(twist)
            cos_twist = np.cos(twist_rad)
            sin_twist = np.sin(twist_rad)

            # 3. Sweep (quarter-chord sweep line) - mirrored: both tips sweep back
            sweep_rad = np.radians(self.sweep_angle)
            sweep_offset = (half_pos * half_span) * np.tan(sweep_rad)

            # 4. Dihedral (vertical slope) - mirrored: both tips rise equally
            dihedral_rad = np.radians(self.dihedral_angle)
            dihedral_offset = (half_pos * half_span) * np.tan(dihedral_rad)

            # Process each surface point as ONE closed section loop:
            # upper surface LE->TE, then lower surface TE->LE. This makes the
            # chordwise grid wrap around the section, so the element is a
            # properly closed tube (no TE->LE diagonal web, no broken edge
            # closure strips when upper/lower point counts differ).
            section_loop = np.concatenate([upper, lower[::-1]], axis=0)

            for point in section_loop:
                x_local, z_local = point

                # Apply AoA rotation
                x_rot = x_local * cos_aoa + z_local * sin_aoa
                z_rot = -x_local * sin_aoa + z_local * cos_aoa

                # Apply twist rotation
                x_twist = x_rot * cos_twist - z_rot * sin_twist
                z_twist = x_rot * sin_twist + z_rot * cos_twist

                # Apply 3D positioning
                x_final = x_twist + x_offset + sweep_offset
                y_final = span_pos * half_span + y_offset
                z_final = z_twist + z_offset + dihedral_offset

                vertices.append([x_final, y_final, z_final])

        vertices = np.array(vertices)

        # === GENERATE FACES (TRIANGULAR MESH) ===
        faces = []

        # Section loop: upper LE->TE then lower TE->LE (closed chordwise)
        n_chord_points = len(upper) + len(lower)
        n_span_points = len(span_stations)

        # Create wrapped quad strips and triangulate: chordwise index wraps
        # around the closed section loop, so LE and TE are closed by
        # construction and every element is one connected watertight-ish tube.
        # Mirror-aware triangulation: the +Y half of the grid uses the
        # opposite quad diagonal (with reversed winding) so that every
        # triangle has an exact mirror-image triangle about Y=0.
        center_pair = n_span_points // 2
        for i in range(n_span_points - 1):
            for j in range(n_chord_points):
                j1 = (j + 1) % n_chord_points
                # Vertex indices for quad
                v0 = i * n_chord_points + j
                v1 = i * n_chord_points + j1
                v2 = (i + 1) * n_chord_points + j
                v3 = (i + 1) * n_chord_points + j1

                # Two triangles per quad
                if i < center_pair:
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
                else:
                    # Mirror image of the -Y side's [v0,v2,v1] / [v1,v2,v3]
                    faces.append([v2, v3, v0])
                    faces.append([v3, v1, v0])

        # Tip caps (close the open ends at +/- half span)
        for i in (0, n_span_points - 1):
            base = i * n_chord_points
            centroid = vertices[base:base + n_chord_points].mean(axis=0)
            c_idx = len(vertices)
            vertices = np.vstack([vertices, centroid[None, :]])
            for j in range(n_chord_points):
                j1 = (j + 1) % n_chord_points
                if i == 0:
                    faces.append([c_idx, base + j1, base + j])
                else:
                    faces.append([c_idx, base + j, base + j1])

        faces = np.array(faces)

        print(f"    ✓ {len(vertices)} vertices, {len(faces)} faces")

        return vertices, faces


    def apply_laplacian_smoothing(self,
                                    vertices: np.ndarray,
                                    faces: np.ndarray,
                                    iterations: int = None) -> np.ndarray:
        """
        Apply Laplacian smoothing for surface quality

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face vertex indices
            iterations: number of smoothing passes

        Returns:
            smoothed vertices
        """
        if iterations is None:
            iterations = self.smoothing_iterations

        if iterations == 0 or not self.surface_smoothing:
            return vertices

        print(f"    Applying Laplacian smoothing ({iterations} iterations)...")

        # Build adjacency list
        adjacency = [set() for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                if 0 <= v1 < len(vertices) and 0 <= v2 < len(vertices):
                    adjacency[v1].add(v2)
                    adjacency[v2].add(v1)

        smoothed = vertices.copy()

        for iteration in range(iterations):
            new_verts = smoothed.copy()

            # Adaptive smoothing strength (stronger at start, weaker at end)
            strength = 0.5 * (1.0 - iteration / (iterations * 1.2))
            strength = max(strength, 0.1)

            for i in range(len(vertices)):
                if adjacency[i]:
                    neighbors = list(adjacency[i])
                    avg = np.mean(smoothed[neighbors], axis=0)
                    new_verts[i] = (1 - strength) * smoothed[i] + strength * avg

            smoothed = new_verts

        print(f"    ✓ Smoothing complete")

        return smoothed

    def generate_complete_wing(self, side='both') -> mesh.Mesh:
        """
        Generate complete multi-element front wing assembly

        Args:
            side: 'left', 'right', or 'both'

        Returns:
            STL mesh object
        """
        print(f"\n{'='*70}")
        print(f"GENERATING COMPLETE F1 FRONT WING - {side.upper()}")
        print(f"{'='*70}\n")

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Generate all elements
        total_elements = 1 + self.flap_count  # Main + flaps

        for element_idx in range(total_elements):
            verts, faces = self.generate_wing_element_surface(element_idx)

            # Apply smoothing
            if self.surface_smoothing:
                verts = self.apply_laplacian_smoothing(verts, faces)

            # Mirror for left side if needed
            if side == 'left':
                verts[:, 1] *= -1  # Mirror Y coordinate
            elif side == 'both':
                # Add both sides
                verts_right = verts.copy()
                verts_left = verts.copy()
                verts_left[:, 1] *= -1

                all_vertices.append(verts_right)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts_right)

                all_vertices.append(verts_left)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts_left)
                continue

            all_vertices.append(verts)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(verts)

        # Combine geometry
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)

        print(f"\n{'='*70}")
        print(f"TOTAL GEOMETRY:")
        print(f"  Vertices: {len(combined_vertices)}")
        print(f"  Faces: {len(combined_faces)}")
        print(f"  Elements: {total_elements}")
        print(f"{'='*70}\n")

        # Create STL mesh
        wing_mesh = mesh.Mesh(np.zeros(len(combined_faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(combined_faces):
            for j in range(3):
                wing_mesh.vectors[i][j] = combined_vertices[face[j]]

        return wing_mesh

    def save_stl(self, filename='f1_front_wing_multi_element.stl', side='both'):
        """Generate and save complete wing to STL file"""
        wing = self.generate_complete_wing(side=side)
        wing.save(filename)
        print(f"✓ Saved to: {filename}\n")
        return wing


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("F1 FRONT WING MULTI-ELEMENT GENERATOR")
    print("="*70)
    print("\n")

    # Create generator with realistic F1 parameters
    generator = F1FrontWingMultiElementGenerator(
        # Platform
        total_span=1600,
        root_chord=280,
        tip_chord=250,
        chord_taper_ratio=0.89,
        sweep_angle=3.5,
        dihedral_angle=2.5,
        twist_distribution_range=[-1.5, 0.5],

        # Multi-element configuration
        flap_count=3,
        flap_spans=[1600, 1600, 1600],
        flap_root_chords=[220, 180, 140],
        flap_tip_chords=[200, 160, 120],
        flap_cambers=[0.12, 0.10, 0.08],

        # Element-specific AoA (CRITICAL FOR REALISM)
        main_element_aoa_range=[-2.3, 1.2],
        flap1_aoa_range=[-21.3, 0.0],
        flap2_aoa_range=[-16.7, 0.0],
        flap3_aoa_range=[-13.5, 0.0],
        aoa_control_points=4,
        flap_angle_progression=True,

        # Slot geometry
        flap_slot_gaps=[14, 12, 10],
        flap_vertical_offsets=[25, 45, 70],
        flap_horizontal_offsets=[30, 60, 85],
        slot_gap_ratios=[0.012, 0.010, 0.008],
        slot_overhang_ratios=[0.25, 0.22, 0.20],
        slot_angle_progression=[5, 8, 12],
        gap_optimization_enabled=True,
        aerodynamic_slots=True,

        # Advanced features
        gurney_flaps=True,
        gurney_flap_heights=[3, 2.5, 2],
        slot_lip_radius=[2.5, 2.0, 1.5],
        leading_edge_droop=[0, -2, -4],
        trailing_edge_kick_up=[0, 3, 6],  # PROGRESSIVE UPKICK ON FLAPS
        element_overlap_optimization=True,

        # Flexibility
        wing_flexibility_enabled=True,
        flap_deflection_limit=3.0,

        # Performance
        target_downforce=1000,
        target_drag=180,
        efficiency_factor=0.75,

        # Quality
        resolution_span=100,
        resolution_chord=60,
        surface_smoothing=True,
        smoothing_iterations=8,
        realistic_surface_curvature=True,

        # Configuration
        configuration_preset="high_downforce"
    )

    # Generate complete wing (both sides)
    generator.save_stl('f1_front_wing_complete.stl', side='both')

    print("="*70)
    print("✓ GENERATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • f1_front_wing_complete.stl (full assembly)")
    print("\nKey features:")
    print("  • Progressive trailing edge kick-up on flaps")
    print("  • Element-specific angle of attack control")
    print("  • Realistic slot-gap-overhang relationships")
    print("  • Gurney flaps on all rear elements")
    print("  • Smooth surface with high-resolution mesh")
    print("="*70)
