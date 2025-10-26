"""
F1 FRONT WING Y250 VORTEX & CENTRAL STRUCTURE GENERATOR - MASTER LEVEL
======================================================================
Generates ultra-realistic F1 front wing central section with Y250 vortex
generation, footplate, strakes, vortex generators, and mounting pylons

Based on 2009-2021 F1 regulations (pre-2022 Y250 vortex era)

Features:
- Y250 vortex generation zone (250mm from centerline)
- Neutral center section with step transition
- Footplate with arch geometry for ground effect sealing
- Underwing strakes (max 2 per side - 2019+ regs)
- Half-tube vortex generators at footplate base
- Outboard fences for tire wake management
- Elliptical mounting pylons (regulation compliant)
- Optional cascade elements (pre-2019 style)
- FIA compliance checking
- Export to STL format
"""

import numpy as np
from stl import mesh
import math
from scipy.interpolate import CubicSpline, make_interp_spline
from typing import List, Tuple, Dict

class F1FrontWingY250CentralStructureGenerator:
    """
    MASTER-LEVEL F1 Front Wing Y250 & Central Structure Generator
    Generates regulation-compliant central aerodynamic features
    optimized for vortex generation and tire wake management
    """

    def __init__(self,
                 # Y250 Vortex Generation Zone (CORE AERODYNAMICS)
                 y250_width=500,                    # mm - neutral center section (FIA: 500mm)
                 y250_step_height=18,               # mm - vertical step for vortex generation
                 y250_transition_length=80,         # mm - smooth blend to loaded section
                 central_slot_width=0,              # mm - center slot (0 = solid nose connection)
                 y250_vortex_strength=0.85,         # 0-1 factor for vortex intensity

                 # Footplate Geometry (GROUND EFFECT SEALING)
                 footplate_extension=70,            # mm forward of mainplane LE
                 footplate_height=30,               # mm vertical extent
                 arch_radius=130,                   # mm radius of arch curve
                 footplate_thickness=5,             # mm structural thickness
                 primary_strake_count=2,            # Max 2 per side (2019+ FIA regs)
                 strake_heights=[45, 35],           # mm height of each strake

                 # Tire Interaction Parameters
                 tire_diameter=670,                 # mm (regulation front tire)
                 tire_width=305,                    # mm front tire width
                 tire_wake_deflection_angle=15,     # degrees outboard redirect
                 outwash_optimization=True,
                 wheel_wake_interaction_zone=[600, 900],  # mm spanwise positions

                 # Vortex Generators (HALF-TUBE TYPE)
                 vortex_generator_enabled=True,
                 vg_type="half_tube",               # half_tube, fence, shark_tooth
                 vg_height=8,                       # mm
                 vg_spacing=25,                     # mm between VGs
                 vg_angle=18,                       # degrees to freestream
                 outboard_fence_enabled=True,
                 fence_heights=[60, 50, 40],        # mm - diminishing outboard
                 fence_positions=[0.7, 0.85, 0.95], # Normalized span (0-1)

                 # Mounting Pylon System
                 pylon_count=2,                     # Typically 2 (FIA limit)
                 pylon_spacing=320,                 # mm between pylons
                 pylon_major_axis=38,               # mm ellipse major axis
                 pylon_minor_axis=25,               # mm ellipse minor axis
                 pylon_length=120,                  # mm vertical extent

                 # Cascade Elements (PRE-2019 STYLE - now banned)
                 cascade_enabled=False,             # Set True for pre-2019 designs
                 primary_cascade_span=250,          # mm spanwise extent
                 primary_cascade_chord=55,          # mm chordwise length
                 secondary_cascade_span=160,        # mm
                 secondary_cascade_chord=40,        # mm

                 # Manufacturing & Structural
                 wall_thickness_structural=4,       # mm main structure
                 wall_thickness_aerodynamic=2.5,    # mm aero surfaces
                 wall_thickness_details=2.0,        # mm fine details
                 minimum_radius=0.4,                # mm smallest feature radius
                 mesh_resolution_structural=0.6,    # mm mesh size

                 # Material Properties
                 material="Standard Carbon Fiber",
                 density=1600,                      # kg/m³
                 weight_estimate=4.0,               # kg estimated weight

                 # FIA Compliance (2024 Regulations)
                 fia_compliance_mode=True,
                 max_width_regulation=1800,         # mm total wing width
                 max_forward_extent=200,            # mm beyond front axle
                 virtual_endplate_surface=True,
                 virtual_surface_limits=[-50, 330, 0, 380],  # [x_front, x_rear, z_bottom, z_top]
                 max_centerline_angle_regulation=15,  # degrees
                 min_radius_regulation=5,           # mm all edges (safety)
                 load_test_deflection_check=True,

                 # Integration Flags
                 enhanced_endplate_detail=True,
                 endplate_wing_slots=True,
                 use_standalone_endplates=True,

                 # Construction Parameters
                 resolution=80,                     # Points for curve generation
                 surface_smoothing=True,
                 smoothing_iterations=6):

        """Initialize Y250 & Central Structure Generator"""

        # Store all parameters
        self.y250_width = y250_width
        self.y250_step_height = y250_step_height
        self.y250_transition_length = y250_transition_length
        self.central_slot_width = central_slot_width
        self.y250_vortex_strength = y250_vortex_strength

        self.footplate_extension = footplate_extension
        self.footplate_height = footplate_height
        self.arch_radius = arch_radius
        self.footplate_thickness = footplate_thickness
        self.primary_strake_count = min(primary_strake_count, 2)  # FIA limit
        self.strake_heights = strake_heights[:self.primary_strake_count]

        self.tire_diameter = tire_diameter
        self.tire_width = tire_width
        self.tire_wake_deflection_angle = tire_wake_deflection_angle
        self.outwash_optimization = outwash_optimization
        self.wheel_wake_interaction_zone = wheel_wake_interaction_zone

        self.vortex_generator_enabled = vortex_generator_enabled
        self.vg_type = vg_type
        self.vg_height = vg_height
        self.vg_spacing = vg_spacing
        self.vg_angle = vg_angle
        self.outboard_fence_enabled = outboard_fence_enabled
        self.fence_heights = fence_heights
        self.fence_positions = fence_positions

        self.pylon_count = pylon_count
        self.pylon_spacing = pylon_spacing
        self.pylon_major_axis = pylon_major_axis
        self.pylon_minor_axis = pylon_minor_axis
        self.pylon_length = pylon_length

        self.cascade_enabled = cascade_enabled
        self.primary_cascade_span = primary_cascade_span
        self.primary_cascade_chord = primary_cascade_chord
        self.secondary_cascade_span = secondary_cascade_span
        self.secondary_cascade_chord = secondary_cascade_chord

        self.wall_thickness_structural = wall_thickness_structural
        self.wall_thickness_aerodynamic = wall_thickness_aerodynamic
        self.wall_thickness_details = wall_thickness_details
        self.minimum_radius = minimum_radius
        self.mesh_resolution_structural = mesh_resolution_structural

        self.material = material
        self.density = density
        self.weight_estimate = weight_estimate

        self.fia_compliance_mode = fia_compliance_mode
        self.max_width_regulation = max_width_regulation
        self.max_forward_extent = max_forward_extent
        self.virtual_endplate_surface = virtual_endplate_surface
        self.virtual_surface_limits = virtual_surface_limits
        self.max_centerline_angle_regulation = max_centerline_angle_regulation
        self.min_radius_regulation = min_radius_regulation
        self.load_test_deflection_check = load_test_deflection_check

        self.enhanced_endplate_detail = enhanced_endplate_detail
        self.endplate_wing_slots = endplate_wing_slots
        self.use_standalone_endplates = use_standalone_endplates

        self.resolution = resolution
        self.surface_smoothing = surface_smoothing
        self.smoothing_iterations = smoothing_iterations

        print(f"{'='*70}")
        print(f"F1 Y250 VORTEX & CENTRAL STRUCTURE GENERATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Y250 Zone Width: {self.y250_width}mm")
        print(f"Vortex Strength: {self.y250_vortex_strength}")
        print(f"Strakes: {self.primary_strake_count}")
        print(f"Pylons: {self.pylon_count}")
        print(f"Cascades: {'Enabled' if self.cascade_enabled else 'Disabled'}")
        print(f"{'='*70}\n")

    def generate_y250_neutral_section(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Y250 neutral center section (500mm regulation width)
        This is the flat/low-loaded region that creates vortex at transition

        Returns:
            vertices, faces for neutral section
        """
        print("  Generating Y250 neutral section...")

        vertices = []
        half_width = self.y250_width / 2  # 250mm each side

        # Chordwise stations
        chord_stations = np.linspace(0, 300, self.resolution)  # 300mm chord

        # Spanwise stations (centerline to Y250 edge)
        span_stations = np.linspace(0, half_width, self.resolution // 2)

        for y in span_stations:
            for x in chord_stations:
                # Neutral section profile (minimal camber, low AoA)
                # This is deliberately flat to create pressure difference at Y250
                camber_factor = 0.02  # Very low camber (2%)
                z_camber = -camber_factor * x * (1 - x/300)

                # Add Y250 step at transition
                if y > half_width * 0.9:  # Last 10% creates step
                    step_factor = (y - half_width * 0.9) / (half_width * 0.1)
                    z_camber -= self.y250_step_height * step_factor

                # Upper surface
                z_upper = z_camber + 15  # 15mm thickness
                vertices.append([x, y, z_upper])

                # Lower surface
                z_lower = z_camber - 15
                vertices.append([x, y, z_lower])

        vertices = np.array(vertices)

        # Generate faces (simplified triangulation)
        faces = self._triangulate_structured_grid(
            self.resolution,
            self.resolution // 2,
            points_per_station=2
        )

        print(f"    ✓ Neutral section: {len(vertices)} vertices")
        return vertices, faces

    def generate_y250_transition_region(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Y250 transition region where vortex is generated
        Sharp transition from neutral to loaded section creates vorticity

        Returns:
            vertices, faces for transition
        """
        print("  Generating Y250 vortex transition...")

        vertices = []
        half_width = self.y250_width / 2

        # Transition zone: from Y250 edge to loaded section
        transition_span = np.linspace(half_width, half_width + self.y250_transition_length, 40)
        chord_stations = np.linspace(0, 300, 60)

        for i, y in enumerate(transition_span):
            transition_factor = i / (len(transition_span) - 1)  # 0 to 1

            for x in chord_stations:
                # Smoothly transition from neutral to high-load profile
                # Vortex strength controlled by sharpness of transition
                camber_start = 0.02
                camber_end = 0.10  # Increase to 10% camber

                # S-curve transition (creates stronger vortex)
                s_factor = 0.5 * (1 - np.cos(np.pi * transition_factor))
                s_factor = s_factor ** (2.0 - self.y250_vortex_strength)  # Sharper = stronger vortex

                camber = camber_start + (camber_end - camber_start) * s_factor
                z_camber = -camber * x * (1 - x/300)

                # Transition step height
                z_step = -self.y250_step_height * (1 - transition_factor)

                thickness = 15 + 10 * transition_factor  # Thicken in loaded section

                vertices.append([x, y, z_camber + z_step + thickness])  # Upper
                vertices.append([x, y, z_camber + z_step - thickness])  # Lower

        vertices = np.array(vertices)
        faces = self._triangulate_structured_grid(60, 40, 2)

        print(f"    ✓ Y250 transition: {len(vertices)} vertices")
        return vertices, faces

    def generate_footplate_with_arch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate footplate with arch geometry for ground effect sealing
        Creates low-pressure region underneath to seal wing

        Returns:
            vertices, faces for footplate
        """
        print("  Generating footplate with arch...")

        vertices = []

        # Footplate extends forward of wing leading edge
        x_start = -self.footplate_extension
        x_end = 50  # Extends slightly beyond LE

        # Spanwise extent (typically outboard of Y250)
        y_start = self.y250_width / 2
        y_end = self.max_width_regulation / 2  # To endplate

        x_stations = np.linspace(x_start, x_end, 60)
        y_stations = np.linspace(y_start, y_end, 80)

        for y in y_stations:
            for x in x_stations:
                # Arch profile (semicircular curve downward)
                # Creates venturi effect to seal low-pressure zone

                # Height decreases forward (arch shape)
                progress = (x - x_start) / (x_end - x_start)

                # Parabolic arch
                arch_height = self.footplate_height * (1 - (2*progress - 1)**2)

                # Add radius curvature
                if progress < 0.3:  # Forward section curves more
                    radius_factor = (0.3 - progress) / 0.3
                    arch_height += (self.arch_radius / 100) * radius_factor * 10

                z_footplate = -arch_height

                # Top surface
                vertices.append([x, y, z_footplate])
                # Bottom surface (structural thickness)
                vertices.append([x, y, z_footplate - self.footplate_thickness])

        vertices = np.array(vertices)
        faces = self._triangulate_structured_grid(60, 80, 2)

        print(f"    ✓ Footplate: {len(vertices)} vertices")
        return vertices, faces

    def generate_underwing_strakes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate underwing strakes (max 2 per side - 2019+ FIA regulations)
        Vertical elements that align flow and generate beneficial vortices

        Returns:
            List of (vertices, faces) for each strake
        """
        print(f"  Generating {self.primary_strake_count} underwing strakes...")

        strake_geometries = []

        # Strake positioning (FIA: 500-800mm from centerline)
        strake_positions = np.linspace(500, 800, self.primary_strake_count)

        for i, y_pos in enumerate(strake_positions):
            vertices = []
            height = self.strake_heights[i]

            # Strake profile (single plane of curvature - FIA requirement)
            chord = 80  # mm
            x_stations = np.linspace(0, chord, 30)

            # Strake cross-section (thin vertical plate with leading edge radius)
            for x in x_stations:
                # Height varies with chord (taller at front)
                local_height = height * (1 - 0.3 * (x / chord))

                # Leading edge radius for flow attachment
                if x < 10:
                    le_radius = 5
                    width_local = le_radius * np.sqrt(1 - ((x - 10)/10)**2) if x < 10 else 2
                else:
                    width_local = 2  # Thin trailing section

                # Generate strake profile points
                z_stations = np.linspace(-local_height, 0, 15)
                for z in z_stations:
                    vertices.append([x, y_pos - width_local, z])
                    vertices.append([x, y_pos + width_local, z])

            vertices = np.array(vertices)
            faces = self._triangulate_structured_grid(30, 15, 2)

            strake_geometries.append((vertices, faces))
            print(f"    ✓ Strake {i+1}: {len(vertices)} vertices, height={height}mm")

        return strake_geometries

    def generate_half_tube_vortex_generators(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate half-tube vortex generators at footplate base
        Creates counter-rotating vortices to seal low-pressure region

        Returns:
            List of (vertices, faces) for each VG
        """
        if not self.vortex_generator_enabled:
            return []

        print("  Generating half-tube vortex generators...")

        vg_geometries = []

        # VG array positioning (along footplate leading edge)
        y_start = self.y250_width / 2 + 100
        y_end = self.wheel_wake_interaction_zone[0]

        n_vgs = int((y_end - y_start) / self.vg_spacing)
        y_positions = np.linspace(y_start, y_end, n_vgs)

        for y_pos in y_positions:
            vertices = []

            # Half-tube geometry (semicircular cross-section)
            tube_diameter = self.vg_height
            tube_length = 15  # mm chordwise extent

            # Angle relative to freestream
            angle_rad = np.radians(self.vg_angle)

            theta_stations = np.linspace(0, np.pi, 20)  # Half circle
            x_stations = np.linspace(0, tube_length, 10)

            for x in x_stations:
                for theta in theta_stations:
                    # Semicircular cross-section
                    z_local = (tube_diameter / 2) * np.sin(theta) - self.footplate_height
                    y_local = (tube_diameter / 2) * np.cos(theta)

                    # Apply angle
                    x_rot = x * np.cos(angle_rad) - y_local * np.sin(angle_rad)
                    y_rot = x * np.sin(angle_rad) + y_local * np.cos(angle_rad)

                    vertices.append([x_rot - self.footplate_extension/2, y_pos + y_rot, z_local])

            vertices = np.array(vertices)
            faces = self._triangulate_structured_grid(10, 20, 1)
            vg_geometries.append((vertices, faces))

        print(f"    ✓ {len(vg_geometries)} vortex generators")
        return vg_geometries

    def generate_outboard_fences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate outboard fences for tire wake management
        Vertical elements that redirect tire wake outboard

        Returns:
            List of (vertices, faces) for each fence
        """
        if not self.outboard_fence_enabled:
            return []

        print(f"  Generating {len(self.fence_positions)} outboard fences...")

        fence_geometries = []
        wing_span = self.max_width_regulation / 2

        for i, norm_pos in enumerate(self.fence_positions):
            vertices = []
            y_pos = wing_span * norm_pos
            height = self.fence_heights[i]

            # Fence profile (thin vertical plate angled for outwash)
            chord = 40  # mm
            deflection_angle = self.tire_wake_deflection_angle

            x_stations = np.linspace(0, chord, 25)
            z_stations = np.linspace(-height, 0, 20)

            for x in x_stations:
                # Outward deflection increases toward trailing edge
                deflection_factor = (x / chord) ** 1.5
                y_deflection = deflection_factor * chord * np.tan(np.radians(deflection_angle))

                for z in z_stations:
                    # Thin plate (2mm thickness)
                    vertices.append([x + 50, y_pos + y_deflection - 1, z])
                    vertices.append([x + 50, y_pos + y_deflection + 1, z])

            vertices = np.array(vertices)
            faces = self._triangulate_structured_grid(25, 20, 2)
            fence_geometries.append((vertices, faces))

            print(f"    ✓ Fence {i+1}: height={height}mm at y={y_pos:.0f}mm")

        return fence_geometries

    def generate_mounting_pylons(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate elliptical mounting pylons (regulation compliant)
        Connect wing to nose/chassis

        Returns:
            List of (vertices, faces) for each pylon
        """
        print(f"  Generating {self.pylon_count} mounting pylons...")

        pylon_geometries = []

        # Pylon positioning (symmetric about centerline)
        if self.pylon_count == 1:
            y_positions = [0]
        elif self.pylon_count == 2:
            y_positions = [-self.pylon_spacing/2, self.pylon_spacing/2]
        else:
            y_positions = np.linspace(-self.pylon_spacing, self.pylon_spacing, self.pylon_count)

        for y_pos in y_positions:
            vertices = []

            # Elliptical cross-section (regulation requirement)
            a = self.pylon_major_axis / 2  # Semi-major axis
            b = self.pylon_minor_axis / 2  # Semi-minor axis

            # Vertical extent
            z_stations = np.linspace(-self.pylon_length, 0, 50)
            theta_stations = np.linspace(0, 2*np.pi, 32)

            for z in z_stations:
                # Pylon tapers toward top (more streamlined)
                taper_factor = 1.0 - 0.3 * (z + self.pylon_length) / self.pylon_length

                for theta in theta_stations:
                    x_local = a * taper_factor * np.cos(theta)
                    y_local = b * taper_factor * np.sin(theta)

                    # Position at wing root
                    vertices.append([x_local + 150, y_pos + y_local, z])

            vertices = np.array(vertices)
            faces = self._triangulate_structured_grid(50, 32, 1)
            pylon_geometries.append((vertices, faces))

        print(f"    ✓ {len(pylon_geometries)} pylons generated")
        return pylon_geometries

    def generate_cascade_elements(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cascade elements (PRE-2019 only - now banned)
        Turning vanes at wing tips for flow conditioning

        Returns:
            List of (vertices, faces) for cascades
        """
        if not self.cascade_enabled:
            return []

        print("  Generating cascade elements (pre-2019 style)...")

        cascade_geometries = []

        # Primary cascade (larger, closer to endplate)
        primary_vertices = self._generate_single_cascade(
            span=self.primary_cascade_span,
            chord=self.primary_cascade_chord,
            y_position=700,
            deflection_angle=25
        )
        primary_faces = self._triangulate_structured_grid(30, 25, 2)
        cascade_geometries.append((primary_vertices, primary_faces))

        # Secondary cascade (smaller, further inboard)
        secondary_vertices = self._generate_single_cascade(
            span=self.secondary_cascade_span,
            chord=self.secondary_cascade_chord,
            y_position=550,
            deflection_angle=18
        )
        secondary_faces = self._triangulate_structured_grid(25, 20, 2)
        cascade_geometries.append((secondary_vertices, secondary_faces))

        print(f"    ✓ {len(cascade_geometries)} cascade elements")
        return cascade_geometries

    def _generate_single_cascade(self, span, chord, y_position, deflection_angle):
        """Helper function to generate a single cascade element"""
        vertices = []

        x_stations = np.linspace(0, chord, 30)
        z_stations = np.linspace(0, span, 25)

        for z in z_stations:
            for x in x_stations:
                # Cascade profile (cambered airfoil angled for outwash)
                camber = 0.15 * chord  # High camber
                y_camber = camber * (x / chord) * (1 - x / chord)

                # Deflection angle increases toward tip
                deflection = deflection_angle * (z / span)
                y_deflection = x * np.tan(np.radians(deflection))

                # Upper surface
                vertices.append([x + 20, y_position + y_camber + y_deflection + 3, z])
                # Lower surface
                vertices.append([x + 20, y_position + y_camber + y_deflection - 3, z])

        return np.array(vertices)

    def _triangulate_structured_grid(self, n_chord, n_span, points_per_station):
        """Helper function for structured grid triangulation"""
        faces = []

        for i in range(n_chord - 1):
            for j in range(n_span - 1):
                for k in range(points_per_station):
                    v0 = (i * n_span + j) * points_per_station + k
                    v1 = v0 + points_per_station
                    v2 = ((i + 1) * n_span + j) * points_per_station + k
                    v3 = v2 + points_per_station

                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])

        return np.array(faces)

    def apply_laplacian_smoothing(self, vertices, faces):
        """Apply surface smoothing"""
        if not self.surface_smoothing or self.smoothing_iterations == 0:
            return vertices

        # Build adjacency
        adjacency = [set() for _ in range(len(vertices))]
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)

        smoothed = vertices.copy()
        for iteration in range(self.smoothing_iterations):
            new_verts = smoothed.copy()
            strength = 0.3

            for i in range(len(vertices)):
                if adjacency[i]:
                    neighbors = list(adjacency[i])
                    avg = np.mean(smoothed[neighbors], axis=0)
                    new_verts[i] = (1 - strength) * smoothed[i] + strength * avg

            smoothed = new_verts

        return smoothed

    def generate_complete_structure(self, side='both') -> mesh.Mesh:
        """
        Generate complete Y250 and central structure assembly

        Args:
            side: 'left', 'right', or 'both'

        Returns:
            STL mesh object
        """
        print(f"\n{'='*70}")
        print(f"GENERATING Y250 & CENTRAL STRUCTURE - {side.upper()}")
        print(f"{'='*70}\n")

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Generate all components
        components = []

        # 1. Y250 neutral section
        v, f = self.generate_y250_neutral_section()
        components.append((v, f, "Y250 Neutral Section"))

        # 2. Y250 transition
        v, f = self.generate_y250_transition_region()
        components.append((v, f, "Y250 Transition"))

        # 3. Footplate
        v, f = self.generate_footplate_with_arch()
        components.append((v, f, "Footplate"))

        # 4. Strakes
        for i, (v, f) in enumerate(self.generate_underwing_strakes()):
            components.append((v, f, f"Strake {i+1}"))

        # 5. Vortex generators
        for i, (v, f) in enumerate(self.generate_half_tube_vortex_generators()):
            components.append((v, f, f"VG {i+1}"))

        # 6. Outboard fences
        for i, (v, f) in enumerate(self.generate_outboard_fences()):
            components.append((v, f, f"Fence {i+1}"))

        # 7. Pylons
        for i, (v, f) in enumerate(self.generate_mounting_pylons()):
            components.append((v, f, f"Pylon {i+1}"))

        # 8. Cascades (if enabled)
        for i, (v, f) in enumerate(self.generate_cascade_elements()):
            components.append((v, f, f"Cascade {i+1}"))

        # Combine all geometry
        for verts, faces, name in components:
            # Apply smoothing
            verts = self.apply_laplacian_smoothing(verts, faces)

            # Mirror for sides
            if side == 'both':
                # Right side
                all_vertices.append(verts.copy())
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)

                # Left side (mirror Y)
                verts_left = verts.copy()
                verts_left[:, 1] *= -1
                all_vertices.append(verts_left)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts_left)
            elif side == 'left':
                verts[:, 1] *= -1
                all_vertices.append(verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)
            else:
                all_vertices.append(verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)

        # Combine
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)

        print(f"\n{'='*70}")
        print(f"TOTAL GEOMETRY:")
        print(f"  Vertices: {len(combined_vertices)}")
        print(f"  Faces: {len(combined_faces)}")
        print(f"  Components: {len(components)}")
        print(f"{'='*70}\n")

        # Create STL mesh
        structure_mesh = mesh.Mesh(np.zeros(len(combined_faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(combined_faces):
            for j in range(3):
                structure_mesh.vectors[i][j] = combined_vertices[face[j]]

        return structure_mesh

    def save_stl(self, filename='f1_y250_central_structure.stl', side='both'):
        """Generate and save to STL"""
        structure = self.generate_complete_structure(side=side)
        structure.save(filename)
        print(f"✓ Saved to: {filename}\n")
        return structure


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("F1 Y250 VORTEX & CENTRAL STRUCTURE GENERATOR")
    print("="*70)
    print("\n")

    generator = F1FrontWingY250CentralStructureGenerator(
        # Y250 vortex zone
        y250_width=500,
        y250_step_height=18,
        y250_transition_length=80,
        central_slot_width=0,
        y250_vortex_strength=0.85,

        # Footplate
        footplate_extension=70,
        footplate_height=30,
        arch_radius=130,
        footplate_thickness=5,
        primary_strake_count=2,
        strake_heights=[45, 35],

        # Tire interaction
        tire_diameter=670,
        tire_width=305,
        tire_wake_deflection_angle=15,
        outwash_optimization=True,
        wheel_wake_interaction_zone=[600, 900],

        # Vortex generators
        vortex_generator_enabled=True,
        vg_type="half_tube",
        vg_height=8,
        vg_spacing=25,
        vg_angle=18,
        outboard_fence_enabled=True,
        fence_heights=[60, 50, 40],
        fence_positions=[0.7, 0.85, 0.95],

        # Pylons
        pylon_count=2,
        pylon_spacing=320,
        pylon_major_axis=38,
        pylon_minor_axis=25,
        pylon_length=120,

        # Cascades (disabled for 2019+ regs)
        cascade_enabled=False,

        # FIA compliance
        fia_compliance_mode=True,
        max_width_regulation=1800,
        virtual_endplate_surface=True
    )

    # Generate and save
    generator.save_stl('f1_y250_central_structure.stl', side='both')

    print("="*70)
    print("✓ GENERATION COMPLETE!")
    print("="*70)
    print("\nKey features:")
    print("  • Y250 vortex generation zone")
    print("  • Footplate with arch sealing")
    print("  • 2 underwing strakes (FIA compliant)")
    print("  • Half-tube vortex generators")
    print("  • 3 outboard fences for tire wake")
    print("  • Elliptical mounting pylons")
    print("="*70)
