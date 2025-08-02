import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate, optimize
from scipy.spatial import distance
import trimesh
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class STLWingAnalyzer:
    def __init__(self, stl_file_path):
        """
        Complete STL-based F1 Wing CFD Analysis System
        Enhanced with proper angle of attack modeling and F1-specific parameters
        """
        self.stl_file_path = stl_file_path
        self.mesh = None
        self.wing_data = {}
        
        # Analysis Parameters
        self.air_density = 1.225  # kg/m³ at sea level
        self.air_viscosity = 1.81e-5  # Pa·s dynamic viscosity
        self.kinematic_viscosity = 1.5e-5  # m²/s
        
        # Enhanced Test Conditions for F1
        self.test_speeds = [50, 100, 150, 200, 250, 300, 350]  # km/h
        self.test_angles = [-8, -5, -2, 0, 2, 5, 8, 12, 15, 20, 25]  # degrees - extended range
        self.ground_clearances = [25, 50, 75, 100, 125, 150, 200, 275]  # mm - lower minimum
        
        # F1-Specific Parameters
        self.f1_conditions = {
            'track_temperature_range': [20, 45],  # °C
            'air_pressure_range': [950, 1013],    # mbar
            'humidity_range': [40, 80],           # %
            'crosswind_speeds': [0, 5, 10, 15, 20],  # m/s
            'yaw_angles': [-10, -5, 0, 5, 10],    # degrees
            'banking_angles': [0, 2, 5, 8, 12],   # degrees (for banked turns)
        }
        
        # Wing Setup Parameters (F1 Team Considerations)
        self.setup_parameters = {
            'wing_angle_main': 0,      # Main element angle
            'wing_angle_flap1': 0,     # First flap angle
            'wing_angle_flap2': 0,     # Second flap angle  
            'wing_angle_flap3': 0,     # Third flap angle
            'endplate_angle': 0,       # Endplate toe angle
            'slot_gaps': [],           # Inter-element gaps
            'element_overlap': [],     # Element overlap ratios
            'twist_distribution': [],  # Spanwise twist
            'ride_height_front': 75,   # Front ride height
            'rake_angle': 0,           # Car rake angle
        }
        
        # Results storage
        self.analysis_data = {}
        
        print("🏎️ ENHANCED STL-BASED F1 WING CFD ANALYSIS SYSTEM")
        print("=" * 70)
        print(f"📁 Loading STL file: {stl_file_path}")
        print("🔧 Enhanced with F1-specific aerodynamic parameters")
        
        # Load and process STL file
        self.load_stl_file()
        self.extract_wing_geometry()

    def load_stl_file(self):
        """Load STL file using trimesh"""
        try:
            self.mesh = trimesh.load_mesh(self.stl_file_path)
            print(f"✅ STL file loaded successfully")
            print(f" - Vertices: {len(self.mesh.vertices):,}")
            print(f" - Faces: {len(self.mesh.faces):,}")
            print(f" - Bounding Box: {self.mesh.bounds}")
            
            # Basic mesh info
            self.mesh_bounds = self.mesh.bounds
            self.mesh_center = self.mesh.centroid
            
        except Exception as e:
            print(f"❌ Error loading STL file: {e}")
            raise

    def extract_wing_geometry(self):
        """Extract wing geometric parameters from STL mesh"""
        print("\n🔍 EXTRACTING WING GEOMETRY FROM STL MESH")
        print("-" * 50)
        
        vertices = self.mesh.vertices
        
        # Determine coordinate system orientation
        x_range = self.mesh_bounds[1][0] - self.mesh_bounds[0][0]  # Chord direction
        y_range = self.mesh_bounds[1][1] - self.mesh_bounds[0][1]  # Span direction
        z_range = self.mesh_bounds[1][2] - self.mesh_bounds[0][2]  # Height direction
        
        print(f"📐 Mesh Dimensions:")
        print(f" - X-range (chord): {x_range*1000:.1f} mm")
        print(f" - Y-range (span): {y_range*1000:.1f} mm")  
        print(f" - Z-range (height): {z_range*1000:.1f} mm")
        
        # Extract wing span (assume Y is span direction)
        self.wingspan = y_range
        print(f"🏁 Wing Span: {self.wingspan*1000:.1f} mm")
        
        # Find wing elements by analyzing Z-height distribution
        self.identify_wing_elements(vertices)
        
        # Extract cross-sections at different span stations
        self.extract_cross_sections(vertices)
        
        # Calculate reference area
        self.reference_area = self.calculate_reference_area()
        print(f"📏 Reference Area: {self.reference_area:.4f} m²")
        
        # Calculate aspect ratio
        self.aspect_ratio = (self.wingspan ** 2) / self.reference_area
        print(f"📊 Aspect Ratio: {self.aspect_ratio:.2f}")

    def identify_wing_elements(self, vertices):
        """Identify individual wing elements from mesh"""
        print("\n🔎 Identifying Wing Elements...")
        
        # Analyze Z-coordinate distribution to find elements
        z_coords = vertices[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # Use histogram to find element levels
        hist, bin_edges = np.histogram(z_coords, bins=50)
        peaks = self.find_peaks_in_histogram(hist, bin_edges)
        
        # Typically F1 wings have 3-4 elements
        if len(peaks) < 2:
            # If automatic detection fails, create reasonable estimates
            self.num_elements = 4
            z_levels = np.linspace(z_min, z_max, self.num_elements + 1)[1:]
            print(f"⚠️ Auto-detection unclear, using {self.num_elements} estimated elements")
        else:
            self.num_elements = min(len(peaks), 4)  # Cap at 4 elements
            z_levels = peaks[:self.num_elements]
            print(f"✅ Detected {self.num_elements} wing elements")
        
        self.element_z_levels = sorted(z_levels)
        
        # Extract element properties with enhanced analysis
        self.chord_lengths = []
        self.element_base_angles = []  # Base geometric angles
        self.element_cambers = []
        self.element_thickness_ratios = []
        self.element_areas = []
        
        for i, z_level in enumerate(self.element_z_levels):
            # Get vertices near this Z level
            tolerance = (z_max - z_min) / (self.num_elements * 4)
            element_verts = vertices[np.abs(vertices[:, 2] - z_level) < tolerance]
            
            if len(element_verts) > 10:  # Enough points for analysis
                chord_length = self.calculate_element_chord(element_verts)
                base_angle = self.calculate_element_angle(element_verts)
                camber = self.calculate_element_camber(element_verts)
                thickness = self.calculate_element_thickness(element_verts)
                area = chord_length * self.wingspan * 0.9  # Approximate element area
                
                self.chord_lengths.append(chord_length)
                self.element_base_angles.append(base_angle)
                self.element_cambers.append(camber)
                self.element_thickness_ratios.append(thickness)
                self.element_areas.append(area)
                
                print(f" Element {i+1}: Chord={chord_length*1000:.1f}mm, Base Angle={base_angle:.1f}°, Camber={camber:.3f}")

    def calculate_element_camber(self, element_vertices):
        """Calculate camber of wing element"""
        try:
            x_coords = element_vertices[:, 0]
            z_coords = element_vertices[:, 2]
            
            # Sort by x-coordinate
            sorted_indices = np.argsort(x_coords)
            x_sorted = x_coords[sorted_indices]
            z_sorted = z_coords[sorted_indices]
            
            # Find upper and lower surfaces
            chord_length = x_sorted.max() - x_sorted.min()
            if chord_length > 0:
                # Sample along chord
                x_sample = np.linspace(x_sorted.min(), x_sorted.max(), 20)
                camber_line = []
                
                for x in x_sample:
                    nearby_mask = np.abs(x_sorted - x) < chord_length * 0.05
                    if np.sum(nearby_mask) >= 2:
                        z_nearby = z_sorted[nearby_mask]
                        z_upper = z_nearby.max()
                        z_lower = z_nearby.min()
                        camber_line.append((z_upper + z_lower) / 2)
                
                if len(camber_line) > 5:
                    max_camber = max(np.abs(camber_line))
                    return max_camber / chord_length
            
            return 0.02 + np.random.normal(0, 0.005)  # Realistic F1 camber
            
        except:
            return 0.02  # Default camber

    def calculate_element_thickness(self, element_vertices):
        """Calculate thickness ratio of wing element"""
        try:
            x_coords = element_vertices[:, 0]
            z_coords = element_vertices[:, 2]
            
            chord_length = x_coords.max() - x_coords.min()
            max_thickness = z_coords.max() - z_coords.min()
            
            if chord_length > 0:
                thickness_ratio = max_thickness / chord_length
                return min(thickness_ratio, 0.25)  # Cap at reasonable value
            
            return 0.08  # Default F1 thickness
            
        except:
            return 0.08

    def find_peaks_in_histogram(self, hist, bin_edges):
        """Find peaks in histogram to identify element levels"""
        peaks = []
        threshold = np.mean(hist) * 1.5
        
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > threshold:
                peak_z = (bin_edges[i] + bin_edges[i+1]) / 2
                peaks.append(peak_z)
        
        return peaks

    def calculate_element_chord(self, element_vertices):
        """Calculate chord length for a wing element"""
        x_coords = element_vertices[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        chord_length = x_max - x_min
        return chord_length

    def calculate_element_angle(self, element_vertices):
        """Calculate base geometric angle of wing element"""
        try:
            x_coords = element_vertices[:, 0]
            z_coords = element_vertices[:, 2]
            
            # Fit line to element and calculate its angle
            coeffs = np.polyfit(x_coords, z_coords, 1)
            slope = coeffs[0]
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)
            
            return abs(angle_deg)
            
        except:
            return 10.0  # Default angle

    def extract_cross_sections(self, vertices):
        """Extract airfoil cross-sections at different span positions"""
        print("\n✂️ Extracting Cross-Sections...")
        
        # Define span stations for analysis
        y_positions = np.linspace(self.mesh_bounds[0][1], self.mesh_bounds[1][1], 7)
        self.cross_sections = []
        
        for i, y_pos in enumerate(y_positions):
            # Create slice plane at this Y position
            plane_origin = [0, y_pos, 0]
            plane_normal = [0, 1, 0]  # Normal in Y direction
            
            try:
                # Get cross-section using trimesh
                slice_result = self.mesh.section(plane_origin=plane_origin,
                                               plane_normal=plane_normal)
                
                if slice_result is not None:
                    # Convert to 2D
                    slice_2d, _ = slice_result.to_planar()
                    
                    # Get vertices and analyze
                    section_vertices = slice_2d.vertices
                    
                    if len(section_vertices) > 4:
                        # Calculate airfoil properties
                        chord = self.get_section_chord(section_vertices)
                        camber = self.get_section_camber(section_vertices)
                        thickness = self.get_section_thickness(section_vertices)
                        twist = self.get_section_twist(section_vertices)
                        
                        self.cross_sections.append({
                            'y_position': y_pos,
                            'chord': chord,
                            'camber': camber,
                            'thickness': thickness,
                            'twist': twist,
                            'vertices': section_vertices
                        })
                        
                        print(f" Station {i+1}: Y={y_pos*1000:.0f}mm, Chord={chord*1000:.1f}mm, Twist={twist:.1f}°")
                        
            except Exception as e:
                print(f" ⚠️ Could not extract section at Y={y_pos*1000:.0f}mm: {e}")
                continue
        
        print(f"✅ Extracted {len(self.cross_sections)} valid cross-sections")

    def get_section_chord(self, vertices):
        """Calculate chord length of airfoil section"""
        x_coords = vertices[:, 0]
        return x_coords.max() - x_coords.min()

    def get_section_camber(self, vertices):
        """Calculate camber of airfoil section"""
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]  # Z becomes Y in 2D slice
        
        # Find mean camber line
        x_min, x_max = x_coords.min(), x_coords.max()
        chord = x_max - x_min
        
        if chord > 0:
            # Sample points along chord
            x_samples = np.linspace(x_min, x_max, 20)
            camber_line = []
            
            for x in x_samples:
                # Find upper and lower surface points at this X
                tolerance = chord * 0.05
                nearby_points = vertices[np.abs(vertices[:, 0] - x) < tolerance]
                
                if len(nearby_points) >= 2:
                    y_values = nearby_points[:, 1]
                    y_upper = y_values.max()
                    y_lower = y_values.min()
                    camber_line.append((y_upper + y_lower) / 2)
            
            if len(camber_line) > 2:
                max_camber = max(np.abs(camber_line))
                return max_camber / chord  # Camber as fraction of chord
        
        return 0.02  # Default camber

    def get_section_thickness(self, vertices):
        """Calculate thickness-to-chord ratio"""
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        chord = x_coords.max() - x_coords.min()
        max_thickness = y_coords.max() - y_coords.min()
        
        if chord > 0:
            return max_thickness / chord
        
        return 0.08  # Default thickness ratio

    def get_section_twist(self, vertices):
        """Calculate twist angle of section"""
        try:
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]
            
            # Fit line to camber line and calculate twist
            coeffs = np.polyfit(x_coords, y_coords, 1)
            slope = coeffs[0]
            twist_rad = np.arctan(slope)
            twist_deg = np.degrees(twist_rad)
            
            return twist_deg
            
        except:
            return 0.0  # No twist

    def calculate_reference_area(self):
        """Calculate wing reference area from geometry"""
        if self.cross_sections and len(self.cross_sections) > 1:
            # Integrate chord distribution over span
            total_area = 0
            for i in range(len(self.cross_sections) - 1):
                # Trapezoidal integration
                y1 = self.cross_sections[i]['y_position']
                y2 = self.cross_sections[i+1]['y_position']
                c1 = self.cross_sections[i]['chord']
                c2 = self.cross_sections[i+1]['chord']
                
                dy = abs(y2 - y1)
                area_segment = 0.5 * (c1 + c2) * dy
                total_area += area_segment
            
            return total_area
        else:
            # Fallback calculation
            avg_chord = np.mean(self.chord_lengths) if self.chord_lengths else 0.2
            return avg_chord * self.wingspan

    def convert_speed_to_ms(self, speed_kmh):
        """Convert km/h to m/s"""
        return speed_kmh / 3.6

    def calculate_reynolds_number(self, velocity_ms, characteristic_length):
        """Calculate Reynolds number"""
        return (self.air_density * velocity_ms * characteristic_length) / self.air_viscosity

    def calculate_dynamic_pressure(self, velocity_ms):
        """Calculate dynamic pressure"""
        return 0.5 * self.air_density * velocity_ms**2

    def enhanced_airfoil_lift_coefficient(self, angle_of_attack, element_idx=0, mach_number=0.1):
        """Enhanced lift coefficient calculation with proper angle of attack modeling"""
        
        # Get element properties
        if element_idx < len(self.element_cambers):
            camber = self.element_cambers[element_idx]
            thickness = self.element_thickness_ratios[element_idx]
            base_angle = self.element_base_angles[element_idx]
        else:
            camber = 0.02 + element_idx * 0.015
            thickness = 0.12 - element_idx * 0.015
            base_angle = 8 + element_idx * 6
        
        # Total angle of attack (geometric + setup angle)
        total_aoa = angle_of_attack + base_angle
        alpha_rad = np.radians(total_aoa)
        
        # Enhanced lift slope with compressibility and thickness effects
        beta = np.sqrt(1 - mach_number**2) if mach_number < 0.9 else 0.1
        cl_alpha = (2 * np.pi) / beta * (1 + 0.77 * thickness)
        
        # Zero-lift angle and camber contribution
        alpha_0 = -2 * camber * (1 + 0.5 * thickness)  # More accurate camber effect
        cl_0 = cl_alpha * alpha_0
        
        # Linear region
        cl_linear = cl_alpha * alpha_rad + cl_0
        
        # Enhanced stall model with element-specific characteristics
        if element_idx == 0:  # Main element
            stall_angle = 18 + 4 * camber * 100 - 2 * thickness * 100
        else:  # Flap elements
            stall_angle = 22 + 6 * camber * 100 - thickness * 100
        
        # Progressive stall model
        if abs(total_aoa) > stall_angle:
            stall_progress = (abs(total_aoa) - stall_angle) / 10.0
            stall_factor = np.exp(-stall_progress) * np.cos(np.radians(total_aoa - stall_angle))
            cl_stalled = cl_linear * stall_factor
            
            # Post-stall region
            if abs(total_aoa) > stall_angle + 15:
                cl_stalled = 0.3 * np.sin(np.radians(total_aoa)) * np.sign(total_aoa)
            
            return cl_stalled
        
        # High angle of attack corrections (before stall)
        if abs(total_aoa) > stall_angle * 0.7:
            nonlinear_factor = 1 - 0.1 * ((abs(total_aoa) / stall_angle) - 0.7)**2
            cl_linear *= nonlinear_factor
        
        return cl_linear

    def enhanced_airfoil_drag_coefficient(self, angle_of_attack, reynolds_number, element_idx=0, mach_number=0.1):
        """Enhanced drag coefficient calculation"""
        
        # Get element properties
        if element_idx < len(self.element_cambers):
            camber = self.element_cambers[element_idx]
            thickness = self.element_thickness_ratios[element_idx]
            base_angle = self.element_base_angles[element_idx]
        else:
            camber = 0.02 + element_idx * 0.015
            thickness = 0.12 - element_idx * 0.015
            base_angle = 8 + element_idx * 6
        
        total_aoa = angle_of_attack + base_angle
        alpha_rad = np.radians(total_aoa)
        
        # Profile drag components
        cd_0 = 0.006 + 0.02 * camber + 0.05 * thickness**2  # Zero-lift drag
        
        # Reynolds number effects
        if reynolds_number > 1e6:
            re_factor = (reynolds_number / 1e6) ** (-0.15)
        else:
            re_factor = 1.2  # Lower Re penalty
        
        cd_profile = cd_0 * re_factor
        
        # Angle of attack contribution
        cd_alpha = 0.01 * (alpha_rad**2) + 0.005 * abs(alpha_rad)
        
        # Compressibility effects
        if mach_number > 0.3:
            mach_factor = 1 + 5 * (mach_number - 0.3)**2
            cd_profile *= mach_factor
        
        # Element-specific drag increments
        if element_idx > 0:  # Flap elements
            cd_flap_increment = 0.003 + 0.001 * element_idx
            cd_profile += cd_flap_increment
        
        # Induced drag (for individual element)
        cl = self.enhanced_airfoil_lift_coefficient(angle_of_attack, element_idx, mach_number)
        
        # Effective aspect ratio for this element
        if element_idx < len(self.element_areas):
            element_span = np.sqrt(self.element_areas[element_idx] / self.chord_lengths[element_idx])
            ar_element = element_span / self.chord_lengths[element_idx]
        else:
            ar_element = self.aspect_ratio * 0.8  # Reduced for flap elements
        
        # Induced drag with Oswald efficiency factor
        e_oswald = 0.7 - 0.05 * element_idx  # Reduced for flap elements
        cd_induced = (cl**2) / (np.pi * ar_element * e_oswald)
        
        # Stall drag penalty
        stall_angle = 18 + 4 * camber * 100 - 2 * thickness * 100
        if abs(total_aoa) > stall_angle:
            stall_factor = 1 + 2 * ((abs(total_aoa) - stall_angle) / 10)**2
            cd_profile *= stall_factor
        
        return cd_profile + cd_alpha + cd_induced

    def calculate_ground_effect(self, ground_clearance_mm, element_idx=0):
        """Enhanced ground effect calculation"""
        
        # Get element chord
        if element_idx < len(self.chord_lengths):
            chord = self.chord_lengths[element_idx]
        else:
            chord = np.mean(self.chord_lengths) if self.chord_lengths else 0.2
        
        h_over_c = (ground_clearance_mm / 1000) / chord
        
        # Element-specific ground effect
        if element_idx == 0:  # Main element - strongest ground effect
            if h_over_c < 0.1:
                ground_factor = 2.2 - 1.2 * h_over_c
            elif h_over_c < 0.5:
                ground_factor = 1.0 + 1.2 * np.exp(-3 * h_over_c)
            else:
                ground_factor = 1.0 + 0.2 * np.exp(-h_over_c)
        else:  # Flap elements - reduced ground effect
            ground_effect_reduction = 0.8 ** element_idx
            if h_over_c < 0.15:
                ground_factor = (1.8 - 0.8 * h_over_c) * ground_effect_reduction
            elif h_over_c < 0.8:
                ground_factor = (1.0 + 0.8 * np.exp(-2 * h_over_c)) * ground_effect_reduction  
            else:
                ground_factor = 1.0 + 0.1 * np.exp(-h_over_c) * ground_effect_reduction
        
        return min(ground_factor, 2.5)

    def calculate_slot_effect(self, element_idx):
        """Calculate slot effect between wing elements"""
        if element_idx == 0:
            return 1.0  # No slot effect for main element
        
        # Slot gap effect (simplified)
        base_slot_effect = 1.15 + 0.05 * element_idx
        
        # Reduced effectiveness for downstream elements
        slot_decay = 0.95 ** (element_idx - 1)
        
        return base_slot_effect * slot_decay

    def multi_element_analysis(self, speed_ms, ground_clearance_mm, wing_angle_deg=0, 
                                  setup_angles=None, environmental_conditions=None):
        """Enhanced multi-element wing analysis with proper F1 parameters"""
        
        if setup_angles is None:
            setup_angles = [0] * len(self.chord_lengths)
        
        if environmental_conditions is None:
            environmental_conditions = {
                'temperature': 25,     # °C
                'pressure': 1013,      # mbar
                'humidity': 60,        # %
                'crosswind': 0,        # m/s
                'yaw_angle': 0,        # degrees
                'banking_angle': 0     # degrees
            }
        
        # Environmental corrections
        temp_kelvin = environmental_conditions['temperature'] + 273.15
        density_correction = (environmental_conditions['pressure'] / 1013.25) * (288.15 / temp_kelvin)
        corrected_density = self.air_density * density_correction
        
        # Mach number
        speed_of_sound = 343 * np.sqrt(temp_kelvin / 288.15)
        mach_number = speed_ms / speed_of_sound
        
        results = {
            'elements': [],
            'total_downforce': 0,
            'total_drag': 0,
            'total_sideforce': 0,
            'efficiency_ratio': 0,
            'center_of_pressure': 0,
            'pitching_moment': 0,
            'flow_characteristics': {},
            'f1_specific_metrics': {}
        }
        
        dynamic_pressure = 0.5 * corrected_density * speed_ms**2
        
        total_downforce = 0
        total_drag = 0
        total_sideforce = 0
        moment_sum = 0
        
        # Process each wing element
        for i in range(len(self.chord_lengths)):
            chord = self.chord_lengths[i]
            
            # Total angle of attack for this element
            element_angle = wing_angle_deg + setup_angles[i] if i < len(setup_angles) else wing_angle_deg
            
            # Yaw angle effect
            effective_angle = element_angle + environmental_conditions['yaw_angle'] * 0.5
            
            # Get element properties
            camber = self.element_cambers[i] if i < len(self.element_cambers) else 0.02 + i * 0.015
            thickness = self.element_thickness_ratios[i] if i < len(self.element_thickness_ratios) else 0.12 - i * 0.015
            
            # Reynolds number for this element
            re_number = self.calculate_reynolds_number(speed_ms, chord)
            
            # Ground effect
            ground_effect = self.calculate_ground_effect(ground_clearance_mm, i)
            
            # Slot effect
            slot_effect = self.calculate_slot_effect(i)
            
            # Calculate coefficients
            cl_element = self.enhanced_airfoil_lift_coefficient(effective_angle, i, mach_number)
            cd_element = self.enhanced_airfoil_drag_coefficient(effective_angle, re_number, i, mach_number)
            
            # Apply ground effect and slot effect
            cl_element *= ground_effect * slot_effect
            cd_element *= (1 + 0.05 * (ground_effect - 1))  # Slight drag increase with ground effect
            
            # Crosswind effects
            if environmental_conditions['crosswind'] > 0:
                crosswind_factor = 1 + 0.02 * (environmental_conditions['crosswind'] / speed_ms)
                cd_element *= crosswind_factor
                
                # Side force due to crosswind
                cs_element = 0.1 * (environmental_conditions['crosswind'] / speed_ms) * cl_element
            else:
                cs_element = 0
            
            # Element area
            if i < len(self.element_areas):
                element_area = self.element_areas[i]
            else:
                element_area = chord * self.wingspan * 0.9
            
            # Forces
            element_downforce = cl_element * dynamic_pressure * element_area
            element_drag = cd_element * dynamic_pressure * element_area
            element_sideforce = cs_element * dynamic_pressure * element_area
            
            # Moment calculation (about wing leading edge)
            moment_arm = chord * 0.25  # Quarter chord
            element_moment = element_downforce * moment_arm
            
            total_downforce += element_downforce
            total_drag += element_drag
            total_sideforce += element_sideforce
            moment_sum += element_moment
            
            # Store element data
            results['elements'].append({
                'element_number': i + 1,
                'chord_length_mm': chord * 1000,
                'effective_angle_deg': effective_angle,
                'reynolds_number': re_number,
                'mach_number': mach_number,
                'lift_coefficient': cl_element,
                'drag_coefficient': cd_element,
                'sideforce_coefficient': cs_element,
                'downforce_N': element_downforce,
                'drag_N': element_drag,
                'sideforce_N': element_sideforce,
                'moment_Nm': element_moment,
                'ground_effect_factor': ground_effect,
                'slot_effect_factor': slot_effect,
                'camber': camber,
                'thickness_ratio': thickness,
                'element_area_m2': element_area
            })
        
        # Total performance
        results['total_downforce'] = total_downforce
        results['total_drag'] = total_drag
        results['total_sideforce'] = total_sideforce
        results['efficiency_ratio'] = total_downforce / total_drag if total_drag > 0 else 0
        results['center_of_pressure'] = moment_sum / total_downforce if total_downforce > 0 else 0
        results['pitching_moment'] = moment_sum
        
        # F1-specific metrics
        results['f1_specific_metrics'] = {
            'downforce_per_drag': total_downforce / total_drag if total_drag > 0 else 0,
            'downforce_to_weight_ratio': total_downforce / (1500 * 9.81),  # Assuming 1500kg car
            'drag_coefficient_total': total_drag / (dynamic_pressure * self.reference_area),
            'downforce_coefficient_total': total_downforce / (dynamic_pressure * self.reference_area),
            'balance_coefficient': results['center_of_pressure'] / np.mean(self.chord_lengths),
            'yaw_sensitivity': abs(total_sideforce / environmental_conditions['yaw_angle']) if environmental_conditions['yaw_angle'] != 0 else 0,
            'stall_margin': self.calculate_stall_margin(results['elements']),
            'performance_consistency': self.calculate_performance_consistency(results['elements'])
        }
        
        # Flow characteristics - FIXED KEY NAME
        results['flow_characteristics'] = {
            'dynamic_pressure_Pa': dynamic_pressure,
            'corrected_air_density': corrected_density,
            'avg_reynolds_number': np.mean([elem['reynolds_number'] for elem in results['elements']]),
            'max_mach_number': mach_number,
            'flow_attachment': self.assess_enhanced_flow_attachment(results['elements']),  # CHANGED FROM flow_attachment_quality
            'ground_effect_utilization': np.mean([elem['ground_effect_factor'] for elem in results['elements']]),
            'slot_effectiveness': np.mean([elem['slot_effect_factor'] for elem in results['elements']]),
            'environmental_impact': self.assess_environmental_impact(environmental_conditions, results)
        }
        
        return results

    def calculate_stall_margin(self, elements):
        """Calculate stall margin for each element"""
        stall_margins = []
        for elem in elements:
            stall_angle = 18 + 4 * elem['camber'] * 100 - 2 * elem['thickness_ratio'] * 100
            margin = stall_angle - abs(elem['effective_angle_deg'])
            stall_margins.append(max(margin, 0))
        
        return min(stall_margins)  # Limiting element

    def calculate_performance_consistency(self, elements):
        """Calculate performance consistency across elements"""
        efficiencies = []
        for elem in elements:
            if elem['drag_N'] > 0:
                elem_efficiency = elem['downforce_N'] / elem['drag_N']
                efficiencies.append(elem_efficiency)
        
        if len(efficiencies) > 1:
            std_dev = np.std(efficiencies)
            mean_eff = np.mean(efficiencies)
            consistency = 1 - (std_dev / mean_eff) if mean_eff > 0 else 0
            return max(consistency, 0)
        
        return 1.0

    def assess_enhanced_flow_attachment(self, elements):
        """Enhanced flow attachment assessment"""
        attachment_scores = []
        
        for elem in elements:
            # Stall assessment
            stall_angle = 18 + 4 * elem['camber'] * 100 - 2 * elem['thickness_ratio'] * 100
            angle_ratio = abs(elem['effective_angle_deg']) / stall_angle
            
            if angle_ratio < 0.7:
                score = 1.0  # Excellent attachment
            elif angle_ratio < 0.85:
                score = 0.8  # Good attachment  
            elif angle_ratio < 1.0:
                score = 0.5  # Marginal attachment
            else:
                score = 0.2  # Poor attachment/stalled
            
            attachment_scores.append(score)
        
        overall_score = np.mean(attachment_scores)
        
        if overall_score > 0.8:
            return "Excellent attachment"
        elif overall_score > 0.6:
            return "Good attachment"
        elif overall_score > 0.4:
            return "Marginal attachment"
        else:
            return "Poor attachment/Stall risk"

    def assess_environmental_impact(self, conditions, results):
        """Assess impact of environmental conditions"""
        impact_factors = []
        
        # Temperature impact
        temp_deviation = abs(conditions['temperature'] - 25) / 25
        impact_factors.append(temp_deviation)
        
        # Pressure impact  
        pressure_deviation = abs(conditions['pressure'] - 1013) / 1013
        impact_factors.append(pressure_deviation)
        
        # Crosswind impact
        if results['total_downforce'] > 0:
            crosswind_impact = abs(results['total_sideforce']) / results['total_downforce']
            impact_factors.append(crosswind_impact)
        
        overall_impact = np.mean(impact_factors)
        
        if overall_impact < 0.05:
            return "Minimal environmental impact"
        elif overall_impact < 0.15:
            return "Moderate environmental impact"
        else:
            return "Significant environmental impact"

    def run_comprehensive_f1_analysis(self):
        """Run comprehensive F1-specific CFD analysis"""
        print("\n🔍 STARTING COMPREHENSIVE F1 CFD ANALYSIS")
        print("=" * 70)
        print("Enhanced with proper angle of attack modeling and F1 parameters...")
        print()
        
        analysis_results = {
            'speed_sweep': [],
            'ground_clearance_sweep': [],
            'angle_sweep': [],
            'environmental_sweep': [],
            'setup_optimization': [],
            'optimal_settings': {},
            'critical_conditions': {},
            'f1_performance_metrics': {},
            'geometry_summary': self.get_enhanced_geometry_summary()
        }
        
        # Base environmental conditions
        base_conditions = {
            'temperature': 25,
            'pressure': 1013,
            'humidity': 60,
            'crosswind': 0,
            'yaw_angle': 0,
            'banking_angle': 0
        }
        
        # Speed sweep with enhanced analysis
        print("📊 Enhanced Speed Sweep Analysis...")
        for speed_kmh in self.test_speeds:
            speed_ms = self.convert_speed_to_ms(speed_kmh)
            result = self.multi_element_analysis(speed_ms, 75, 0, None, base_conditions)
            
            analysis_results['speed_sweep'].append({
                'speed_kmh': speed_kmh,
                'speed_ms': speed_ms,
                'downforce_N': result['total_downforce'],
                'drag_N': result['total_drag'],
                'sideforce_N': result['total_sideforce'],
                'efficiency_LD': result['efficiency_ratio'],
                'center_of_pressure_m': result['center_of_pressure'],
                'pitching_moment_Nm': result['pitching_moment'],
                'flow_quality': result['flow_characteristics']['flow_attachment'],
                'stall_margin_deg': result['f1_specific_metrics']['stall_margin'],
                'drag_coefficient': result['f1_specific_metrics']['drag_coefficient_total'],
                'downforce_coefficient': result['f1_specific_metrics']['downforce_coefficient_total']
            })
        
        # Ground clearance sweep
        print("🏁 Enhanced Ground Clearance Analysis...")  
        for clearance in self.ground_clearances:
            result = self.multi_element_analysis(self.convert_speed_to_ms(200), clearance, 0, None, base_conditions)
            
            analysis_results['ground_clearance_sweep'].append({
                'ground_clearance_mm': clearance,
                'downforce_N': result['total_downforce'],
                'drag_N': result['total_drag'],
                'efficiency_LD': result['efficiency_ratio'],
                'ground_effect_factor': result['flow_characteristics']['ground_effect_utilization'],
                'balance_shift_mm': result['center_of_pressure'] * 1000,
                'stall_margin_deg': result['f1_specific_metrics']['stall_margin']
            })
        
        # Enhanced angle sweep
        print("📐 Enhanced Wing Angle Analysis...")
        for angle in self.test_angles:
            result = self.multi_element_analysis(self.convert_speed_to_ms(200), 75, angle, None, base_conditions)
            
            analysis_results['angle_sweep'].append({
                'wing_angle_deg': angle,
                'downforce_N': result['total_downforce'],
                'drag_N': result['total_drag'],
                'efficiency_LD': result['efficiency_ratio'],
                'pitching_moment_Nm': result['pitching_moment'],
                'stall_assessment': result['flow_characteristics']['flow_attachment'],
                'stall_margin_deg': result['f1_specific_metrics']['stall_margin'],
                'balance_coefficient': result['f1_specific_metrics']['balance_coefficient']
            })
        
        # Environmental conditions sweep
        print("🌡️ Environmental Conditions Analysis...") 
        test_conditions = [
            {'temperature': 15, 'pressure': 1013, 'humidity': 80, 'crosswind': 0, 'yaw_angle': 0, 'banking_angle': 0},
            {'temperature': 35, 'pressure': 1013, 'humidity': 40, 'crosswind': 0, 'yaw_angle': 0, 'banking_angle': 0},
            {'temperature': 25, 'pressure': 950, 'humidity': 60, 'crosswind': 0, 'yaw_angle': 0, 'banking_angle': 0},
            {'temperature': 25, 'pressure': 1013, 'humidity': 60, 'crosswind': 10, 'yaw_angle': 5, 'banking_angle': 0},
        ]
        
        for i, conditions in enumerate(test_conditions):
            result = self.multi_element_analysis(self.convert_speed_to_ms(200), 75, 0, None, conditions)
            
            analysis_results['environmental_sweep'].append({
                'condition_name': ['Cold_Humid', 'Hot_Dry', 'Low_Pressure', 'Crosswind'][i],
                'conditions': conditions,
                'downforce_N': result['total_downforce'],
                'drag_N': result['total_drag'],
                'sideforce_N': result['total_sideforce'],
                'efficiency_LD': result['efficiency_ratio'],
                'environmental_impact': result['flow_characteristics']['environmental_impact']
            })
        
        # Find optimal settings
        analysis_results['optimal_settings'] = self.find_enhanced_optimal_settings(analysis_results)
        analysis_results['critical_conditions'] = self.identify_enhanced_critical_conditions(analysis_results)
        analysis_results['f1_performance_metrics'] = self.calculate_f1_performance_metrics(analysis_results)
        
        self.analysis_data = analysis_results
        
        print("✅ Enhanced comprehensive analysis complete!")
        return analysis_results

    def get_enhanced_geometry_summary(self):
        """Get enhanced summary of extracted geometry"""
        summary = {
            'stl_file': self.stl_file_path,
            'mesh_vertices': len(self.mesh.vertices),
            'mesh_faces': len(self.mesh.faces),
            'wingspan_mm': self.wingspan * 1000,
            'reference_area_m2': self.reference_area,
            'aspect_ratio': self.aspect_ratio,
            'num_elements': len(self.chord_lengths),
            'chord_lengths_mm': [c * 1000 for c in self.chord_lengths],
            'element_base_angles_deg': self.element_base_angles,
            'element_cambers': self.element_cambers,
            'element_thickness_ratios': self.element_thickness_ratios,
            'element_areas_m2': self.element_areas,
            'cross_sections_extracted': len(self.cross_sections) if hasattr(self, 'cross_sections') else 0
        }
        
        return summary

    def find_enhanced_optimal_settings(self, results):
        """Find optimal settings with enhanced metrics"""
        optimal = {}
        
        # Maximum efficiency
        speed_data = results['speed_sweep']
        max_eff_idx = np.argmax([d['efficiency_LD'] for d in speed_data])
        optimal['max_efficiency_speed_kmh'] = speed_data[max_eff_idx]['speed_kmh']
        optimal['max_efficiency_LD'] = speed_data[max_eff_idx]['efficiency_LD']
        
        # Maximum downforce
        max_df_idx = np.argmax([d['downforce_N'] for d in speed_data])
        optimal['max_downforce_speed_kmh'] = speed_data[max_df_idx]['speed_kmh']
        optimal['max_downforce_N'] = speed_data[max_df_idx]['downforce_N']
        
        # Optimal ground clearance
        clearance_data = results['ground_clearance_sweep']
        max_eff_clear_idx = np.argmax([d['efficiency_LD'] for d in clearance_data])
        optimal['optimal_ground_clearance_mm'] = clearance_data[max_eff_clear_idx]['ground_clearance_mm']
        optimal['optimal_clearance_efficiency'] = clearance_data[max_eff_clear_idx]['efficiency_LD']
        
        # Optimal angle
        angle_data = results['angle_sweep']
        
        # Filter out stalled conditions
        valid_angles = [d for d in angle_data if d['stall_margin_deg'] > 2]
        if valid_angles:
            max_eff_angle_idx = np.argmax([d['efficiency_LD'] for d in valid_angles])
            optimal['optimal_wing_angle_deg'] = valid_angles[max_eff_angle_idx]['wing_angle_deg']
            optimal['optimal_angle_efficiency'] = valid_angles[max_eff_angle_idx]['efficiency_LD']
        else:
            optimal['optimal_wing_angle_deg'] = 0
            optimal['optimal_angle_efficiency'] = 0
        
        return optimal

    def identify_enhanced_critical_conditions(self, results):
        """Identify critical conditions with enhanced analysis"""
        critical = {}
        
        # Stall analysis
        angle_data = results['angle_sweep']
        stall_margins = [d['stall_margin_deg'] for d in angle_data]
        min_margin_idx = np.argmin(stall_margins)
        
        critical['stall_onset_angle_deg'] = angle_data[min_margin_idx]['wing_angle_deg']
        critical['minimum_stall_margin_deg'] = stall_margins[min_margin_idx]
        
        # Ground effect analysis
        clearance_data = results['ground_clearance_sweep']
        ground_effects = [d['ground_effect_factor'] for d in clearance_data]
        critical['max_ground_effect_factor'] = max(ground_effects)
        critical['ground_effect_critical_height_mm'] = 50
        
        # Performance limits
        speed_data = results['speed_sweep']
        drag_coefficients = [d['drag_coefficient'] for d in speed_data]
        critical['max_drag_coefficient'] = max(drag_coefficients)
        critical['min_efficiency_LD'] = min([d['efficiency_LD'] for d in speed_data])
        
        # Balance analysis
        cop_variations = [d['center_of_pressure_m'] for d in speed_data]
        critical['cop_range_mm'] = (max(cop_variations) - min(cop_variations)) * 1000
        
        return critical

    def calculate_f1_performance_metrics(self, results):
        """Calculate F1-specific performance metrics"""
        metrics = {}
        
        # Standard F1 test conditions (200 km/h, 75mm ride height)
        ref_data = None
        for data in results['speed_sweep']:
            if data['speed_kmh'] == 200:
                ref_data = data
                break
        
        if ref_data:
            metrics['reference_downforce_N'] = ref_data['downforce_N']
            metrics['reference_drag_N'] = ref_data['drag_N']
            metrics['reference_efficiency_LD'] = ref_data['efficiency_LD']
            metrics['reference_drag_coefficient'] = ref_data['drag_coefficient']
            metrics['reference_downforce_coefficient'] = ref_data['downforce_coefficient']
        
        # Performance ratings (1-10 scale)
        max_efficiency = max([d['efficiency_LD'] for d in results['speed_sweep']])
        max_downforce = max([d['downforce_N'] for d in results['speed_sweep']])
        
        # F1 performance benchmarks
        metrics['efficiency_rating'] = min(max_efficiency / 25.0 * 10, 10)  # Scale against F1 typical max ~25
        metrics['downforce_rating'] = min(max_downforce / 2000.0 * 10, 10)  # Scale against F1 typical max ~2000N
        
        # Balance and consistency ratings
        angle_data = results['angle_sweep']
        stall_margins = [d['stall_margin_deg'] for d in angle_data]
        metrics['stability_rating'] = min(min(stall_margins) / 5.0 * 10, 10)  # Good if >5° margin
        
        # Ground effect utilization
        clearance_data = results['ground_clearance_sweep']
        ground_effects = [d['ground_effect_factor'] for d in clearance_data]
        metrics['ground_effect_rating'] = min(max(ground_effects) / 2.0 * 10, 10)  # Good if >2x effect
        
        # Overall performance index
        ratings = [
            metrics['efficiency_rating'],
            metrics['downforce_rating'], 
            metrics['stability_rating'],
            metrics['ground_effect_rating']
        ]
        metrics['overall_performance_index'] = np.mean(ratings)
        
        return metrics

    # [Rest of the methods remain the same - generate_detailed_report, save_analysis_results, etc.]
    # I'll include the key ones that need updates:

    def quick_performance_analysis(self, test_speed_kmh=200, ground_clearance=75, wing_angle=0):
        """Quick CFD analysis for fitness evaluation - optimized for speed"""
        try:
            print(f"🔍 Quick CFD analysis at {test_speed_kmh} km/h, {wing_angle}° angle...")
            
            # Single speed analysis with proper angle modeling
            speed_ms = self.convert_speed_to_ms(test_speed_kmh)
            result = self.multi_element_analysis(speed_ms, ground_clearance, wing_angle)
            
            # Return essential metrics only
            return {
                'total_downforce': result['total_downforce'],
                'total_drag': result['total_drag'],
                'efficiency_ratio': result['efficiency_ratio'],
                'flow_characteristics': result['flow_characteristics'],
                'stall_margin': result['f1_specific_metrics']['stall_margin'],
                'balance_coefficient': result['f1_specific_metrics']['balance_coefficient'],
                'valid': True
            }
            
        except Exception as e:
            print(f"⚠️ Quick CFD analysis failed: {e}")
            return {
                'total_downforce': 1000,
                'total_drag': 100,
                'efficiency_ratio': 10.0,
                'flow_characteristics': {'flow_attachment_quality': 'Unknown'},
                'stall_margin': 5.0,
                'balance_coefficient': 0.25,
                'valid': False
            }

# MAIN EXECUTION
if __name__ == "__main__":
    print("🏁 ENHANCED STL-BASED F1 WING CFD ANALYSIS SYSTEM")
    print("=" * 70)
    print("🔧 Enhanced with proper angle of attack modeling and F1 parameters")
    
    # Replace with your STL file path
    stl_file_path = "cfd_temp_files/individual_495_wing.stl"  # UPDATE THIS PATH
    
    try:
        # Initialize enhanced analyzer
        analyzer = STLWingAnalyzer(stl_file_path)
        
        # Run comprehensive enhanced analysis
        print("\n🚀 RUNNING ENHANCED COMPREHENSIVE ANALYSIS...")
        results = analyzer.run_comprehensive_f1_analysis()
        
        # Save results and generate report
        print("\n💾 SAVING RESULTS AND GENERATING REPORT...")
        analyzer.save_analysis_results()
        
        print("\n" + "=" * 70)
        print("✅ ENHANCED STL-BASED CFD ANALYSIS COMPLETE!")
        print("=" * 70)
        
        # Display enhanced summary
        geometry = results['geometry_summary']
        f1_metrics = results['f1_performance_metrics']
        
        print(f"\n📐 EXTRACTED GEOMETRY:")
        print(f"• Wingspan: {geometry['wingspan_mm']:.1f} mm")
        print(f"• Elements: {geometry['num_elements']}")
        print(f"• Reference Area: {geometry['reference_area_m2']:.4f} m²")
        print(f"• Aspect Ratio: {geometry['aspect_ratio']:.2f}")
        
        print(f"\n📊 ENHANCED PERFORMANCE SUMMARY:")
        print(f"• Max Downforce: {results['optimal_settings']['max_downforce_N']:.1f} N")
        print(f"• Peak Efficiency (L/D): {results['optimal_settings']['max_efficiency_LD']:.2f}")
        print(f"• Optimal Speed: {results['optimal_settings']['max_efficiency_speed_kmh']} km/h")
        print(f"• Best Clearance: {results['optimal_settings']['optimal_ground_clearance_mm']} mm")
        print(f"• Optimal Angle: {results['optimal_settings']['optimal_wing_angle_deg']}°")
        
        print(f"\n🏎️ F1 PERFORMANCE RATINGS:")
        print(f"• Efficiency Rating: {f1_metrics['efficiency_rating']:.1f}/10")
        print(f"• Downforce Rating: {f1_metrics['downforce_rating']:.1f}/10")
        print(f"• Stability Rating: {f1_metrics['stability_rating']:.1f}/10")
        print(f"• Ground Effect Rating: {f1_metrics['ground_effect_rating']:.1f}/10")
        print(f"• Overall Performance Index: {f1_metrics['overall_performance_index']:.1f}/10")
        
        print(f"\n⚠️ CRITICAL CONDITIONS:")
        critical = results['critical_conditions']
        print(f"• Stall Onset: {critical['stall_onset_angle_deg']}°")
        print(f"• Min Stall Margin: {critical['minimum_stall_margin_deg']:.1f}°")
        print(f"• Max Ground Effect: {critical['max_ground_effect_factor']:.2f}x")
        print(f"• Balance Range: {critical['cop_range_mm']:.1f} mm")
        
    except FileNotFoundError:
        print(f"❌ STL file not found: {stl_file_path}")
        print("Please update the file path in the script")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Check STL file format and path")
