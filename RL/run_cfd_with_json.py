import os
import sys
from cfd_analysis import STLWingAnalyzer, F1CFDPipeline

def run_cfd_with_accurate_parameters(stl_path, json_path=None):
    """
    Run CFD analysis with accurate parameters from wing generator
    
    Args:
        stl_path: Path to STL file
        json_path: Optional path to CFD parameters JSON (auto-detected if None)
    """
    
    # Auto-detect JSON file if not provided
    if json_path is None:
        base_name = stl_path.replace('.stl', '')
        json_path = f"{base_name}_cfd_params.json"
    
    # Check files exist
    if not os.path.exists(stl_path):
        print(f"❌ STL file not found: {stl_path}")
        return None
    
    if not os.path.exists(json_path):
        print(f"⚠️ JSON parameters file not found: {json_path}")
        print("   Will proceed with auto-detection (less accurate)")
        json_path = None
    
    print("=" * 80)
    print("🏎️ F1 WING CFD ANALYSIS WITH ACCURATE PARAMETERS")
    print("=" * 80)
    print(f"📁 STL File: {stl_path}")
    if json_path:
        print(f"📋 Parameters: {json_path}")
    print()
    
    # Initialize analyzer
    analyzer = STLWingAnalyzer(stl_path, cfd_params_json=json_path)
    
    # Generate comparison report if JSON available
    if json_path:
        analyzer.generate_comparison_report()
    
    # Run comprehensive analysis
    print("\n🚀 Running comprehensive CFD analysis...")
    results = analyzer.run_comprehensive_f1_analysis()
    
    # Display key results
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS SUMMARY")
    print("=" * 80)
    
    # Optimal settings
    opt = results['optimal_settings']
    print(f"\n🎯 OPTIMAL SETTINGS:")
    print(f"   Max Efficiency Speed: {opt['max_efficiency_speed_kmh']} km/h")
    print(f"   Max Efficiency L/D: {opt['max_efficiency_LD']:.2f}")
    print(f"   Max Downforce Speed: {opt['max_downforce_speed_kmh']} km/h")
    print(f"   Max Downforce: {opt['max_downforce_N']:.0f} N")
    print(f"   Optimal Ground Clearance: {opt['optimal_ground_clearance_mm']} mm")
    print(f"   Optimal Wing Angle: {opt['optimal_wing_angle_deg']:.1f}°")
    
    # Critical conditions
    crit = results['critical_conditions']
    print(f"\n⚠️ CRITICAL CONDITIONS:")
    print(f"   Stall Onset Angle: {crit['stall_onset_angle_deg']:.1f}°")
    print(f"   Minimum Stall Margin: {crit['minimum_stall_margin_deg']:.1f}°")
    print(f"   Max Ground Effect Factor: {crit['max_ground_effect_factor']:.2f}x")
    print(f"   Center of Pressure Range: {crit['cop_range_mm']:.1f} mm")
    
    # Performance metrics
    metrics = results['f1_performance_metrics']
    print(f"\n📈 F1 PERFORMANCE RATINGS (1-10 scale):")
    print(f"   Efficiency: {metrics['efficiency_rating']:.1f}/10")
    print(f"   Downforce: {metrics['downforce_rating']:.1f}/10")
    print(f"   Stability: {metrics['stability_rating']:.1f}/10")
    print(f"   Ground Effect: {metrics['ground_effect_rating']:.1f}/10")
    print(f"   Overall Index: {metrics['overall_performance_index']:.1f}/10")
    
    # Element-by-element breakdown at reference speed
    ref_speed_data = None
    for data in results['speed_sweep']:
        if data['speed_kmh'] == 200:
            ref_speed_data = data
            break
    
    if ref_speed_data and 'elements' in ref_speed_data:
        print(f"\n🔬 ELEMENT BREAKDOWN (200 km/h, 75mm ride height):")
        print(f"   {'Element':<10} {'Chord(mm)':<12} {'CL':<8} {'CD':<8} {'DF(N)':<10} {'Slot Eff':<10}")
        print("   " + "-" * 70)
        for elem in ref_speed_data['elements']:
            elem_num = elem['element_number']
            chord = elem['chord_length_mm']
            cl = elem['lift_coefficient']
            cd = elem['drag_coefficient']
            df = elem['downforce_N']
            slot_eff = elem.get('slot_efficiency', 1.0)
            print(f"   {elem_num:<10} {chord:<12.1f} {cl:<8.3f} {cd:<8.4f} {df:<10.0f} {slot_eff:<10.2f}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    
    return results


def quick_performance_check(stl_path, json_path=None, speed_kmh=200):
    """
    Quick performance check at single speed
    
    Args:
        stl_path: Path to STL file
        json_path: Optional path to CFD parameters JSON
        speed_kmh: Test speed in km/h
    """
    
    # Auto-detect JSON
    if json_path is None:
        base_name = stl_path.replace('.stl', '')
        json_path = f"{base_name}_cfd_params.json"
        if not os.path.exists(json_path):
            json_path = None
    
    print(f"\n🔍 Quick performance check at {speed_kmh} km/h...")
    
    analyzer = STLWingAnalyzer(stl_path, cfd_params_json=json_path)
    result = analyzer.quick_performance_analysis(test_speed_kmh=speed_kmh)
    
    print(f"\n📊 Results at {speed_kmh} km/h:")
    print(f"   Downforce: {result['total_downforce']:.0f} N")
    print(f"   Drag: {result['total_drag']:.0f} N")
    print(f"   Efficiency L/D: {result['efficiency_ratio']:.2f}")
    print(f"   Flow Status: {result['flow_characteristics']['flow_attachment']}")
    
    return result


if __name__ == "__main__":
    # Example usage
    
    # Method 1: Full analysis with auto-detected JSON
    if len(sys.argv) > 1:
        stl_file = sys.argv[1]
        json_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        results = run_cfd_with_accurate_parameters(stl_file, json_file)
    
    else:
        # Default example paths
        print("Usage:")
        print("  python run_cfd_with_json.py <stl_file> [json_file]")
        print()
        print("Example:")
        print("  python run_cfd_with_json.py f1_wing_output/my_wing.stl")
        print("  python run_cfd_with_json.py f1_wing_output/my_wing.stl f1_wing_output/my_wing_cfd_params.json")
        print()
        print("If json_file is not specified, it will be auto-detected as <stl_file>_cfd_params.json")
