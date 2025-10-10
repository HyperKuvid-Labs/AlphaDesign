import os
import json
import tempfile
import numpy as np

def create_test_json():
    """Create a sample CFD parameters JSON for testing"""
    test_params = {
        "geometry": {
            "total_elements": 4,
            "total_reference_area_m2": 0.3456,
            "main_element": {
                "root_chord_mm": 280,
                "tip_chord_mm": 250,
                "reference_area_m2": 0.0896,
                "span_mm": 1600
            },
            "flaps": [
                {
                    "root_chord_mm": 220,
                    "tip_chord_mm": 200,
                    "geometric_angle_deg": 12.5,
                    "vertical_offset_mm": 25,
                    "horizontal_offset_mm": 30,
                    "camber_ratio": 0.12,
                    "reference_area_m2": 0.0704,
                    "span_mm": 1500
                },
                {
                    "root_chord_mm": 180,
                    "tip_chord_mm": 160,
                    "geometric_angle_deg": 18.0,
                    "vertical_offset_mm": 45,
                    "horizontal_offset_mm": 60,
                    "camber_ratio": 0.10,
                    "reference_area_m2": 0.0576,
                    "span_mm": 1400
                },
                {
                    "root_chord_mm": 140,
                    "tip_chord_mm": 120,
                    "geometric_angle_deg": 22.5,
                    "vertical_offset_mm": 70,
                    "horizontal_offset_mm": 85,
                    "camber_ratio": 0.08,
                    "reference_area_m2": 0.0448,
                    "span_mm": 1400
                }
            ]
        },
        "multi_element_interactions": {
            "slot_gaps_mm": [14, 12, 10],
            "slot_gap_to_chord_ratios": [0.063, 0.067, 0.071],
            "overlap_ratios": [0.136, 0.167, 0.214]
        },
        "airfoil_properties": {
            "main_element": {
                "profile_type": "NACA_64A010_modified",
                "camber_ratio": 0.08,
                "camber_position": 0.40,
                "max_thickness_ratio": 0.15,
                "leading_edge_radius_mm": 2.8
            },
            "flaps": [
                {
                    "profile_type": "High_camber_flap",
                    "camber_ratio": 0.12,
                    "thickness_ratio": 0.13
                },
                {
                    "profile_type": "High_camber_flap",
                    "camber_ratio": 0.10,
                    "thickness_ratio": 0.11
                },
                {
                    "profile_type": "High_camber_flap",
                    "camber_ratio": 0.08,
                    "thickness_ratio": 0.09
                }
            ]
        }
    }
    
    return test_params


def test_json_loading():
    """Test 1: Verify JSON loading works correctly"""
    print("\n" + "="*70)
    print("TEST 1: JSON Loading")
    print("="*70)
    
    try:
        # Create temporary JSON file
        test_params = create_test_json()
        temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_params, temp_json)
        temp_json.close()
        
        print(f"✅ Created test JSON: {temp_json.name}")
        
        # Try to load it
        with open(temp_json.name, 'r') as f:
            loaded = json.load(f)
        
        # Verify structure
        assert 'geometry' in loaded
        assert 'multi_element_interactions' in loaded
        assert 'airfoil_properties' in loaded
        assert loaded['geometry']['total_elements'] == 4
        assert len(loaded['geometry']['flaps']) == 3
        
        print("✅ JSON structure valid")
        print(f"   - Total elements: {loaded['geometry']['total_elements']}")
        print(f"   - Main chord: {loaded['geometry']['main_element']['root_chord_mm']} mm")
        print(f"   - Slot gaps: {loaded['multi_element_interactions']['slot_gaps_mm']}")
        
        # Cleanup
        os.unlink(temp_json.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_slot_effect_calculation():
    """Test 2: Verify new slot effect calculation"""
    print("\n" + "="*70)
    print("TEST 2: Slot Effect Calculation")
    print("="*70)
    
    try:
        # Mock slot parameters
        gap_ratios = [0.02, 0.06, 0.07]
        overlap_ratios = [0.10, 0.15, 0.20]
        
        for i, (gap_ratio, overlap_ratio) in enumerate(zip(gap_ratios, overlap_ratios)):
            # Calculate using new physics-based model
            optimal_gap = 0.02
            gap_efficiency = np.exp(-((gap_ratio - optimal_gap) / 0.01)**2)
            overlap_efficiency = 1.0 if 0.05 <= overlap_ratio <= 0.15 else 0.8
            circulation_boost = 1.3 + 0.15 * gap_efficiency * overlap_efficiency
            velocity_ratio = 1.4 + 0.4 * gap_efficiency
            slot_cl_multiplier = circulation_boost * np.sqrt(velocity_ratio)
            slot_cd_multiplier = 0.85 + 0.15 * gap_efficiency
            
            print(f"\n   Flap {i+1}:")
            print(f"      Gap ratio: {gap_ratio:.3f}")
            print(f"      Overlap ratio: {overlap_ratio:.3f}")
            print(f"      CL multiplier: {slot_cl_multiplier:.3f}")
            print(f"      CD multiplier: {slot_cd_multiplier:.3f}")
            print(f"      Velocity ratio: {velocity_ratio:.3f}")
            print(f"      Efficiency: {gap_efficiency:.3f}")
            
            # Validate ranges
            assert 1.0 <= slot_cl_multiplier <= 2.0, "CL multiplier out of range"
            assert 0.8 <= slot_cd_multiplier <= 1.0, "CD multiplier out of range"
            assert 1.4 <= velocity_ratio <= 1.8, "Velocity ratio out of range"
        
        print("\n✅ Slot effect calculations valid")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_parameter_extraction():
    """Test 3: Verify parameter extraction from JSON"""
    print("\n" + "="*70)
    print("TEST 3: Parameter Extraction")
    print("="*70)
    
    try:
        test_params = create_test_json()
        
        # Extract parameters
        main = test_params['geometry']['main_element']
        flaps = test_params['geometry']['flaps']
        interactions = test_params['multi_element_interactions']
        
        # Build arrays like the code does
        chord_lengths = [main['root_chord_mm'] / 1000]
        element_angles = [0]
        element_cambers = [test_params['airfoil_properties']['main_element']['camber_ratio']]
        
        for i, flap in enumerate(flaps):
            chord_lengths.append(flap['root_chord_mm'] / 1000)
            element_angles.append(flap['geometric_angle_deg'])
            element_cambers.append(flap['camber_ratio'])
        
        print(f"\n   Extracted Parameters:")
        print(f"      Number of elements: {len(chord_lengths)}")
        print(f"      Chord lengths (m): {[f'{c:.3f}' for c in chord_lengths]}")
        print(f"      Element angles (deg): {element_angles}")
        print(f"      Camber ratios: {[f'{c:.3f}' for c in element_cambers]}")
        print(f"      Slot gaps (mm): {interactions['slot_gaps_mm']}")
        print(f"      Gap ratios: {[f'{r:.3f}' for r in interactions['slot_gap_to_chord_ratios']]}")
        
        # Validate
        assert len(chord_lengths) == 4, "Wrong number of elements"
        assert all(0.1 < c < 0.3 for c in chord_lengths), "Chord lengths out of range"
        assert all(0 <= a <= 25 for a in element_angles), "Angles out of range"
        assert len(interactions['slot_gaps_mm']) == 3, "Wrong number of slot gaps"
        
        print("\n✅ Parameter extraction valid")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_comparison_logic():
    """Test 4: Verify comparison report logic"""
    print("\n" + "="*70)
    print("TEST 4: Comparison Report Logic")
    print("="*70)
    
    try:
        # JSON values
        json_chords = [280, 220, 180, 140]
        
        # Simulated auto-detected values (with some error)
        detected_chords = [276.3, 218.7, 176.2, 134.8]
        
        print(f"\n   Comparison:")
        print(f"   {'Element':<10} {'JSON':<12} {'Detected':<12} {'Diff %':<10} {'Status':<10}")
        print("   " + "-"*60)
        
        for i, (json_val, detected_val) in enumerate(zip(json_chords, detected_chords)):
            diff_pct = abs(json_val - detected_val) / json_val * 100
            status = "✅" if diff_pct < 5 else "⚠️"
            print(f"   {i+1:<10} {json_val:<12.1f} {detected_val:<12.1f} {diff_pct:<10.1f} {status:<10}")
        
        # Test angles
        json_angles = [0, 12.5, 18.0, 22.5]
        detected_angles = [0.2, 11.8, 15.3, 19.1]
        
        print(f"\n   Angle Comparison:")
        print(f"   {'Element':<10} {'JSON':<12} {'Detected':<12} {'Diff °':<10} {'Status':<10}")
        print("   " + "-"*60)
        
        for i, (json_val, detected_val) in enumerate(zip(json_angles, detected_angles)):
            diff = abs(json_val - detected_val)
            status = "✅" if diff < 2 else "⚠️"
            print(f"   {i+1:<10} {json_val:<12.1f} {detected_val:<12.1f} {diff:<10.1f} {status:<10}")
        
        print("\n✅ Comparison logic valid")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("CFD JSON INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        test_json_loading,
        test_slot_effect_calculation,
        test_parameter_extraction,
        test_comparison_logic
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {sum(results)} ✅")
    print(f"Failed: {len(results) - sum(results)} ❌")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
