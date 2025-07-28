import os
import subprocess
import numpy as np
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Basics.DataStructures import Vector
from PyFoam.Applications.PyFoamSetup import PyFoamSetup
import vtk
from vtk.util import numpy_support

class STLCFDAnalyzer:
    def __init__(self, stl_file, case_name="wing_analysis"):
        self.stl_file = stl_file
        self.case_name = case_name
        self.case_dir = f"./{case_name}"
        
        # CFD Parameters
        self.velocity = 50.0  # m/s
        self.density = 1.225  # kg/m3 (air at sea level)
        self.kinematic_viscosity = 1.5e-5  # m2/s
        self.turbulence_intensity = 0.05
        self.characteristic_length = 1.0  # Default characteristic length in meters
        
        # Calculate Reynolds number dynamically
        self.reynolds_number = (self.density * self.velocity * self.characteristic_length) / self.kinematic_viscosity
        
        # Domain parameters
        self.domain_scale = 10  # Domain size relative to object
        self.mesh_refinement = 3
        
    def setup_openfoam_case(self):
        """Setup complete OpenFOAM case structure"""
        print("Setting up OpenFOAM case structure...")
        
        # Create case directory structure
        dirs = ['0', 'constant', 'system', 'constant/triSurface']
        for d in dirs:
            os.makedirs(f"{self.case_dir}/{d}", exist_ok=True)
        
        # Copy STL file to triSurface directory
        import shutil
        shutil.copy(self.stl_file, f"{self.case_dir}/constant/triSurface/wing.stl")
        
        # Create blockMeshDict
        self.create_blockmesh_dict()
        
        # Create snappyHexMeshDict
        self.create_snappyhex_dict()
        
        # Create boundary conditions
        self.create_boundary_conditions()
        
        # Create solver controls
        self.create_solver_controls()
        
        print("✓ OpenFOAM case setup complete")
    
    def create_blockmesh_dict(self):
        """Create blockMeshDict for background mesh"""
        blockmesh_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    (-{self.domain_scale} -{self.domain_scale} -{self.domain_scale/2})  // 0
    ( {self.domain_scale} -{self.domain_scale} -{self.domain_scale/2})  // 1
    ( {self.domain_scale}  {self.domain_scale} -{self.domain_scale/2})  // 2
    (-{self.domain_scale}  {self.domain_scale} -{self.domain_scale/2})  // 3
    (-{self.domain_scale} -{self.domain_scale}  {self.domain_scale/2})  // 4
    ( {self.domain_scale} -{self.domain_scale}  {self.domain_scale/2})  // 5
    ( {self.domain_scale}  {self.domain_scale}  {self.domain_scale/2})  // 6
    (-{self.domain_scale}  {self.domain_scale}  {self.domain_scale/2})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (40 40 20) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);
'''
        with open(f"{self.case_dir}/system/blockMeshDict", 'w') as f:
            f.write(blockmesh_content)
    
    def create_snappyhex_dict(self):
        """Create snappyHexMeshDict for STL meshing"""
        snappy_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    wing.stl
    {{
        type triSurfaceMesh;
        name wing;
    }}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features
    (
        {{
            file "wing.eMesh";
            level {self.mesh_refinement};
        }}
    );

    refinementSurfaces
    {{
        wing
        {{
            level ({self.mesh_refinement} {self.mesh_refinement + 1});
        }}
    }}

    resolveFeatureAngle 60;

    refinementRegions
    {{
    }}

    locationInMesh (0.1 0.1 0.1);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers
    {{
        wing
        {{
            nSurfaceLayers 3;
        }}
    }}
    expansionRatio 1.3;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minFlatness 0.5;
    minVol 1e-13;
    minTetQuality 1e-9;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
'''
        with open(f"{self.case_dir}/system/snappyHexMeshDict", 'w') as f:
            f.write(snappy_content)
    
    def create_boundary_conditions(self):
        """Create boundary condition files"""
        # Velocity (U) boundary conditions
        u_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({self.velocity} 0 0);
    }}
    
    outlet
    {{
        type            zeroGradient;
    }}
    
    walls
    {{
        type            slip;
    }}
    
    wing
    {{
        type            noSlip;
    }}
}}
'''
        
        # Pressure (p) boundary conditions
        p_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    
    walls
    {
        type            zeroGradient;
    }
    
    wing
    {
        type            zeroGradient;
    }
}
'''
        
        with open(f"{self.case_dir}/0/U", 'w') as f:
            f.write(u_content)
        with open(f"{self.case_dir}/0/p", 'w') as f:
            f.write(p_content)
    
    def create_solver_controls(self):
        """Create solver control files"""
        # controlDict
        control_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    forces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   10;
        patches         (wing);
        rho             rhoInf;
        log             true;
        rhoInf          1.225;
        CofR            (0 0 0);
    }
    
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   10;
        patches         (wing);
        rho             rhoInf;
        log             true;
        rhoInf          1.225;
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        CofR            (0 0 0);
        pitchAxis       (0 1 0);
        magUInf         50.0;
        lRef            0.4;
        Aref            0.2;
    }
}
'''
        
        # fvSchemes
        schemes_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
'''
        
        # fvSolution
        solution_content = '''/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
'''
        
        with open(f"{self.case_dir}/system/controlDict", 'w') as f:
            f.write(control_content)
        with open(f"{self.case_dir}/system/fvSchemes", 'w') as f:
            f.write(schemes_content)
        with open(f"{self.case_dir}/system/fvSolution", 'w') as f:
            f.write(solution_content)
    
    def run_cfd_analysis(self):
        """Execute complete CFD analysis pipeline"""
        print("Starting CFD analysis pipeline...")
        
        # Change to case directory
        original_dir = os.getcwd()
        os.chdir(self.case_dir)
        
        try:
            # Step 1: Generate background mesh
            print("1. Generating background mesh...")
            subprocess.run(['blockMesh'], check=True)
            
            # Step 2: Extract features from STL
            print("2. Extracting surface features...")
            subprocess.run(['surfaceFeatureExtract'], check=True)
            
            # Step 3: Generate mesh around STL
            print("3. Generating mesh around STL geometry...")
            subprocess.run(['snappyHexMesh', '-overwrite'], check=True)
            
            # Step 4: Run CFD solver
            print("4. Running CFD solver...")
            subprocess.run(['simpleFoam'], check=True)
            
            print("✓ CFD analysis complete!")
            
        except subprocess.CalledProcessError as e:
            print(f"Error during CFD analysis: {e}")
        finally:
            os.chdir(original_dir)
    
    def extract_results(self):
        """Extract and analyze CFD results"""
        print("Extracting CFD results...")
        
        results = {
            'lift_coefficient': 0.0,
            'drag_coefficient': 0.0,
            'pressure_distribution': [],
            'velocity_field': [],
            'forces': {}
        }
        
        try:
            # Read force coefficients
            force_file = f"{self.case_dir}/postProcessing/forceCoeffs/0/forceCoeffs.dat"
            if os.path.exists(force_file):
                force_data = np.loadtxt(force_file, skiprows=13)
                if len(force_data) > 0:
                    # Get final converged values
                    results['lift_coefficient'] = force_data[-1, 3]  # Cl
                    results['drag_coefficient'] = force_data[-1, 2]  # Cd
                    
            print(f"✓ Lift Coefficient (Cl): {results['lift_coefficient']:.4f}")
            print(f"✓ Drag Coefficient (Cd): {results['drag_coefficient']:.4f}")
            print(f"✓ L/D Ratio: {results['lift_coefficient']/results['drag_coefficient']:.2f}")
            
        except Exception as e:
            print(f"Warning: Could not extract all results: {e}")
        
        return results
    
    def visualize_results(self):
        """Create visualization of CFD results"""
        print("Creating result visualizations...")
        
        # Convert OpenFOAM results to VTK format
        os.chdir(self.case_dir)
        subprocess.run(['foamToVTK'], check=True)
        
        # You can add more visualization code here using matplotlib or vtk
        print("✓ Results exported to VTK format for visualization")

# Complete usage example
def run_stl_cfd_analysis(stl_filename):
    """Complete CFD analysis workflow for STL file"""
    
    # Initialize CFD analyzer
    analyzer = STLCFDAnalyzer(stl_filename, "ferrari_wing_cfd")
    
    # Setup OpenFOAM case
    analyzer.setup_openfoam_case()
    
    # Run CFD analysis
    analyzer.run_cfd_analysis()
    
    # Extract and display results
    results = analyzer.extract_results()
    
    # Create visualizations
    analyzer.visualize_results()
    
    return results

# Installation requirements
def install_requirements():
    """Install required packages"""
    packages = [
        "PyFoam",
        "numpy", 
        "vtk",
        "matplotlib"
    ]
    
    for package in packages:
        subprocess.run(['pip', 'install', package])

if __name__ == "__main__":
    # Run CFD analysis on the Ferrari wing STL file
    stl_file = "ferrari_sf24_frontwing.stl"
    
    if os.path.exists(stl_file):
        print("=== STL CFD ANALYSIS SYSTEM ===")
        print(f"Analyzing: {stl_file}")
        
        results = run_stl_cfd_analysis(stl_file)
        
        print("\n" + "="*50)
        print("CFD ANALYSIS RESULTS")
        print("="*50)
        print(f"Lift Coefficient: {results.get('lift_coefficient', 0):.4f}")
        print(f"Drag Coefficient: {results.get('drag_coefficient', 0):.4f}")
        print("Analysis complete!")
    else:
        print(f"Error: STL file '{stl_file}' not found!")
