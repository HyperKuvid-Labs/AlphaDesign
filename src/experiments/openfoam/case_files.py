"""OpenFOAM dictionary text templates for the F1 front-wing validation case.

Every case uses the same physical operating point as the empirical surrogate
(``FitnessEval.evaluate_formula_constratins`` calls the multi-element analysis
at 200 km/h, 75 mm ground clearance, 0 deg wing angle -- see
``fitness_evaluation.py``), so surrogate-vs-OpenFOAM comparisons are apples to
apples. Geometry is exported in millimetres by the wing generator and
converted to metres here; all dictionary values below are already in metres,
seconds, and kilograms.
"""

from __future__ import annotations

MESH_LEVELS = {
    "coarse": {"cells_per_metre": 12, "surface_refinement": (2, 3), "layers": 0, "end_time": 800},
    "medium": {"cells_per_metre": 20, "surface_refinement": (3, 4), "layers": 3, "end_time": 1500},
    "fine": {"cells_per_metre": 32, "surface_refinement": (4, 5), "layers": 6, "end_time": 2500},
}

U_INF = 200.0 / 3.6  # m/s, matches the surrogate's 200 km/h test speed
RHO = 1.225  # kg/m^3, matches FitnessEval / cfd_analysis air_density
NU = 1.5e-5  # m^2/s kinematic viscosity, matches the codebase's own Reynolds formula
GROUND_CLEARANCE_M = 0.075  # matches the surrogate's 75 mm ground clearance
TURBULENCE_INTENSITY = 0.05  # assumption: not specified by the surrogate


def _foam_header(object_name: str, foam_class: str, location: str) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {foam_class};
    object      {object_name};
}}
"""


def turbulence_initial_conditions(reference_length_m: float) -> tuple[float, float]:
    length_scale = 0.07 * reference_length_m
    k = 1.5 * (U_INF * TURBULENCE_INTENSITY) ** 2
    c_mu = 0.09
    omega = k**0.5 / (c_mu**0.25 * length_scale)
    return k, omega


def block_mesh_dict(domain: dict[str, float], cells_per_metre: float) -> str:
    xmin, xmax = domain["xmin"], domain["xmax"]
    ymin, ymax = domain["ymin"], domain["ymax"]
    zmin, zmax = domain["zmin"], domain["zmax"]
    nx = max(10, round((xmax - xmin) * cells_per_metre))
    ny = max(10, round((ymax - ymin) * cells_per_metre))
    nz = max(6, round((zmax - zmin) * cells_per_metre))
    return _foam_header("blockMeshDict", "dictionary", "system") + f"""
convertToMeters 1;

vertices
(
    ({xmin} {ymin} {zmin})
    ({xmax} {ymin} {zmin})
    ({xmax} {ymax} {zmin})
    ({xmin} {ymax} {zmin})
    ({xmin} {ymin} {zmax})
    ({xmax} {ymin} {zmax})
    ({xmax} {ymax} {zmax})
    ({xmin} {ymax} {zmax})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    ground
    {{
        type wall;
        faces ((0 1 2 3));
    }}
    top
    {{
        type patch;
        faces ((4 5 6 7));
    }}
    sides
    {{
        type patch;
        faces ((0 1 5 4) (3 7 6 2));
    }}
);

mergePatchPairs ();
"""


def snappy_hex_mesh_dict(
    location_in_mesh: tuple[float, float, float],
    surface_refinement: tuple[int, int],
    layers: int,
) -> str:
    lo, hi = surface_refinement
    layers_block = ""
    if layers > 0:
        layers_block = f"""
    layers
    {{
        wing
        {{
            nSurfaceLayers {layers};
        }}
    }}

    relativeSizes true;
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.001;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
"""
    return _foam_header("snappyHexMeshDict", "dictionary", "system") + f"""
castellatedMesh true;
snap true;
addLayers {str(layers > 0).lower()};

geometry
{{
    wing.stl
    {{
        type triSurfaceMesh;
        name wing;
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 4000000;
    maxGlobalCells 12000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features ();

    refinementSurfaces
    {{
        wing
        {{
            level ({lo} {hi});
            patchInfo {{ type wall; }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions {{}}

    locationInMesh ({location_in_mesh[0]} {location_in_mesh[1]} {location_in_mesh[2]});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}}

addLayersControls
{{
{layers_block}
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
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

mergeTolerance 1e-6;
"""


def decompose_par_dict(number_of_subdomains: int) -> str:
    return _foam_header("decomposeParDict", "dictionary", "system") + f"""
numberOfSubdomains {number_of_subdomains};
method scotch;
"""


def control_dict(end_time: int, reference: dict[str, float]) -> str:
    return _foam_header("controlDict", "dictionary", "system") + f"""
application simpleFoam;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {end_time};
deltaT 1;
writeControl timeStep;
writeInterval {max(1, end_time // 2)};
purgeWrite 2;
writeFormat ascii;
writePrecision 7;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (wing);
        rho             rhoInf;
        rhoInf          {RHO};
        CofR            ({reference["cofr_x"]} {reference["cofr_y"]} {reference["cofr_z"]});
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (wing);
        rho             rhoInf;
        rhoInf          {RHO};
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        CofR            ({reference["cofr_x"]} {reference["cofr_y"]} {reference["cofr_z"]});
        pitchAxis       (0 1 0);
        magUInf         {U_INF};
        lRef            {reference["l_ref"]};
        Aref            {reference["a_ref"]};
    }}
}}
"""


def fv_schemes() -> str:
    return _foam_header("fvSchemes", "dictionary", "system") + """
ddtSchemes { default steadyState; }

gradSchemes { default Gauss linear; }

divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwindV grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes { default Gauss linear corrected; }

interpolationSchemes { default linear; }

snGradSchemes { default corrected; }

wallDist { method meshWave; }
"""


def fv_solution() -> str:
    return _foam_header("fvSolution", "dictionary", "system") + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.1;
        smoother        GaussSeidel;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    consistent yes;
    nNonOrthogonalCorrectors 1;
    residualControl
    {
        p               1e-4;
        U               1e-5;
        "(k|omega)"     1e-5;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        "(k|omega)"     0.9;
    }
}
"""


def transport_properties() -> str:
    return _foam_header("transportProperties", "dictionary", "constant") + f"""
transportModel Newtonian;
nu              [0 2 -1 0 0 0 0] {NU};
"""


def turbulence_properties() -> str:
    return _foam_header("turbulenceProperties", "dictionary", "constant") + """
simulationType RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""


def field_u() -> str:
    return _foam_header("U", "volVectorField", "0") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({U_INF} 0 0);

boundaryField
{{
    inlet   {{ type fixedValue; value uniform ({U_INF} 0 0); }}
    outlet  {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({U_INF} 0 0); }}
    ground  {{ type movingWallVelocity; value uniform ({U_INF} 0 0); }}
    wing    {{ type noSlip; }}
    top     {{ type slip; }}
    sides   {{ type slip; }}
}}
"""


def field_p() -> str:
    return _foam_header("p", "volScalarField", "0") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{
    inlet   { type zeroGradient; }
    outlet  { type fixedValue; value uniform 0; }
    ground  { type zeroGradient; }
    wing    { type zeroGradient; }
    top     { type slip; }
    sides   { type slip; }
}
"""


def field_k(k_value: float) -> str:
    return _foam_header("k", "volScalarField", "0") + f"""
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_value};

boundaryField
{{
    inlet   {{ type fixedValue; value uniform {k_value}; }}
    outlet  {{ type inletOutlet; inletValue uniform {k_value}; value uniform {k_value}; }}
    ground  {{ type kqRWallFunction; value uniform {k_value}; }}
    wing    {{ type kqRWallFunction; value uniform {k_value}; }}
    top     {{ type slip; }}
    sides   {{ type slip; }}
}}
"""


def field_omega(omega_value: float) -> str:
    return _foam_header("omega", "volScalarField", "0") + f"""
dimensions [0 0 -1 0 0 0 0];
internalField uniform {omega_value};

boundaryField
{{
    inlet   {{ type fixedValue; value uniform {omega_value}; }}
    outlet  {{ type inletOutlet; inletValue uniform {omega_value}; value uniform {omega_value}; }}
    ground  {{ type omegaWallFunction; value uniform {omega_value}; }}
    wing    {{ type omegaWallFunction; value uniform {omega_value}; }}
    top     {{ type slip; }}
    sides   {{ type slip; }}
}}
"""


def field_nut() -> str:
    return _foam_header("nut", "volScalarField", "0") + """
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{
    inlet   { type calculated; value uniform 0; }
    outlet  { type calculated; value uniform 0; }
    ground  { type nutkWallFunction; value uniform 0; }
    wing    { type nutkWallFunction; value uniform 0; }
    top     { type slip; }
    sides   { type slip; }
}
"""


__all__ = [
    "MESH_LEVELS", "U_INF", "RHO", "NU", "GROUND_CLEARANCE_M",
    "turbulence_initial_conditions", "block_mesh_dict", "snappy_hex_mesh_dict",
    "decompose_par_dict", "control_dict", "fv_schemes", "fv_solution",
    "transport_properties", "turbulence_properties",
    "field_u", "field_p", "field_k", "field_omega", "field_nut",
]
