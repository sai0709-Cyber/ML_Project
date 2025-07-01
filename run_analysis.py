import argparse
import csv
import os
import numpy as np 
from datetime import datetime
from ansys.mapdl.core.launcher import launch_mapdl 



def _get_field(mapdl, label: str, comp: str):
    pp = mapdl.post_processing          
    if label == "U":
        if comp == "SUM":
            disp = pp.nodal_displacement()
            disp = np.asarray(disp)

            if disp.ndim == 2 and disp.shape[1] == 3:
                return np.linalg.norm(disp,axis=1)
            elif disp.ndim==1:
                return np.abs(disp)
                magnitude = np.linalg.norm(disp, axis=1)
            else:
                raise ValueError(f"Unexpected displacement shape: {disp.shape}")
            

        elif comp in ("X", "Y", "Z"):
            return pp.nodal_displacement(component=comp)
        else:
            raise ValueError(f"Bad comp '{comp}' for displacements")
    elif label == "S":
        if comp == "1":
            return pp.nodal_principal_stress(1)        # S1
        elif comp == "3":
            return pp.nodal_principal_stress(3)        # S3
        elif comp == "EQV":

        
         # Pull each component one by one
         sx = pp.nodal_component_stress("X")
         sy = pp.nodal_component_stress("Y")
         sz = pp.nodal_component_stress("Z")
         sxy = pp.nodal_component_stress("XY")
         syz = pp.nodal_component_stress("YZ")
         sxz = pp.nodal_component_stress("XZ")

    # Convert to NumPy arrays
         sx, sy, sz = map(np.asarray, (sx, sy, sz))
         sxy, syz, sxz = map(np.asarray, (sxy, syz, sxz))

         return np.sqrt(
             0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) +
             3 * (sxy**2 + syz**2 + sxz**2)
          )

            
        else:
         raise ValueError(f"Unsupported label '{label}'")


def stats(mapdl, label: str, comp: str):
    """Wrapper around _stats + _get_field so the call site stays tiny."""
    return _stats(_get_field(mapdl, label, comp))



######----------

def init_mapdl(jobname="beam"):
    from ansys.mapdl.core.launcher import launch_mapdl 

    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_location = f"beam_sim_{timestamp}"

    mapdl = launch_mapdl(run_location=run_location,
                          jobname=jobname,
                            override=True,
                            port=50058,
                            license_type="ansys" ,
                            additional_switches="-smp -d win32 -s noread",
                            start_timeout=120,
                            loglevel="ERROR",
                            )

    mapdl.clear()
    mapdl.prep7()
    mapdl.units("SI")  
    mapdl.et(1, "SOLID185")
    mapdl.mp("EX", 1, 210e9)   
    mapdl.mp("PRXY", 1, 0.3)    
    return mapdl

#-----------output--------------
def _stats(vec):
    
    return float(vec.min()), float(vec.max()), float(vec.mean())


def solve_and_report(mapdl):
    
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.solve()
    mapdl.finish()

    # --- post‑process
    mapdl.post1() 
    mapdl.set(1) 
    #-------- POST process--------
    def stats(label, comp):
     values = _get_field(mapdl, label, comp)
     return float(np.min(values)), float(np.max(values)), float(np.mean(values))
     
     if label == "U":
        values = mapdl.prnsol("U", comp)
     elif label == "S":
        values = mapdl.prnsol("S", comp, item="ELEM")
     else:
        raise ValueError("Unsupported label for stats.")
     return (
            mapdl.get_scalar("STAT", 1),  
            mapdl.get_scalar("STAT", 2), 
            mapdl.get_scalar("STAT", 3),  
        )

  # Displacements
    tdef_min, tdef_max, tdef_avg = stats("U", "SUM")
    dx_min,   dx_max,   dx_avg   = stats("U", "X")

    # Principal stresses
    s1_min, s1_max, s1_avg = stats("S", "1")  # max principal
    s3_min, s3_max, s3_avg = stats("S", "3")  # min principal

    # Von Mises equivalent stress
    seqv_min, seqv_max, seqv_avg = stats("S", "EQV")

    print("\n--- Result Summary ---")
    print(f"Total Deformation (m): min={tdef_min:.6e}, max={tdef_max:.6e}, avg={tdef_avg:.6e}")
    print(f"X‑Dir Deformation (m): min={dx_min:.6e}, max={dx_max:.6e}, avg={dx_avg:.6e}")
    print(f"Max Principal Stress (Pa): min={s1_min:.6e}, max={s1_max:.6e}, avg={s1_avg:.6e}")
    print(f"Min Principal Stress (Pa): min={s3_min:.6e}, max={s3_max:.6e}, avg={s3_avg:.6e}")
    print(f"Von Mises Stress (Pa): min={seqv_min:.6e}, max={seqv_max:.6e}, avg={seqv_avg:.6e}")

# --- write CSV inside run folder
    run_dir = mapdl.directory  # the working directory MAPDL is running in
    csv_path = os.path.join(run_dir, "results.csv")
    header = [
        "timestamp", "shape", "tdef_min", "tdef_max", "tdef_avg",
        "dx_min", "dx_max", "dx_avg",
        "s1_min", "s1_max", "s1_avg",
        "s3_min", "s3_max", "s3_avg",
        "seqv_min", "seqv_max", "seqv_avg",
    ]

    row = [
        datetime.now().isoformat(timespec="seconds"),  # time
        mapdl.jobname,  # shape/job identifier
        tdef_min, tdef_max, tdef_avg,
        dx_min, dx_max, dx_avg,
        s1_min, s1_max, s1_avg,
        s3_min, s3_max, s3_avg,
        seqv_min, seqv_max, seqv_avg,
    ]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    mapdl.finish()
    return tdef_max

# --------------------------- geometries --------------------------------

def model_rectangular_beam(mapdl, length, width, height, pressure):
    mapdl.prep7()
    mapdl.block(0, length, -width/2, width/2, -height/2, height/2)
    mapdl.esize(min(length, width, height)/4); mapdl.vmesh("ALL")
    mapdl.nsel("S", "LOC", "X", 0); mapdl.d("ALL", "ALL", 0)
    mapdl.allsel(); mapdl.nsel("S", "LOC", "X", length); mapdl.sf("ALL", "PRES", pressure); mapdl.allsel()
    return solve_and_report(mapdl)

def model_spherical_beam(mapdl, radius, pressure):
    mapdl.prep7()

    mapdl.block(-radius, radius, -radius, radius, -radius, radius)

    mapdl.esize(radius/4); mapdl.vmesh("ALL")
    mapdl.nsel("S", "LOC", "Y", -radius); mapdl.d("ALL", "ALL", 0)
    mapdl.allsel(); mapdl.sf("ALL", "PRES", pressure); mapdl.allsel()
    return solve_and_report(mapdl)

def model_t_beam(mapdl, length, web_thk, flange_width, flange_thk, pressure):
    mapdl.clear("NOSTART")
    mapdl.prep7()

    num_volumes = mapdl.get_value("VOLU", 0, "NUM", "VOLU")
    print(f"Number of volumes = {int(num_volumes)}")


    # Web block (vertical stem of T)
    web_ymin = -web_thk / 2
    web_ymax = web_thk / 2
    web_zmin = 0
    web_zmax = flange_thk
    mapdl.block(0, length, web_ymin, web_ymax, web_zmin, web_zmax)

    # Flange block (top cap of T)
    flange_ymin = -flange_width / 2
    flange_ymax = flange_width / 2
    flange_zmin = web_zmax
    flange_zmax = web_zmax + web_thk
    mapdl.block(0, length, flange_ymin, flange_ymax, flange_zmin, flange_zmax)

    # Combine both volumes
    #mapdl.vadd("ALL")
    print(mapdl.vstatus())

    mapdl.nummrg("NODE")
    mapdl.nummrg("KP")
    mapdl.nummrg("ALL")
    mapdl.nummrg("ELEM")


    # Element size and mesh
    mapdl.esize(min(length, web_thk, flange_width, flange_thk) / 4)
    mapdl.vmesh("ALL")

    # Fix one end of the beam
    mapdl.nsel("S", "LOC", "X", 0)
    mapdl.d("ALL", "ALL", 0)

    # Apply pressure on the opposite end
    mapdl.allsel()
    mapdl.nsel("S", "LOC", "X", length)
    mapdl.sf("ALL", "PRES", pressure)
    mapdl.allsel()

    return solve_and_report(mapdl)





# --------------------------- CLI driver ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate & solve simple 3‑D beams in Ansys MAPDL with CSV output")
    sub = parser.add_subparsers(dest="shape", required=True)

    p_rect = sub.add_parser("rect"); p_rect.add_argument("--length", type=float, default=0.3); p_rect.add_argument("--width", type=float, default=0.05); p_rect.add_argument("--height", type=float, default=0.05); p_rect.add_argument("--pressure", type=float, default=5e6)
    p_sph = sub.add_parser("sphere"); p_sph.add_argument("--radius", type=float, default=0.1); p_sph.add_argument("--pressure", type=float, default=1e6)
    p_t = sub.add_parser("tbeam"); p_t.add_argument("--length", type=float, default=0.3); p_t.add_argument("--web_thk", type=float, default=0.02); p_t.add_argument("--flange_width", type=float, default=0.08); p_t.add_argument("--flange_thk", type=float, default=0.02); p_t.add_argument("--pressure", type=float, default=5e6)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_loc = f"beam_sim_{timestamp}"
    mapdl = init_mapdl(jobname=args.shape)

    if args.shape == "rect":
        model_rectangular_beam(mapdl, args.length, args.width, args.height, args.pressure)
    elif args.shape == "sphere":
        model_spherical_beam(mapdl, args.radius, args.pressure)
    else:
        model_t_beam(mapdl, args.length, args.web_thk, args.flange_width, args.flange_thk, args.pressure)

    mapdl.exit()

if __name__ == "__main__":
    main()
