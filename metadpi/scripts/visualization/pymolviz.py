import pymol2
import sys


with pymol2.PyMOL() as p1:
    cmd = p1.cmd
    output_path_dir = sys.argv[1]
    pdbid = sys.argv[2]
    pred_residues = sys.argv[3]
    annotated_resiues = sys.argv[4]
    pred = sys.argv[5]

    cmd.fetch(f"{pdbid}")
    cmd.orient(f"{pdbid}")

    cmd.color("blue")
    cmd.set("cartoon_transparency", "0.75")
    cmd.select("ann", f"resi {annotated_resiues}")
    cmd.indicate("bycalpha ann")
    cmd.create("annotated", "indicate")
    cmd.select("pred", f"resi {pred_residues}")
    cmd.indicate("bycalpha pred")
    cmd.create("predicted", "indicate")

    cmd.show("sphere", "annotated")
    cmd.color("pink","annotated")
    cmd.set("sphere_transparency","0.5","annotated")

    cmd.show("sphere", "predicted")
    cmd.set("sphere_scale","0.5","predicted")
    cmd.color("green","predicted")
    cmd.set("sphere_transparency","0","predicted")
    cmd.set("cartoon_transparency", "1", "predicted")
    cmd.remove("resn hoh")
    cmd.bg_color("white")
    cmd.zoom(complete=1)
    cmd.save(f"{output_path_dir}/{pdbid}/pymol_{pred}.pse")
    cmd.png(f"{output_path_dir}/{pdbid}/pymol_viz_{pred}.png")
    