import sys
import os

original_export = None

def patched_export(self, op, graph, **kwargs):
    from esp_ppq.parser.espdl.layout_patterns import (
        ExporterPatternInfo, 
        restore_origin_shape, 
        logger
    )
    
    if op.type in ["Concat"]:
        perm_dict = {}
        info = ExporterPatternInfo()

        for var in op.inputs:
            var_perm = info.get_var_permute(var.name)
            perm_str = str(var_perm)
            if perm_str not in perm_dict:
                perm_dict[perm_str] = var_perm

        output_var = op.outputs[0]

        if len(perm_dict) == 1:  # all input have same perm, output bypass
            var_perm = list(perm_dict.values())[0]
            if not var_perm:
                restore_origin_shape(op, graph)
            else:
                axis = op.attributes["axis"]
                if axis < 0:
                    axis = len(var_perm) + axis
                new_axis = var_perm.index(int(axis))
                op.attributes["axis"] = new_axis
                info.add_var_permute(output_var.name, var_perm)
                logger.debug(f"{op.name} update axes from {axis} to {new_axis}")
        else:
            logger.debug(f"transpose perm {perm_dict}")
            restore_origin_shape(op, graph)

    return op

def apply_patch():
    from esp_ppq.parser.espdl import layout_patterns
    
    global original_export
    original_export = layout_patterns.ResetConcatPattern.export
    
    layout_patterns.ResetConcatPattern.export = patched_export
    
    print("✓ ESP-PPQ patch applied: Fixed axis=-1 handling in ResetConcatPattern")

if __name__ == "__main__":
    apply_patch()
