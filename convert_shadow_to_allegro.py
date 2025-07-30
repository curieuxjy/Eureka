#!/usr/bin/env python3
"""
Script to convert all shadow_hand_*.py files to allegro_hand_*.py files
in the eureka/envs/bidex directory.
"""

import os
import re
import shutil
from pathlib import Path

def convert_shadow_to_allegro(input_file_path, output_file_path):
    """Convert a single shadow_hand file to allegro_hand file."""

    print(f"Converting {input_file_path} -> {output_file_path}")

    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Basic class name conversion
    content = re.sub(r'class ShadowHand(\w+)', r'class AllegroHand\1', content)

    # Variable name conversions
    content = content.replace('shadow_hand_dof_speed_scale', 'allegro_dof_speed_scale')
    content = content.replace('num_shadow_hand_dofs', 'num_allegro_hand_dofs')
    content = content.replace('num_shadow_hand_bodies', 'num_allegro_hand_bodies')
    content = content.replace('num_shadow_hand_shapes', 'num_allegro_hand_shapes')
    content = content.replace('num_shadow_hand_actuators', 'num_allegro_hand_actuators')
    content = content.replace('num_shadow_hand_tendons', 'num_allegro_hand_tendons')
    content = content.replace('shadow_hand_dof_', 'allegro_hand_dof_')
    content = content.replace('shadow_hand_another_dof_', 'allegro_hand_another_dof_')
    content = content.replace('shadow_hand_default_dof_', 'allegro_hand_default_dof_')

    # Asset file conversions for allegro hand
    content = content.replace(
        'shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"',
        'allegro_hand_asset_file = "mjcf/open_ai_assets/hand/allegro_hand.xml"'
    )
    content = content.replace(
        'shadow_hand_another_asset_file = "mjcf/open_ai_assets/hand/shadow_hand1.xml"',
        'allegro_hand_another_asset_file = "mjcf/open_ai_assets/hand/allegro_hand1.xml"'
    )

    # Variable references in asset loading
    content = content.replace('shadow_hand_asset_file', 'allegro_hand_asset_file')
    content = content.replace('shadow_hand_another_asset_file', 'allegro_hand_another_asset_file')
    content = content.replace('shadow_hand_asset', 'allegro_hand_asset')
    content = content.replace('shadow_hand_another_asset', 'allegro_hand_another_asset')

    # Actor and DOF properties
    content = content.replace('shadow_hand_dof_props', 'allegro_hand_dof_props')
    content = content.replace('shadow_hand_another_dof_props', 'allegro_hand_another_dof_props')
    content = content.replace('shadow_hand_actor', 'allegro_hand_actor')
    content = content.replace('shadow_hand_another_actor', 'allegro_hand_another_actor')

    # Collections and arrays
    content = content.replace('self.shadow_hands', 'self.allegro_hands')

    # Print statements
    content = content.replace('self.num_shadow_hand_', 'self.num_allegro_hand_')

    # Comments and documentation
    content = content.replace('Shadow Hand', 'Allegro Hand')
    content = content.replace('shadow hand', 'allegro hand')

    # Fingertip names for Allegro Hand (different from Shadow Hand)
    # Note: Allegro hand has different finger naming convention
    allegro_fingertips = [
        '"robot0:link_15.0_tip"',  # index finger
        '"robot0:link_3.0_tip"',   # middle finger
        '"robot0:link_7.0_tip"',   # ring finger
        '"robot0:link_11.0_tip"',  # little finger
        '"robot0:link_15.0_tip"'   # thumb (using index as placeholder)
    ]

    # Replace fingertip definitions for dual hand setup
    fingertip_pattern = r'self\.fingertips = \["robot0:ffdistal",.*?"robot0:thdistal"\]'
    allegro_fingertip_str = 'self.fingertips = ["robot0:link_15.0_tip", "robot0:link_3.0_tip", "robot0:link_7.0_tip", "robot0:link_11.0_tip", "robot0:link_15.0_tip"]'
    content = re.sub(fingertip_pattern, allegro_fingertip_str, content)

    a_fingertip_pattern = r'self\.a_fingertips = \["robot1:ffdistal",.*?"robot1:thdistal"\]'
    allegro_a_fingertip_str = 'self.a_fingertips = ["robot1:link_15.0_tip", "robot1:link_3.0_tip", "robot1:link_7.0_tip", "robot1:link_11.0_tip", "robot1:link_15.0_tip"]'
    content = re.sub(a_fingertip_pattern, allegro_a_fingertip_str, content)

    # Tendon references for Allegro Hand (remove tendon logic as Allegro doesn't use tendons)
    content = re.sub(r'relevant_tendons = \[.*?\]', 'relevant_tendons = []', content, flags=re.DOTALL)
    content = re.sub(r'a_relevant_tendons = \[.*?\]', 'a_relevant_tendons = []', content, flags=re.DOTALL)

    # Remove tendon property settings (Allegro hand doesn't use tendons)
    tendon_block_pattern = r'for i in range\(self\.num_allegro_hand_tendons\):.*?self\.gym\.set_asset_tendon_properties\(allegro_hand_another_asset, a_tendon_props\)'
    content = re.sub(tendon_block_pattern, '# Allegro hand does not use tendons', content, flags=re.DOTALL)

    # Update DOF configurations for Allegro Hand (16 DOFs vs Shadow's 24)
    # Allegro hand has 16 DOFs (4 fingers x 4 joints each)
    content = re.sub(
        r'to_torch\(\[0\.0, 0\.0, -0,.*?-1\.57\]',
        'to_torch([0.0] * 16',  # 16 zeros for Allegro hand default positions
        content
    )

    # Update force tensor slicing for Allegro hand
    content = content.replace('self.dof_force_tensor[:, :24]', 'self.dof_force_tensor[:, :16]')
    content = content.replace('self.dof_force_tensor[:, 24:48]', 'self.dof_force_tensor[:, 16:32]')

    # Update action slicing for Allegro hand
    content = content.replace('self.actions[:, 6:26]', 'self.actions[:, 6:22]')  # 16 DOFs instead of 20
    content = content.replace('self.actions[:, 32:52]', 'self.actions[:, 22:38]')  # Adjusted for 16 DOFs
    content = content.replace('+ 24]', '+ 16]')  # Update DOF indices

    # Write the converted content
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Successfully converted {os.path.basename(input_file_path)}")

def main():
    """Main conversion function."""

    # Define paths
    bidex_dir = Path("/home/avery/Documents/Eureka/eureka/envs/bidex")

    if not bidex_dir.exists():
        print(f"Error: Directory {bidex_dir} does not exist!")
        return

    # Find all shadow_hand_*.py files
    shadow_files = list(bidex_dir.glob("shadow_hand_*.py"))

    if not shadow_files:
        print("No shadow_hand_*.py files found!")
        return

    print(f"Found {len(shadow_files)} shadow_hand files to convert:")

    converted_count = 0
    skipped_count = 0

    for shadow_file in shadow_files:
        # Generate corresponding allegro_hand filename
        allegro_filename = shadow_file.name.replace("shadow_hand_", "allegro_hand_")
        allegro_file_path = bidex_dir / allegro_filename

        # Check if allegro file already exists
        if allegro_file_path.exists():
            print(f"⚠️  Skipping {shadow_file.name} - {allegro_filename} already exists")
            skipped_count += 1
            continue

        # Convert the file
        try:
            convert_shadow_to_allegro(shadow_file, allegro_file_path)
            converted_count += 1
        except Exception as e:
            print(f"❌ Error converting {shadow_file.name}: {e}")

    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {converted_count} files")
    print(f"   Skipped: {skipped_count} files (already exist)")
    print(f"   Total shadow files: {len(shadow_files)}")

if __name__ == "__main__":
    main()
