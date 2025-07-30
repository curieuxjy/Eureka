#!/usr/bin/env python3
"""
Script to convert all shadow_*.yaml files to allegro_*.yaml files
in the eureka/cfg/env directory.
"""

import os
import re
import yaml
from pathlib import Path

def convert_shadow_yaml_to_allegro(input_file_path, output_file_path):
    """Convert a single shadow_*.yaml file to allegro_*.yaml file."""
    
    print(f"Converting {input_file_path} -> {output_file_path}")
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the original filename to get the task name
    input_filename = os.path.basename(input_file_path)
    task_name = input_filename.replace('shadow_', '').replace('.yaml', '')
    
    # Convert class name in task field
    content = re.sub(r'task:\s*ShadowHand(\w+)', r'task: AllegroHand\1', content)
    
    # Convert env_name field
    content = re.sub(r'env_name:\s*shadow_hand_(\w+)', r'env_name: allegro_hand_\1', content)
    
    # Convert description references
    content = content.replace('Shadow Hand', 'Allegro Hand')
    content = content.replace('shadow hand', 'allegro hand')
    content = content.replace('ShadowHand', 'AllegroHand')
    
    # Convert any asset file references if they exist
    content = content.replace('shadow_hand.xml', 'allegro_hand.xml')
    content = content.replace('shadow_hand1.xml', 'allegro_hand1.xml')
    
    # Update any DOF-related configurations for Allegro Hand
    # Allegro hand has 16 DOFs per hand vs Shadow's 24
    content = re.sub(r'numActions:\s*24', 'numActions: 16', content)
    content = re.sub(r'numActions:\s*48', 'numActions: 32', content)  # For dual-hand setups
    
    # Write the converted content
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully converted {os.path.basename(input_file_path)}")

def main():
    """Main conversion function."""
    
    # Define paths
    config_dir = Path("/home/avery/Documents/Eureka/eureka/cfg/env")
    
    if not config_dir.exists():
        print(f"Error: Directory {config_dir} does not exist!")
        return
    
    # Find all shadow_*.yaml files
    shadow_files = list(config_dir.glob("shadow_*.yaml"))
    
    if not shadow_files:
        print("No shadow_*.yaml files found!")
        return
    
    print(f"Found {len(shadow_files)} shadow YAML files to convert:")
    
    converted_count = 0
    skipped_count = 0
    
    for shadow_file in shadow_files:
        # Generate corresponding allegro filename
        allegro_filename = shadow_file.name.replace("shadow_", "allegro_")
        allegro_file_path = config_dir / allegro_filename
        
        # Check if allegro file already exists
        if allegro_file_path.exists():
            print(f"⚠️  Skipping {shadow_file.name} - {allegro_filename} already exists")
            skipped_count += 1
            continue
        
        # Convert the file
        try:
            convert_shadow_yaml_to_allegro(shadow_file, allegro_file_path)
            converted_count += 1
        except Exception as e:
            print(f"❌ Error converting {shadow_file.name}: {e}")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {converted_count} files")
    print(f"   Skipped: {skipped_count} files (already exist)")
    print(f"   Total shadow YAML files: {len(shadow_files)}")

if __name__ == "__main__":
    main()
