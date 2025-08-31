import os

def check_data_health(base_path):
    """
    Verifies the file structure and completeness of the organized UR Fall dataset.
    """
    print("--- Starting Data Health Check ---")
    
    # Define the base directory for the processed data
    data_dir = os.path.join(base_path, 'data', 'processed')
    if not os.path.isdir(data_dir):
        print(f"\n[ERROR] Data directory not found at: {data_dir}")
        print("Please ensure your 'processed' data folder is inside a 'data' folder.")
        return

    # Configuration for falls and ADLs
    event_configs = {
        'falls': {'count': 30, 'prefix': 'fall'},
        'adls': {'count': 40, 'prefix': 'adl'}
    }
    
    missing_items = []
    total_events_checked = 0

    # Loop through each event type (falls, adls)
    for event_type, config in event_configs.items():
        print(f"\n--- Checking '{event_type.upper()}' directory ---")
        
        # Loop through each event number (e.g., fall-01 to fall-30)
        for i in range(1, config['count'] + 1):
            event_id = f"{i:02d}"
            event_name = f"{config['prefix']}-{event_id}"
            event_path = os.path.join(data_dir, event_type, event_name)
            
            total_events_checked += 1
            
            # Define the expected structure for each event
            # Note: Unzipped folders are named after the zip file, so we check for those.
            expected_structure = [
                # Metadata and Previews
                f"{event_name}-data.csv",
                f"{event_name}-cam0.mp4",
                # Accelerometer data
                f"acc/{event_name}-acc.csv",
                # RGB and Depth image folders (after unzipping)
                "cam0/rgb/",
                "cam0/depth/",
            ]
            
            # Falls have an additional camera (cam1)
            if event_type == 'falls':
                expected_structure.extend([
                    f"{event_name}-cam1.mp4",
                    "cam1/rgb/",
                    "cam1/depth/",
                ])

            # Check if the base event folder exists
            if not os.path.isdir(event_path):
                missing_items.append(f"Missing event directory: {event_path}")
                continue # Skip checks for this event if its folder is missing

            # Check each expected file/directory within the event folder
            for item in expected_structure:
                item_path = os.path.join(event_path, item)
                if not os.path.exists(item_path):
                    missing_items.append(f"Missing item: {item_path}")

    # --- Final Report ---
    print("\n--- Health Check Report ---")
    print(f"Total events checked: {total_events_checked}")
    
    if not missing_items:
        print("\n✅ SUCCESS: All files and directories are correctly in place.")
        print("Your dataset has passed the health check!")
    else:
        print(f"\n❌ FAILED: Found {len(missing_items)} missing item(s).")
        print("Please review the following issues:")
        for item in missing_items:
            print(f"  - {item}")
            
    print("\n--- End of Report ---")


if __name__ == "__main__":
    # Get the current working directory, which should be the project's root folder
    project_root = os.getcwd()
    check_data_health(project_root)