import argparse
import pathlib
import os
import tempfile
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle
from multiprocessing import Pool
from functools import partial

from general_motion_retargeting.utils.lafan1 import (
    SUPPORTED_BVH_FORMATS,
    load_bvh_file,
    normalize_bvh_format,
)
from general_motion_retargeting.utils.motive_rigid_torso_reduce import convert_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print


def process_single_file(args_dict):
    """
    Process a single BVH file and retarget it to the target robot.
    
    Args:
        args_dict: Dictionary containing:
            - bvh_file_path: Path to the BVH file
            - tgt_file_path: Path where to save the output
            - robot: Target robot type
            - src_format: Source BVH format
            - override: Whether to override existing files
            - target_fps: Target FPS for the motion
            
    Returns:
        tuple: (success: bool, message: str, bvh_file_path: str)
    """
    bvh_file_path = args_dict['bvh_file_path']
    tgt_file_path = args_dict['tgt_file_path']
    robot = args_dict['robot']
    src_format = args_dict['src_format']
    override = args_dict['override']
    target_fps = args_dict['target_fps']
    
    try:
        if os.path.exists(tgt_file_path) and not override:
            return (True, f"Skipped (already exists)", bvh_file_path)
        
        # Load BVH file
        temp_bvh_path = None
        try:
            loaded_bvh_file = bvh_file_path
            load_format = src_format
            if src_format == "motive5spine":
                temp_bvh = tempfile.NamedTemporaryFile(suffix=".bvh", delete=False)
                temp_bvh.close()
                temp_bvh_path = temp_bvh.name
                convert_file(pathlib.Path(bvh_file_path), pathlib.Path(temp_bvh_path))
                loaded_bvh_file = temp_bvh_path
                load_format = "motive2spine"

            lafan1_data_frames, actual_human_height = load_bvh_file(
                loaded_bvh_file, format=load_format
            )
            src_fps = target_fps
        except Exception as e:
            return (False, f"Error loading: {str(e)}", bvh_file_path)
        finally:
            if temp_bvh_path is not None and os.path.exists(temp_bvh_path):
                os.unlink(temp_bvh_path)

        # Initialize the retargeting system
        retarget = GMR(
            src_human=f"bvh_{load_format}",
            tgt_robot=robot,
            actual_human_height=actual_human_height,
        )
        model = mj.MjModel.from_xml_path(retarget.xml_file)
        data = mj.MjData(model)

        # Retarget to get all qpos
        qpos_list = []
        for curr_frame in range(len(lafan1_data_frames)):
            smplx_data = lafan1_data_frames[curr_frame]
            qpos = retarget.retarget(smplx_data)
            qpos_list.append(qpos.copy())
        
        qpos_list = np.array(qpos_list)

        # Initialize the forward kinematics
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kinematics_model = KinematicsModel(retarget.xml_file, device=device)
        
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]
        root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
        dof_pos = qpos_list[:, 7:]
        num_frames = root_pos.shape[0]
        
        # Obtain local body pos
        identity_root_pos = torch.zeros((num_frames, 3), device=device)
        identity_root_rot = torch.zeros((num_frames, 4), device=device)
        identity_root_rot[:, -1] = 1.0
        local_body_pos, _ = kinematics_model.forward_kinematics(
            identity_root_pos, 
            identity_root_rot, 
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
        body_names = kinematics_model.body_names

        HEIGHT_ADJUST = False
        PERFRAME_ADJUST = False
        if HEIGHT_ADJUST:
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
            ground_offset = 0.00
            if not PERFRAME_ADJUST:
                lowest_height = torch.min(body_pos[..., 2]).item()
                root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
            else:
                for i in range(root_pos.shape[0]):
                    lowest_body_part = torch.min(body_pos[i, :, 2])
                    root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

        motion_data = {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos.detach().cpu().numpy(),
            "fps": src_fps,
            "link_body_list": body_names,
        }
        
        os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
        with open(tgt_file_path, "wb") as f:
            pickle.dump(motion_data, f)
        
        return (True, "Completed", bvh_file_path)
        
    except Exception as e:
        return (False, f"Error processing: {str(e)}", bvh_file_path)


def main():
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )

    parser.add_argument(
        "--format",
        choices=SUPPORTED_BVH_FORMATS,
        default="lafan1",
        help="BVH data format.",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )
    
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of worker processes for parallel processing.",
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    src_format = normalize_bvh_format(args.format)

    # Collect all BVH files to process
    files_to_process = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if not filename.endswith(".bvh"):
                continue
                
            bvh_file_path = os.path.join(dirpath, filename)
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")
            
            files_to_process.append({
                'bvh_file_path': bvh_file_path,
                'tgt_file_path': tgt_file_path,
                'robot': args.robot,
                'src_format': src_format,
                'override': args.override,
                'target_fps': args.target_fps,
            })
    
    if not files_to_process:
        print("No BVH files found to process.")
    else:
        print(f"Found {len(files_to_process)} BVH files to process.")
        print(f"Using {args.num_workers} worker processes.")
        
        # Process files using multiprocessing
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_file, files_to_process),
                total=len(files_to_process),
                desc="Retargeting files"
            ))
        
        # Print results
        successful = sum(1 for success, _, _ in results if success)
        failed = len(results) - successful
        
        print(f"\n[green]Completed: {successful}/{len(results)} files processed successfully[/green]")
        
        if failed > 0:
            print(f"[red]Failed: {failed} files[/red]")
            for success, message, bvh_path in results:
                if not success:
                    print(f"  - {bvh_path}: {message}")
        
        print(f"Saved to {tgt_folder}")


if __name__ == "__main__":
    main()
