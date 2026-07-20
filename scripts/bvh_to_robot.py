import argparse
import pathlib
import pickle
import tempfile
import time
from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    ROBOT_MOTION_VIEWER_IMPORT_ERROR,
    RobotMotionViewer,
)
from general_motion_retargeting.utils.lafan1 import (
    SUPPORTED_BVH_FORMATS,
    load_bvh_file,
    normalize_bvh_format,
)
from general_motion_retargeting.utils.motive_rigid_torso_reduce import convert_file
from rich import print
from tqdm import tqdm
import os
import numpy as np


def infer_bvh_format(bvh_file):
    path_parts = pathlib.Path(bvh_file).parts
    for part in reversed(path_parts):
        normalized = normalize_bvh_format(part.lower())
        if normalized in SUPPORTED_BVH_FORMATS:
            return normalized
    return "lafan1"


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        "--input",
        help="BVH motion file to load.",
        required=True,
        dest="bvh_file",
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=("auto", *SUPPORTED_BVH_FORMATS),
        default="auto",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
        ],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        "--output",
        default=None,
        dest="save_path",
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Show the MuJoCo viewer while retargeting.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )
    
    args = parser.parse_args()
    if args.format == "auto":
        src_format = infer_bvh_format(args.bvh_file)
    else:
        src_format = normalize_bvh_format(args.format)
    visualize = args.visualize or args.save_path is None or args.record_video

    if visualize and RobotMotionViewer is None:
        raise ImportError(
            "RobotMotionViewer dependencies are not installed. Install the package "
            "requirements, including loop_rate_limiters, or pass --output without "
            "--visualize to retarget without opening the viewer."
        ) from ROBOT_MOTION_VIEWER_IMPORT_ERROR
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    loaded_bvh_file = args.bvh_file
    load_format = src_format
    temp_bvh = None
    if src_format == "motive5spine":
        temp_bvh = tempfile.NamedTemporaryFile(suffix=".bvh", delete=False)
        temp_bvh.close()
        convert_file(pathlib.Path(args.bvh_file), pathlib.Path(temp_bvh.name))
        loaded_bvh_file = temp_bvh.name
        load_format = "motive2spine"

    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(
        loaded_bvh_file, format=load_format
    )
    if temp_bvh is not None:
        os.unlink(temp_bvh.name)
        temp_bvh = None
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{load_format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps

    robot_motion_viewer = None
    if visualize:
        robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                motion_fps=motion_fps,
                                                transparent_robot=0,
                                                record_video=args.record_video,
                                                video_path=args.video_path,
                                                # video_width=2080,
                                                # video_height=1170
                                                )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)

        if args.save_path is not None:
            qpos_list.append(qpos)

        if visualize:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   

    if args.save_path is not None:
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
