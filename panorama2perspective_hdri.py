# Reference Code: https://github.com/haruishi43/equilib
# Code Maintainer: Myeong Gyu LEE

import sys, os, random, json, argparse, imageio, cv2
import more_itertools as mit
import numpy as np
from multiprocessing import Process
from equilib import equi2pers
from tqdm import tqdm

def fix_seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def percentage_match(mainvalue,comparevalue):
    if mainvalue >= comparevalue:
        matched_less = mainvalue - comparevalue
        no_percentage_matched = 100 - matched_less*100.0/mainvalue
        no_percentage_matched = str(no_percentage_matched) + ' %'
        return no_percentage_matched 

def split_dict(input_dict: dict, chunk_count: int) -> list:
    list_len: int = len(input_dict)
    return [dict(list(input_dict.items())[i * list_len // chunk_count:(i + 1) * list_len // chunk_count])
        for i in range(chunk_count)]

def read_hdr(equirectangular_hdri_path):
    equi_img = cv2.imread(equirectangular_hdri_path, flags=cv2.IMREAD_ANYDEPTH)
    equi_img = np.transpose(equi_img, (2, 0, 1))
    
    return equi_img # return to torch tensor channel order

def randomCamCoordSampler(range, interval, count):
    return np.sort(random.sample(list(mit.numeric_range(range['min'], range['max'], interval)), count))

def randomPatchCoordSampler(args):
    x_start = random.randint(0, args.w_pers - args.patch_res) # 가로 시작점
    y_start = random.randint(0, args.h_pers - args.patch_res) # 세로 시작점
    
    return x_start, y_start

def randomCoordGrabber(args):
    random_coords = dict()

    for viewport_index in range(args.viewpoint_count):
        coord_range = {'min':0, 'max':np.pi*2}
        random_camera_coords = randomCamCoordSampler(range=coord_range, interval=0.5, count=3)
        
        patch_coords = dict()

        for patch_index in range(args.patch_count):
            x_start, y_start = randomPatchCoordSampler(args)
            patch_coords.update({'patch_'+str(patch_index):{'start_x':x_start, 'start_y':y_start, 'end_x':x_start+args.patch_res, 'end_y':y_start+args.patch_res}})
            
        if args.fix_roll == True:
            random_coords.update({'viewpoint_'+str(viewport_index):{'roll': 0, 'pitch': random_camera_coords[1], 'yaw': random_camera_coords[-1], 'patch_coords':patch_coords}})
        elif args.fix_pitch == True:
            random_coords.update({'viewpoint_'+str(viewport_index):{'roll': random_camera_coords[0], 'pitch': 0, 'yaw': random_camera_coords[-1], 'patch_coords':patch_coords}})
        elif args.fix_yaw == True:
            random_coords.update({'viewpoint_'+str(viewport_index):{'roll': random_camera_coords[0], 'pitch': random_camera_coords[1], 'yaw': 0, 'patch_coords':patch_coords}})
        elif args.fix_roll == False and args.fix_pitch == False and args.fix_yaw == False:
            random_coords.update({'viewpoint_'+str(viewport_index):{'roll': random_camera_coords[0], 'pitch': random_camera_coords[1], 'yaw': random_camera_coords[-1], 'patch_coords':patch_coords}})
        else:
            print('Please check fix_ arguments. Only one axis could be fixed.')
            raise NotImplementedError
    
    return random_coords

def pers_converter(args, chunk_hdr_randomcoords):
    PID = os.getpid()
    if args.sample_mode == 'both':
        print('Both perspective and square patch sampling will be implemented soon.')
        raise NotImplementedError
    elif args.sample_mode == 'patchonly':
        hdr_save_dir = os.path.join(args.save_path+'/HDR/')            
        ldr_save_dir = os.path.join(args.save_path+'/LDR/')
        os.makedirs(hdr_save_dir, exist_ok=True)
        os.makedirs(ldr_save_dir, exist_ok=True)
        
        t_filename = tqdm(iterable=chunk_hdr_randomcoords.items(), total=len(chunk_hdr_randomcoords))
        for file_index, (filename, random_coords) in enumerate(t_filename):
            t_filename.set_postfix({'Current file':filename, 'PID':PID})
            t_view = tqdm(iterable=random_coords.items(), total=len(random_coords), leave=False)
            eq_img = read_hdr(os.path.join(args.hdri_path, filename))

            for viewID, random_view_coords in t_view:
                view_count = int(viewID.split('_')[-1])
                t_view.set_postfix({'Current viewpoint ID':view_count, 'PID':PID})
                rot = {'roll': random_view_coords['roll'], 'pitch': random_view_coords['pitch'], 'yaw': random_view_coords['yaw']}
                pers_img = equi2pers(
                    equi=eq_img,
                    rot=rot,
                    w_pers=args.w_pers,
                    h_pers=args.h_pers,
                    fov_x=args.fov,
                    skew=args.skew,
                    sampling_method="default",
                    mode="bilinear",)
                pers_img = np.transpose(pers_img, (1, 2, 0))

                t_patch = tqdm(iterable=random_view_coords['patch_coords'].items(), total=len(random_view_coords['patch_coords'].items()), leave=False)
                for patchID, random_patch_coords in t_patch:
                    patch_count = int(patchID.split('_')[-1])
                    t_patch.set_postfix({'Current patch ID':patch_count, 'PID':PID})

                    # Get 1:1 perspective patch image
                    pers_hdr = pers_img[random_patch_coords['start_y']:random_patch_coords['end_y'], random_patch_coords['start_x']:random_patch_coords['end_x'],:].astype('float32')

                    # Write perspective captured HDR Image
                    cv2.imwrite(os.path.join(hdr_save_dir, filename+'-view{:04d}_patch{:04d}_hdr.hdr'.format(view_count, patch_count)), pers_hdr)

                    # Simply Clamp values to 0-1 range.
                    ldr = np.clip(pers_hdr, 0, 1)

                    # Color space conversion
                    ldr = ldr**(1/2.2)

                    # 0-255 Remapping for Bit-depth Conversion
                    ldr = 255.0 * ldr
                    ldr = ldr.astype('uint8')

                    # Write Tone-mapped LDR Image
                    filename = filename.replace('.hdr', '')
                    dst = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(ldr_save_dir, filename+'-view{:04d}_patch{:04d}_ldr.png'.format(view_count, patch_count)), ldr)

            if args.change_pname == True:
                setproctitle.setproctitle('HDRI dataset preprocessor...{} of {}({})% done.'.format(file_index+1, len(chunk_hdr_randomcoords), percentage_match(file_index+1, len(chunk_hdr_randomcoords))))

    elif args.sample_mode == 'personly':
        print('Perspective sampling only will be implemented soon.')
        raise NotImplementedError
    else:
        print('Please check sample_mode args.')
        raise NotImplementedError

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()

    # Set Path
    '''
    - hdri_path: 입력 HDR 이미지들이 저장되어 있는 경로를 지정합니다. 
    '''
    parser.add_argument('--hdri_path', type=str, default='/home/datasets/HDRIHaven_subset', help='Set source panorama hdri path')
    parser.add_argument('--save_path', type=str, default='/home/project_src/debug/HDRIDataset_debug', help='Set target image path')
    
    # Set Camera Parameters
    '''
    - w_pers: Output HDR, LDR 이미지의 가로 해상도를 지정합니다. 
    - h_pers: Output HDR, LDR 이미지의 세로 해상도를 지정합니다. 
    - fov: 캡쳐되는 뷰포인트 영역의 시야각을 지정합니다. 
    - skew: 캡쳐되는 뷰포인트 영역의 기울기 정도를 지정합니다. 
    - fix_roll: 카메라의 X축 회전을 고정하고 랜덤 샘플링할지 여부를 결정합니다. 
    - fix_pitch: 카메라의 Y축 회전을 고정하고 랜덤 샘플링할지 여부를 결정합니다. 
    - fix_yaw: 카메라의 Z축 회전을 고정하고 랜덤 샘플링할지 여부를 결정합니다. 
      (Right-handed rule XYZ global coordinate system, x-axis faces forward and z-axis faces up.)
    '''
    parser.add_argument('--w_pers', type=int, default=3840, help='Set width length of output image')
    parser.add_argument('--h_pers', type=int, default=2160, help='Set height length of output image')
    parser.add_argument('--fov', type=float, default=90.0, help='Set perspective image FOV of x-axis')
    parser.add_argument('--skew', type=float, default=0.0, help='Set skew intrinsic parameter')
    parser.add_argument('--fix_roll', type=bool, default=False, help='Fix x axis rotation')
    parser.add_argument('--fix_pitch', type=bool, default=True, help='Fix y axis rotation(vertical)')
    parser.add_argument('--fix_yaw', type=bool, default=False, help='Fix z axis rotation(horizontal)')
    
    # Set Sample Count
    '''
    - viewpoint_count: 뷰포인트의 개수를 지정합니다. 
    - patch_count: 캡쳐된 뷰포인트에서 샘플링할 1:1 사이즈 패치 개수를 지정합니다. 
    - patch_res: 1:1 사이즈 패치의 해상도를 지정합니다. 
    '''
    parser.add_argument('--sample_mode', type=str, default='patchonly', choices=['personly', 'patchonly', 'both', 'jsononly'], help='Set sampling mode')
    parser.add_argument('--seed', type=int, default=2021, help='Set seed number')
    parser.add_argument('--viewpoint_count', type=int, default=6, help='Set viewpoint count')
    parser.add_argument('--patch_count', type=int, default=5, help='Set patch count')
    parser.add_argument('--patch_res', type=int, default=512, help='Set patch resolution')   

    # Set process count for multiprocessing
    parser.add_argument('--num_proc', type=int, default=64, help='Set number of process')
    parser.add_argument('--change_pname', type=bool, default=True, help='Change process name while processing.')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.change_pname == True:
        import setproctitle
        setproctitle.setproctitle('Panorama2Perspective HDRI Dataset Preprocessor - Initializing...')
    
    procs = [] # Process List
    fix_seed_everything(args.seed)
    original_paths = os.listdir(args.hdri_path)
    num_slice = int(len(original_paths)/args.num_proc) + 1
    
    print("Constructing random sampled coordinate sets...")
    hdr_randomcoords = dict()
    for index, filename in enumerate(original_paths):
        hdr_randomcoords.update({filename:randomCoordGrabber(args)})

    # Save sampled random coordinates to *.json
    with open(os.path.join(args.save_path, 'hdr_randomcoords.json'),'w') as f:
        json.dump(hdr_randomcoords, f)

    if args.sample_mode == 'jsononly':
        print('Done saving json file.')
        sys.exit(0)
    
    separated_hdr_randomcoords = split_dict(input_dict=hdr_randomcoords, chunk_count=num_slice)
    print("Done!\nPanorama HDRI to {}*{} Tone-mapped LDR image Processing in progress with {} process...".format(args.patch_res, args.patch_res, args.num_proc))

    for index, chunk_hdr_randomcoords in enumerate(separated_hdr_randomcoords):
        os.makedirs(args.save_path, exist_ok=True)
        proc = Process(target=pers_converter, args=(args, chunk_hdr_randomcoords))
        procs.append(proc)
        proc.start()